"""
34_evaluate_g_v2.py

자동 평가기 v2 — 항목별로 채점 방식을 분리한다.

v1 이 실패한 이유
---------------
v1 은 5개 항목을 한 번의 LLM 호출로 물었고, 5개 문항 17개 판정칸을
놓쳤다. 환각 항목의 kappa 는 0.00, 즉 우연히 맞춘 것과 구별되지 않았다.

놓친 5건의 초안 근거를 보면 실패 구조가 하나다.

    [03] "답변 내용이 근거와 일치하며 누락 없음"
    [23] "답변 내용이 근거와 일치하며 누락 없음"
    [24] "답변이 근거와 일치하며 미확인 신고도 적절함"

평가기는 답변-근거 축만 본다. [23] 에서 답변은 주택청약예금 근거에
충실했고, 근거 대비로는 실제로 일치한다. 문제는 그 근거가 질문과 다른
상품이라는 것인데, 질문-근거 축은 아예 보지 않았다.

프롬프트를 고치는 방식은 이미 세 번 실패했다. 생성기, 근거 검증기,
평가기 모두 같은 지시를 받고도 상품 불일치를 통과시켰다. 네 번째로
같은 방법을 쓰지 않는다.

v2 의 채점 방식
-------------
    환각        문장 단위 LLM. 답변을 쪼개 각 문장의 근거 유무를 묻는다.
    상품일치    LLM 없음. 메타데이터 대조로 결정적으로 판정한다.
    수치정확    통합 LLM 유지. v1 에서 kappa 0.73 으로 유일하게 쓸 만했다.
    미확인신고  분리 호출.
    실무활용    분리 호출.
    오거절      LLM 없음. 정상 답변 문항이 거절되면 계산으로 잡힌다.

왜 환각을 문장 단위로 쪼개는가
--------------------------
답변 전체를 한 번에 보면 "대체로 일치"로 뭉개진다. [11] 의
"계약이 바로 체결되지 않고" 는 나머지 문장이 정확했기 때문에 묻혔다.
문장별로 물으면 그 한 구절만 걸린다.

부수 효과로 각 문장이 어느 근거를 참조하는지 알 수 있고, 서로 다른
상품 문서를 참조하는 답변([03] 의 문서 혼합)이 드러난다.

왜 상품일치에서 LLM 을 빼는가
--------------------------
query_analyzer 가 뽑은 상품명과 근거의 문서명을 대조하면 된다.
게이트로 쓸 때는 과잉 거절이 심해 연결하지 않았지만, 평가 신호로는
사정이 다르다. 판정이 틀려도 시스템이 거절하지 않으므로 위험이 없고,
사람 판정과 대조해 정확도를 잴 수 있다.

오거절을 분리하는 이유
------------------
[14] 는 답변 가능한 질문이 no_evidence 로 거절된 사례다. 생성 품질이
아니라 검색·거절 단계의 문제인데 v1 은 이를 미확인신고 항목에 욱여넣었고,
그것이 해당 항목 kappa 0.17 의 원인 중 하나다. 별도 지표로 뺀다.

사용법
-----
    python scripts/34_evaluate_g_v2.py
    python scripts/34_evaluate_g_v2.py --refresh    # LLM 캐시 무시
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.product_match import (  # noqa: E402
    ProductMatcher,
    judge_product_match,
)
from finguide_rag.generation.query_analyzer import QueryAnalyzer  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
REPORT_DIR = PROJECT_ROOT / "reports"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_eval_answers.json"
JUDGE_CACHE = PROJECT_ROOT / "data" / "interim" / "g_judge_v2_cache.json"
QA_CACHE = PROJECT_ROOT / "data" / "interim" / "query_analysis_cache.json"
CATALOG = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"
RECHECK_MD = EVAL_DIR / "g_eval_recheck.md"

CRITERIA = ["환각", "상품일치", "수치정확", "미확인신고", "실무활용"]

# 자동 평가 대상.
#
# 미확인신고와 실무활용은 제외한다. v2 측정에서 두 항목의 kappa 는 각각
# 0.05, 0.02 로 우연 수준이었다. 채점 방식을 분리해도 개선되지 않았다.
#
# 원인은 두 항목이 본질적으로 주관적이라는 데 있다. "직원이 이 답변으로
# 안내할 수 있는가"는 판단자의 실무 기준에 따라 달라지고, 사람 검수에서도
# 이 둘의 fail 이 다른 항목보다 두 배 이상 많았다(9건, 8건).
#
# 일치하지 않는 판정을 자동 지표로 보고하면 수치가 실제보다 좋아 보인다.
# 자동화하지 못하는 항목은 자동화하지 않는다고 명시하는 편이 정직하다.
# 두 항목은 사람 검수 전용으로 남긴다.
AUTOMATED = ["환각", "상품일치", "수치정확"]
MANUAL_ONLY = ["미확인신고", "실무활용"]
VALID = {"pass", "fail", "na"}

ROW = re.compile(
    r"^\|\s*(\d{1,2})\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|(.*)\|\s*$",
    re.M,
)

# 한국어 종결어미 뒤에서 자른다. 숫자 뒤 마침표(1. 2.)는 자르지 않는다.
SENT_SPLIT = re.compile(r"(?<=[다요음])\.\s+|(?<=[다요음])\.$|\n+")


# ==================================================================
# 유틸
# ==================================================================


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENT_SPLIT.split(text or "") if s and s.strip()]
    return [p if p.endswith(("다", "요", "음", ".")) else p for p in parts if len(p) > 5]


def load_human(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in ROW.findall(path.read_text(encoding="utf-8")):
        verdicts = [v.strip().lower() for v in m[2:7]]
        if not all(v in VALID for v in verdicts):
            continue
        out[f"{int(m[0].strip()):02d}"] = {
            "kind": m[1].strip(), "memo": m[7].strip(),
            **dict(zip(CRITERIA, verdicts)),
        }
    return out


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    labels = {x for p in pairs for x in p}
    pe = sum((sum(1 for a, _ in pairs if a == l) / n)
             * (sum(1 for _, b in pairs if b == l) / n) for l in labels)
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


class Judge:
    """LLM 호출 + 디스크 캐시.

    캐시 키에 과제명과 입력 전체를 넣는다. 프롬프트를 고치면 키가
    달라지므로 자동으로 새로 호출된다.
    """

    def __init__(self, client, model: str, path: Path, refresh: bool):
        self.client, self.model, self.path = client, model, path
        self.cache: dict = {}
        if path.exists() and not refresh:
            try:
                self.cache = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self.calls = 0
        self.tokens = 0

    def ask(self, task: str, system: str, user: str) -> dict:
        key = hashlib.sha256(
            f"{self.model}|{task}|{system}|{user}".encode()).hexdigest()[:24]
        if key in self.cache:
            return self.cache[key]
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0, max_tokens=250,
                response_format={"type": "json_object"},
            )
            data = json.loads(r.choices[0].message.content)
            self.tokens += r.usage.prompt_tokens + r.usage.completion_tokens
        except Exception as exc:
            data = {"_error": str(exc)[:80]}
        self.calls += 1
        self.cache[key] = data
        return data

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2),
                             encoding="utf-8")


# ==================================================================
# 프롬프트
# ==================================================================

SENT_SYSTEM = """당신은 은행 문서 RAG 답변의 근거 검증기입니다.
문장 하나와 근거 목록을 받아, 그 문장이 근거로 뒷받침되는지만 판정합니다.

가장 흔한 오판은 "주제가 같으면 뒷받침된 것으로 보는 것"입니다.
주제가 같아도 단정의 강도나 시점이 근거를 넘어서면 뒷받침되지 않은 것입니다.

supported=false 로 판정해야 하는 예:

  근거: "은행은 이용계약을 해지할 수 있다"
  문장: "계약이 바로 체결되지 않고 해지됩니다"
  -> false. 근거는 해지 가능성만 말하고, 체결 여부나 즉시성은 없다.

  근거: "만 14세 이상 본인 계좌 보유 고객이 이용할 수 있다"
  문장: "미성년자도 보호자 동의가 있으면 이용할 수 있습니다"
  -> false. 보호자 동의라는 조건은 근거에 없다.

  근거: "6개월 이상 경과 시 차등율을 적용한다"
  문장: "6개월 이상이면 90%가 적용됩니다"
  -> false. 구체 수치가 근거와 다르거나 근거에 없다.

supported=true 조건:
- 문장의 내용이 근거에 명시되어 있다.
- 표현이 달라도 의미와 단정의 강도가 같으면 뒷받침된 것으로 봅니다.

supported=false 조건:
- 근거에 없는 사실을 말한다.
- 근거보다 강하게 단정한다. ("할 수 있다" -> "한다", "해지될 수 있다" -> "해지된다")
- 근거에 없는 시점, 즉시성, 순서, 조건, 예외를 덧붙인다.
- 근거의 수치나 기간을 바꿨다.
- 서로 다른 문서의 규정을 합쳐 근거에 없는 규칙을 만든다.

evidence 에는 그 문장을 뒷받침하는 근거 번호를 모두 적으십시오.
문장이 두 개 이상의 근거를 합쳐야 성립한다면 그 번호를 모두 적으십시오.
뒷받침하는 근거가 없으면 빈 배열입니다.

JSON으로만 답하십시오:
{"supported": true, "evidence": [1], "why": "판정 이유 30자 이내"}"""

NUM_SYSTEM = """당신은 은행 문서 RAG 답변의 수치 검증기입니다.
답변에 나오는 금리, 기간, 금액, 비율, 횟수, 나이가 근거와 일치하는지만 봅니다.

pass: 모든 수치가 근거와 일치한다.
fail: 하나라도 근거와 다르거나, 근거에 없는 수치를 만들어냈다.
na: 답변에 수치가 없다.

JSON으로만 답하십시오:
{"verdict": "pass", "why": "30자 이내"}"""

NF_SYSTEM = """당신은 은행 문서 RAG 답변의 미확인 항목 신고 검증기입니다.
질문이 물었으나 근거로 답할 수 없는 부분을 답변이 제대로 밝혔는지만 봅니다.

pass: 근거로 답할 수 없는 부분을 답변이나 미확인 항목에 밝혔다.
      또는 질문 전부가 근거로 답변 가능해 밝힐 것이 없다.
fail: 근거에 없는 부분을 밝히지 않고 답변한 것처럼 넘어갔다.

JSON으로만 답하십시오:
{"verdict": "pass", "why": "30자 이내"}"""

USE_SYSTEM = """당신은 은행 영업점 직원용 답변의 실무 활용도 평가기입니다.
직원이 이 답변을 그대로 고객 안내에 쓸 수 있는지만 봅니다.

pass: 질문에 답하고 있고, 조건이나 예외가 필요하면 함께 담겨 있다.
fail: 질문에 답하지 못했거나, 그대로 안내하면 고객이 오해할 수 있다.
      근거가 질문과 다른 상품 기준인데 그대로 안내하는 경우도 fail 입니다.

JSON으로만 답하십시오:
{"verdict": "pass", "why": "30자 이내"}"""


def ev_block(evidences: list[dict], limit: int = 700) -> str:
    return "\n\n".join(
        f"[{e['rank']}] {e['citation']}\n{e['text'][:limit]}" for e in evidences
    ) or "(근거 없음)"


# ==================================================================
# 항목별 채점
# ==================================================================


def score_hallucination(judge: Judge, item: dict) -> tuple[str, list[dict]]:
    """문장 단위 근거 검증."""
    sents = split_sentences(item["answer"])
    if not sents or not item["evidences"]:
        return "na", []

    block = ev_block(item["evidences"])
    details = []
    for s in sents:
        d = judge.ask("sentence", SENT_SYSTEM,
                      f"[문장]\n{s}\n\n[근거]\n{block}")
        details.append({
            "sentence": s,
            "supported": bool(d.get("supported", True)),
            "evidence": d.get("evidence", []) or [],
            "why": str(d.get("why", ""))[:40],
            "error": d.get("_error", ""),
        })
    bad = [d for d in details if not d["supported"]]
    return ("fail" if bad else "pass"), details


def score_product_match(item, analysis, sent_details, matcher):
    """상품 일치 판정을 product_match 모듈에 위임한다.

    판정 로직을 평가 스크립트마다 두면 한쪽만 고쳤을 때 25건에서 검증한
    결과가 96건에 적용된다는 보장이 깨진다. 한 곳에만 둔다.
    """
    evs = item["evidences"]
    by_rank = {e["rank"]: e for e in evs}
    cited = {i for d in sent_details for i in d["evidence"]
             if isinstance(i, int) and i in by_rank}
    return judge_product_match(matcher, evs, analysis.extracted_product or "", cited)


def score_simple(judge: Judge, task: str, system: str, item: dict) -> tuple[str, str]:
    user = (f"[질문]\n{item['question']}\n\n"
            f"[답변]\n{item['answer']}\n\n"
            f"[미확인 항목]\n{', '.join(item['not_found']) or '(없음)'}\n\n"
            f"[근거]\n{ev_block(item['evidences'])}")
    d = judge.ask(task, system, user)
    v = str(d.get("verdict", "")).lower()
    return (v if v in VALID else "na"), str(d.get("why", ""))[:40]


# ==================================================================
# 메인
# ==================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="data/eval/g_eval_final.md")
    ap.add_argument("--llm-model", default="gpt-4.1-mini")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 76)
    print("자동 평가기 v2 — 항목별 채점 방식 분리")
    print("=" * 76)

    if not ANSWERS.exists():
        sys.exit(f"{ANSWERS} 없음. 32번을 먼저 실행하세요.")
    items = json.loads(ANSWERS.read_text(encoding="utf-8"))

    review = PROJECT_ROOT / args.review
    human = load_human(review) if review.exists() else {}
    print(f"  문항 {len(items)}건 / 사람 판정 {len(human)}건")

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    client = OpenAI()

    judge = Judge(client, args.llm_model, JUDGE_CACHE, args.refresh)
    analyzer = QueryAnalyzer(client, args.llm_model, QA_CACHE)
    matcher = ProductMatcher.from_catalog(CATALOG)

    print(f"\n  채점 중...")
    results = []
    for n, it in enumerate(items, 1):
        refused = it["decision"] == "refuse"
        analysis = analyzer.analyze(it["question"])

        if refused:
            # 거절 응답에는 답변 본문이 없다. 생성 품질 항목은 na 로 둔다.
            verdicts = {c: "na" for c in AUTOMATED}
            verdicts.update({c: "manual" for c in MANUAL_ONLY})
            sent_details = []
            why = {c: "거절 응답 — 답변 본문 없음" for c in CRITERIA}
        else:
            hall, sent_details = score_hallucination(judge, it)
            prod, prod_why = score_product_match(it, analysis, sent_details, matcher)
            num, num_why = score_simple(judge, "number", NUM_SYSTEM, it)
            verdicts = {"환각": hall, "상품일치": prod, "수치정확": num,
                        "미확인신고": "manual", "실무활용": "manual"}
            why = {"환각": f"미근거 문장 {sum(1 for d in sent_details if not d['supported'])}개",
                   "상품일치": prod_why, "수치정확": num_why,
                   "미확인신고": "자동 평가 제외 — 사람 검수 전용",
                   "실무활용": "자동 평가 제외 — 사람 검수 전용"}

        # 오거절 — LLM 없이 계산으로 판정
        false_refusal = refused and it["group"] == "정상 답변"

        results.append({
            "id": it["id"], "group": it["group"], "question": it["question"],
            "decision": it["decision"], "stage": it["stage"],
            "false_refusal": false_refusal,
            "extracted_product": analysis.extracted_product or "",
            "unsupported": sum(1 for d in sent_details if not d["supported"]),
            "n_sentences": len(sent_details),
            **{f"v2_{c}": verdicts[c] for c in CRITERIA},
            **{f"why_{k}": v for k, v in why.items()},
            "_sent": sent_details,
        })
        if n % 5 == 0:
            print(f"    {n}/{len(items)}")
            judge.flush()

    judge.flush()
    analyzer.flush()
    print(f"  LLM 호출 {judge.calls}건 / {judge.tokens:,} 토큰 "
          f"(캐시 {len(judge.cache)}건)")

    # ---------- 오거절 ----------
    print("\n" + "=" * 76)
    print("오거절 (false refusal) — 신규 지표")
    print("=" * 76)
    normal = [r for r in results if r["group"] == "정상 답변"]
    fr = [r for r in normal if r["false_refusal"]]
    print(f"  정상 답변 문항 {len(normal)}건 중 거절 {len(fr)}건 "
          f"= {len(fr) / len(normal) if normal else 0:.3f}")
    for r in fr:
        print(f"    [{r['id']}] stage={r['stage']}  {r['question'][:50]}")
    print("\n  이 지표는 LLM 판정이 아니라 계산이다. 생성 품질이 아니라")
    print("  검색·거절 단계의 문제이므로 평가기에게 묻지 않는다.")

    # ---------- 사람 판정 대조 ----------
    #
    # 세 가지 조건으로 나눠 본다. 조건을 섞으면 평가기 성능과 비교 방식의
    # 차이가 구별되지 않는다.
    #
    #   A. 전체 25건
    #   B. 오거절 5건 제외
    #      사람은 오거절을 미확인신고·실무활용에 fail 로 넣었고, v2 는
    #      거절 문항을 na 로 둔다. 오거절을 별도 지표로 뺐기 때문이다.
    #      이 5건은 평가기의 오판이 아니라 라벨링 규약 차이이므로 분리한다.
    #   C. v1 과 같은 조건 (초안이 있던 20건)
    #      v1 은 초안이 있는 20건만 비교했다. 같은 분모로 봐야 v1 대비
    #      개선 여부를 말할 수 있다.
    rows_out = []
    if human:
        blind_ids = {it["id"] for it in items if it["blind"]}
        fr_ids = {r["id"] for r in results if r["false_refusal"]}

        views = [
            ("A. 전체", {r["id"] for r in results}),
            ("B. 오거절 제외", {r["id"] for r in results} - fr_ids),
            ("C. v1과 동일(초안 20건)", {r["id"] for r in results} - blind_ids),
        ]

        for label, ids in views:
            print("\n" + "=" * 76)
            print(f"사람 판정 대조 — {label}  ({len(ids)}건)")
            print("=" * 76)
            print(f"  {'항목':<12}{'일치':>8}{'일치율':>9}{'kappa':>9}"
                  f"{'놓친 실패':>11}{'과잉 탐지':>11}")
            print("  " + "-" * 60)

            total_missed = total_over = 0
            for c in AUTOMATED:
                pairs, missed, over = [], [], []
                for r in results:
                    if r["id"] not in ids:
                        continue
                    h = human.get(r["id"], {}).get(c)
                    v = r[f"v2_{c}"]
                    if h not in VALID or v not in VALID:
                        continue
                    pairs.append((v, h))
                    if h == "fail" and v != "fail":
                        missed.append(r["id"])
                    if v == "fail" and h != "fail":
                        over.append(r["id"])
                if not pairs:
                    continue
                agree = sum(1 for a, b in pairs if a == b)
                total_missed += len(missed)
                total_over += len(over)
                print(f"  {c:<12}{agree:>4}/{len(pairs):<3}"
                      f"{agree / len(pairs):>9.2f}{kappa(pairs):>9.2f}"
                      f"{len(missed):>11}{len(over):>11}")
                rows_out.append({
                    "view": label, "criterion": c, "n": len(pairs),
                    "agreement": round(agree / len(pairs), 3),
                    "kappa": round(kappa(pairs), 3),
                    "missed_fail": len(missed), "over_flag": len(over),
                    "missed_ids": "|".join(missed), "over_ids": "|".join(over),
                })
            print(f"\n  놓친 실패 {total_missed}칸 / 과잉 탐지 {total_over}칸")
            if label.startswith("C"):
                print(f"  v1 동일 조건 기준: 놓친 실패 17칸 / 과잉 탐지 2칸")
                print(f"     (단, v1 은 5개 항목 전부를 자동 채점했다. v3 는 3개만")
                print(f"      채점하므로 칸 수를 직접 비교할 수 없다. 항목별 kappa 로 본다.)")

        print("\n  [자동 평가 제외 항목]")
        for c in MANUAL_ONLY:
            hf = sum(1 for i in human if human[i].get(c) == "fail")
            print(f"    {c:<12} 사람 판정 fail {hf}건 — 자동 채점하지 않음")
        print("    v2 에서 두 항목의 kappa 는 0.05, 0.02 로 우연 수준이었다.")
        print("    판단자의 실무 기준에 따라 달라지는 항목이므로 자동화하지 않고")
        print("    사람 검수 전용으로 남긴다.")

        # v1 이 놓쳤던 문항을 잡았는지
        v1_missed = ["03", "11", "14", "23", "24"]
        print("\n  [v1 이 놓쳤던 문항을 v2 가 잡았는가]")
        for i in v1_missed:
            r = next((x for x in results if x["id"] == i), None)
            if not r:
                continue
            h = human.get(i, {})
            hf = [c for c in CRITERIA if h.get(c) == "fail"]
            vf = [c for c in AUTOMATED if r[f"v2_{c}"] == "fail"]
            caught = set(hf) & set(vf)
            mark = "잡음" if caught else "놓침"
            print(f"    [{i}] {mark:<4} 사람 fail={','.join(hf) or '-'}")
            print(f"          v2 fail={','.join(vf) or '-'}")
            if r["why_상품일치"] if "why_상품일치" in r else "":
                print(f"          상품일치 근거: {r.get('why_상품일치', '')}")

    # ---------- 문장별 근거 인용 진단 ----------
    #
    # 상품일치가 "질문에 상품 언급 없음"으로 na 가 된 문항이 실제로는
    # 서로 다른 상품 문서를 섞어 답한 경우가 있다([03]). 문서 혼합 탐지가
    # 왜 발동하지 않았는지 보려면 문장이 어느 근거를 인용했는지 봐야 한다.
    print("\n" + "=" * 76)
    print("문장별 근거 인용 — 문서 혼합 탐지 진단")
    print("=" * 76)
    for r in results:
        if not r["_sent"]:
            continue
        h = human.get(r["id"], {})
        interesting = (h.get("환각") == "fail" or h.get("상품일치") == "fail"
                       or r["unsupported"] > 0)
        if not interesting:
            continue
        it = next(x for x in items if x["id"] == r["id"])
        by_rank = {e["rank"]: e for e in it["evidences"]}
        print(f"\n  [{r['id']}] {r['question'][:48]}")
        print(f"      v2 상품일치={r['v2_상품일치']} ({r.get('why_상품일치', '')})")
        for e in it["evidences"]:
            print(f"      근거[{e['rank']}] {e.get('doc_type', '?')} "
                  f"{e['citation'][:52]}")
        for d in r["_sent"]:
            cited = d["evidence"] or []
            docs = {by_rank[i]["citation"][:24] for i in cited
                    if isinstance(i, int) and i in by_rank}
            mark = "  " if d["supported"] else "X "
            print(f"      {mark}{str(cited):<10} {d['sentence'][:46]}")
            if len(docs) >= 2:
                print(f"         -> 한 문장이 서로 다른 문서 {len(docs)}건 인용")
        used = {i for d in r["_sent"] for i in d["evidence"] if isinstance(i, int)}
        used_docs = {by_rank[i]["citation"] for i in used if i in by_rank
                     and by_rank[i].get("doc_type") == "설명서"}
        print(f"      답변 전체가 인용한 상품설명서: {len(used_docs)}건")

    # ---------- 재검수 대상 ----------
    rng = random.Random(args.seed)
    prev_missed = [r for r in results if r["id"] in {"03", "11", "14", "23", "24"}]
    pool = [r for r in results if r["id"] not in {"03", "11", "14", "23", "24"}]
    rng.shuffle(pool)
    recheck = prev_missed + pool[:5]

    L = ["# G 평가셋 재검수 — 자동 평가기 v2 검증", "",
         f"생성일: {date.today().isoformat()} / {len(recheck)}건", "",
         "이번에는 초안을 제공하지 않습니다. 1차 검수에서 초안 제시 문항의",
         "fail 비율이 0.25, 백지 문항이 0.43 으로 초안 편향 신호가 확인됐기",
         "때문입니다.", "",
         "각 항목의 `판정:` 뒤를 `pass`, `fail`, `na` 중 하나로 채워 주세요.",
         "", "---", ""]
    for r in recheck:
        L.append(f"## [{r['id']}] {r['group']}")
        L.append("")
        L.append(f"**질문**  {r['question']}")
        L.append("")
        it = next(x for x in items if x["id"] == r["id"])
        L.append(f"**시스템 판정**  `{r['decision']}` (stage={r['stage']})")
        L.append("")
        L.append("**답변**")
        L.append("")
        for line in (it["answer"] or "").split("\n"):
            L.append(f"> {line}")
        L.append("")
        if it["not_found"]:
            L.append("**미확인 항목**  " + ", ".join(it["not_found"]))
            L.append("")
        if it["evidences"]:
            L.append("<details><summary><b>근거 원문 (펼치기)</b></summary>")
            L.append("")
            for e in it["evidences"]:
                L.append(f"**[{e['rank']}] {e['citation']}**")
                L.append("")
                L.append("```")
                L.append(e["text"][:1200])
                L.append("```")
                L.append("")
            L.append("</details>")
            L.append("")
        L.append(f"<!-- ITEM {r['id']} -->")
        for c in CRITERIA:
            L.append(f"- {c} 판정: ?")
        L.append("- 메모: ")
        L.append("")
        L.append("---")
        L.append("")
    RECHECK_MD.write_text("\n".join(L), encoding="utf-8")

    # ---------- 저장 ----------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"g_eval_v2_{date.today().isoformat()}.csv"
    flat = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    sent_out = REPORT_DIR / f"g_eval_v2_sentences_{date.today().isoformat()}.csv"
    rows = [{"id": r["id"], **d} for r in results for d in r["_sent"]]
    if rows:
        with sent_out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n" + "=" * 76)
    print(f"저장 → {out.relative_to(PROJECT_ROOT)}")
    if rows:
        print(f"저장 → {sent_out.relative_to(PROJECT_ROOT)}  (문장 단위 판정)")
    print(f"저장 → {RECHECK_MD.relative_to(PROJECT_ROOT)}  <- 재검수 10건")
    print("=" * 76)


if __name__ == "__main__":
    main()
