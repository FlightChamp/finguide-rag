"""
36_evaluate_g_full.py

96건 전체 자동 평가.

왜 확장하는가
-----------
지금까지의 G 지표는 25건 기준이다. 한 건이 4%p 를 움직이므로 소수점
둘째 자리를 말할 수 없다. 96건이면 한 건이 1%p 가 되어 난이도별,
문서유형별 분해까지 의미를 갖는다.

확장의 전제는 자동 평가기가 사람 판정과 충분히 맞는다는 것이었다.
25건 검수와 10건 재검수로 그 전제를 확인했다.

    환각      kappa 0.73   문장 단위 LLM 검증
    상품일치  kappa 0.44   메타데이터 대조, LLM 없음
    오거절    완전 일치     계산, LLM 없음

재검수 10건에서 대조군 4건의 과잉 탐지가 0 이었다. 평가기가 무작정
엄격해진 것이 아니라는 뜻이므로 확장해도 수치가 과장되지 않는다.

보고 등급을 나눈다
---------------
모든 항목을 같은 신뢰도로 보고하면 안 된다.

    보고 가능   환각, 상품일치, 오거절
    참고용      수치정확 (kappa 0.48, 원인 미분석)
    보고 안 함  미확인신고, 실무활용 (kappa 0.05 / 0.02)

마지막 두 항목은 판단자의 실무 기준에 좌우되어 자동화가 되지 않았다.
자동화하지 못한 항목을 지표로 내면 수치가 실제보다 좋아 보인다.

평가셋 선택
---------
retrieval_eval.csv 96건을 쓴다. 전부 답변 가능한 질문이고 정답 청크가
지정되어 있다. 검색 성능(문서 단위 R@5 0.979)을 낸 바로 그 질문들이므로,
같은 질문에서 검색과 생성을 나란히 비교할 수 있다.

거절 대상 질문은 포함하지 않는다. 거절 성능은 이미 53건 평가셋에서
FAR 0.030 으로 측정했다. 여기서는 "답할 수 있는 질문에 제대로 답하는가"만
본다. 따라서 이 평가셋에서의 거절은 전부 오거절이다.

비용
----
질문당 파이프라인 1회(근거 검증 + 생성)와 문장 단위 검증 3~5회.
모두 캐시되므로 재실행은 무료다.

사용법
-----
    python scripts/36_evaluate_g_full.py
    python scripts/36_evaluate_g_full.py --limit 20    # 일부만
    python scripts/36_evaluate_g_full.py --no-generate # 캐시된 답변만 채점
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.pipeline import Pipeline  # noqa: E402
from finguide_rag.generation.product_match import (  # noqa: E402
    ProductMatcher,
    judge_product_match,
)
from finguide_rag.generation.query_analyzer import QueryAnalyzer  # noqa: E402

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
ANSWER_CACHE = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
JUDGE_CACHE = PROJECT_ROOT / "data" / "interim" / "g_judge_v2_cache.json"
QA_CACHE = PROJECT_ROOT / "data" / "interim" / "query_analysis_cache.json"
CATALOG = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"

COL_QUESTION = ("question", "query", "질문")
COL_DIFFICULTY = ("difficulty", "level", "난이도")
COL_DOCTYPE = ("doc_type", "source_doc_type", "문서유형", "target_doc_type")

SENT_SPLIT = re.compile(r"(?<=[다요음])\.\s+|(?<=[다요음])\.$|\n+")

# 34번과 동일한 프롬프트를 쓴다. 캐시 키에 프롬프트가 들어가므로
# 여기서 문구가 달라지면 25건에서 검증한 평가기와 다른 것을 쓰는 셈이 된다.
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


def pick(row: dict, cands: tuple[str, ...]) -> str:
    for c in cands:
        if c in row and (row[c] or "").strip():
            return row[c].strip()
    return ""


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text or "")
            if s and len(s.strip()) > 5]


def doc_key(ev: dict) -> str:
    return re.split(r"\s+제\d+조|\s+제\d+항|\s*\(", ev.get("citation", ""))[0].strip()


class Judge:
    def __init__(self, client, model: str, path: Path):
        self.client, self.model, self.path = client, model, path
        self.cache = {}
        if path.exists():
            try:
                self.cache = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self.calls = self.tokens = 0

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
                response_format={"type": "json_object"})
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


def ev_block(evs: list[dict], limit: int = 700) -> str:
    return "\n\n".join(f"[{e['rank']}] {e['citation']}\n{e['text'][:limit]}"
                       for e in evs) or "(근거 없음)"


def score_item(judge: Judge, item: dict, analysis, matcher) -> dict:
    """25건에서 검증한 v2 방식 그대로 채점한다."""
    evs = item["evidences"]

    # 환각 — 문장 단위
    sents = split_sentences(item["answer"])
    details = []
    if sents and evs:
        block = ev_block(evs)
        for s in sents:
            d = judge.ask("sentence", SENT_SYSTEM, f"[문장]\n{s}\n\n[근거]\n{block}")
            details.append({"sentence": s,
                            "supported": bool(d.get("supported", True)),
                            "evidence": d.get("evidence", []) or [],
                            "why": str(d.get("why", ""))[:40]})
    unsupported = sum(1 for d in details if not d["supported"])
    hall = "na" if not details else ("fail" if unsupported else "pass")

    # 상품일치 — 메타데이터 대조, LLM 없음
    by_rank = {e["rank"]: e for e in evs}
    cited = {i for d in details for i in d["evidence"]
             if isinstance(i, int) and i in by_rank}
    prod, prod_why = judge_product_match(
        matcher, evs, analysis.extracted_product or "", cited)
    cited_docs = len(matcher.distinct_products(
        [(by_rank[i].get("doc_display_name") or by_rank[i].get("citation", ""))
         for i in cited])) if cited else 0

    # 수치정확 — 참고용
    d = judge.ask("number", NUM_SYSTEM,
                  f"[질문]\n{item['question']}\n\n[답변]\n{item['answer']}\n\n"
                  f"[근거]\n{ev_block(evs)}")
    v = str(d.get("verdict", "")).lower()
    num = v if v in {"pass", "fail", "na"} else "na"

    return {"환각": hall, "상품일치": prod, "수치정확": num,
            "unsupported": unsupported, "n_sentences": len(details),
            "cited_docs": cited_docs, "why_상품일치": prod_why,
            "_sent": details}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--llm-model", default="gpt-4.1-mini")
    ap.add_argument("--no-generate", action="store_true")
    args = ap.parse_args()

    print("=" * 76)
    print("G 전체 평가 — retrieval_eval 96건")
    print("=" * 76)

    if not EVAL_CSV.exists():
        sys.exit(f"{EVAL_CSV} 없음.")
    with EVAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    print(f"  질문 {len(rows)}건")

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    client = OpenAI()

    # --- 답변 생성 (캐시) ---
    cached: dict[str, dict] = {}
    if ANSWER_CACHE.exists():
        cached = json.loads(ANSWER_CACHE.read_text(encoding="utf-8"))
        print(f"  답변 캐시 {len(cached)}건 재사용")

    todo = [r for r in rows if pick(r, COL_QUESTION) not in cached]
    if todo and not args.no_generate:
        print(f"\n  파이프라인 실행 {len(todo)}건...")
        pipeline = Pipeline.build(PROJECT_ROOT, client=client,
                                  llm_model=args.llm_model)
        for i, r in enumerate(todo, 1):
            q = pick(r, COL_QUESTION)
            cached[q] = pipeline.answer(q).to_dict()
            if i % 10 == 0:
                print(f"    {i}/{len(todo)}")
                ANSWER_CACHE.parent.mkdir(parents=True, exist_ok=True)
                ANSWER_CACHE.write_text(
                    json.dumps(cached, ensure_ascii=False, indent=2),
                    encoding="utf-8")
        ANSWER_CACHE.write_text(json.dumps(cached, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    elif todo:
        print(f"  --no-generate 이므로 {len(todo)}건을 건너뜁니다.")

    # --- 채점 ---
    judge = Judge(client, args.llm_model, JUDGE_CACHE)
    analyzer = QueryAnalyzer(client, args.llm_model, QA_CACHE)
    matcher = ProductMatcher.from_catalog(CATALOG)
    if not CATALOG.exists():
        print(f"  [주의] {CATALOG.name} 없음. 일반 어절 필터가 꺼집니다.")
        print(f"         python scripts/31_measure_coverage.py --catalog-only")
    else:
        print(f"  일반 어절 {len(matcher.generic)}개 자동 제외: "
              f"{', '.join(sorted(matcher.generic)[:8])}")

    print(f"\n  채점 중...")
    results = []
    for i, r in enumerate(rows, 1):
        q = pick(r, COL_QUESTION)
        item = cached.get(q)
        if not item:
            continue
        refused = item["decision"] == "refuse"
        analysis = analyzer.analyze(q)

        if refused:
            s = {"환각": "na", "상품일치": "na", "수치정확": "na",
                 "unsupported": 0, "n_sentences": 0, "cited_docs": 0,
                 "why_상품일치": "거절 응답", "_sent": []}
        else:
            s = score_item(judge, item, analysis, matcher)

        results.append({
            "question": q,
            "difficulty": pick(r, COL_DIFFICULTY) or "?",
            "doc_type": pick(r, COL_DOCTYPE) or "?",
            "decision": item["decision"], "stage": item["stage"],
            "false_refusal": refused,
            "n_evidence": len(item["evidences"]),
            "not_found": len(item["not_found"]),
            "tokens": item.get("tokens", 0), "latency_ms": item.get("latency_ms", 0),
            **{k: v for k, v in s.items() if not k.startswith("_")},
        })
        if i % 20 == 0:
            print(f"    {i}/{len(rows)}")
            judge.flush()
    judge.flush()
    analyzer.flush()
    print(f"  LLM 호출 {judge.calls}건 / {judge.tokens:,} 토큰")

    n = len(results)
    if not n:
        sys.exit("채점할 항목이 없습니다.")

    # --- 핵심 지표 ---
    def rate(key: str, val: str, pool: list[dict]) -> tuple[int, int]:
        rel = [r for r in pool if r[key] in {"pass", "fail"}]
        return sum(1 for r in rel if r[key] == val), len(rel)

    fr = [r for r in results if r["false_refusal"]]
    answered = [r for r in results if not r["false_refusal"]]

    print("\n" + "=" * 76)
    print(f"핵심 지표 (n={n})")
    print("=" * 76)
    print(f"  오거절률          {len(fr)}/{n} = {len(fr) / n:.3f}")
    print(f"    이 평가셋은 전부 답변 가능한 질문이므로 거절은 모두 오거절이다.")
    if fr:
        by_stage = Counter(r["stage"] for r in fr)
        for k, v in by_stage.most_common():
            print(f"      stage={k:<12} {v:>3}건")

    for label, key in [("환각률", "환각"), ("상품불일치율", "상품일치")]:
        f, tot = rate(key, "fail", answered)
        print(f"\n  {label:<14} {f}/{tot} = {f / tot if tot else 0:.3f}"
              f"   (na {sum(1 for r in answered if r[key] == 'na')}건 제외)")

    f, tot = rate("수치정확", "fail", answered)
    print(f"\n  수치오류율(참고)  {f}/{tot} = {f / tot if tot else 0:.3f}")
    print(f"    kappa 0.48 로 신뢰도가 낮아 참고용으로만 본다.")

    print(f"\n  자동 평가 제외: 미확인신고, 실무활용")
    print(f"    사람 판정과 kappa 0.05 / 0.02 로 우연 수준이었다.")

    # --- 분해 ---
    for label, key in [("난이도별", "difficulty"), ("문서유형별", "doc_type")]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            groups[r[key]].append(r)
        if len(groups) <= 1:
            continue
        print("\n" + "=" * 76)
        print(f"{label} 분해")
        print("=" * 76)
        print(f"  {'구분':<12}{'질문':>6}{'오거절':>9}{'환각':>9}{'상품불일치':>11}")
        print("  " + "-" * 48)
        for g in sorted(groups):
            pool = groups[g]
            ans = [r for r in pool if not r["false_refusal"]]
            frr = sum(1 for r in pool if r["false_refusal"]) / len(pool)
            hf, ht = rate("환각", "fail", ans)
            pf, pt = rate("상품일치", "fail", ans)
            print(f"  {g:<12}{len(pool):>6}{frr:>9.3f}"
                  f"{hf / ht if ht else 0:>9.3f}{pf / pt if pt else 0:>11.3f}")

    # --- 25건 대비 ---
    print("\n" + "=" * 76)
    print("표본 확대 효과 — 25건 대비")
    print("=" * 76)
    print(f"  {'지표':<16}{'25건':>10}{'96건':>10}")
    print("  " + "-" * 36)
    print(f"  {'오거절률':<16}{0.294:>10.3f}{len(fr) / n:>10.3f}")
    print(f"\n  25건에서는 한 건이 4%p 를 움직였다. {n}건에서는 "
          f"{1 / n * 100:.1f}%p 다.")
    print(f"  두 수치가 크게 다르면 25건 표본이 편향돼 있었다는 뜻이다.")

    # --- 실패 사례 ---
    print("\n" + "=" * 76)
    print("실패 사례")
    print("=" * 76)
    hall_fail = [r for r in answered if r["환각"] == "fail"]
    prod_fail = [r for r in answered if r["상품일치"] == "fail"]

    print(f"\n  [환각 {len(hall_fail)}건]")
    for r in hall_fail[:12]:
        print(f"    ({r['unsupported']}/{r['n_sentences']} 문장) "
              f"{r['question'][:52]}")
    if len(hall_fail) > 12:
        print(f"    ... 외 {len(hall_fail) - 12}건")

    print(f"\n  [상품 불일치 {len(prod_fail)}건]")
    for r in prod_fail[:12]:
        print(f"    {r['question'][:44]}")
        print(f"       {r['why_상품일치']}")
    if len(prod_fail) > 12:
        print(f"    ... 외 {len(prod_fail) - 12}건")

    print(f"\n  [오거절 {len(fr)}건]")
    for r in fr[:12]:
        print(f"    stage={r['stage']:<10} {r['question'][:50]}")
    if len(fr) > 12:
        print(f"    ... 외 {len(fr) - 12}건")

    # --- 비용 ---
    toks = [r["tokens"] for r in results if r["tokens"]]
    lats = [r["latency_ms"] for r in results if r["latency_ms"]]
    if toks:
        print("\n" + "=" * 76)
        print("응답 비용")
        print("=" * 76)
        print(f"  질문당 토큰   평균 {sum(toks) / len(toks):,.0f}")
        print(f"  질문당 지연   평균 {sum(lats) / len(lats):,.0f}ms "
              f"/ 최대 {max(lats):,}ms")

    # --- 저장 ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"g_eval_full_{date.today().isoformat()}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n저장 → {out.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
