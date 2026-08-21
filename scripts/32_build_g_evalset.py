"""
32_build_g_evalset.py

목적
----
답변 생성(G) 평가셋 25건을 만들고, 사람이 검수할 마크다운을 출력한다.

왜 25건인가
----------
전량 사람 검수는 신뢰도가 높지만 재측정 비용이 사람 시간이다. 프롬프트를
고칠 때마다 두 시간을 다시 쓰면 개선을 주저하게 된다.

그래서 25건은 사람이 판정하고, 그 결과로 자동 평가기의 일치율을 잰다.
일치율이 충분하면 나머지 전체를 자동으로 돌린다. 거절 임계값 탐색에서
LLM 판정을 캐싱해 504조합을 비용 없이 돌렸던 것과 같은 구조다.
사람 판정은 자동 평가기를 검증하는 데 쓰고, 검증된 평가기가 규모를 맡는다.

초안 편향 통제
------------
20건은 LLM 초안 판정을 미리 채워 주고, 5건은 백지로 둔다.
초안이 있으면 "판단"이 "검토"가 되어 시간이 줄지만, 초안에 끌려가는
편향이 생긴다. 백지 5건과 초안 20건에서 사람 판정 분포가 체계적으로
다르면 편향이 있다는 뜻이고, 그 자체가 기록할 가치가 있는 결과다.

평가 항목
--------
    환각       답변이 근거에 없는 내용을 포함하는가
    상품일치   근거가 질문과 다른 상품인데 그렇게 답하는가
    수치정확   금리·기간·금액이 근거와 일치하는가
    미확인신고 확인되지 않은 항목을 not_found 로 제대로 신고했는가
    실무활용   직원이 이 답변으로 고객에게 안내할 수 있는가

상품일치를 따로 두는 이유는 데모 첫 실행에서 잡힌 실패 때문이다.
정기예금 질문에 주택청약예금 근거로 답했는데, 답변 자체는 근거에
충실했으므로 "환각 여부"만 봤다면 통과했을 사례다.

사용법
-----
    python scripts/32_build_g_evalset.py
    python scripts/32_build_g_evalset.py --no-draft   # 초안 없이 전건 백지
    python scripts/32_build_g_evalset.py --seed 7     # 표본 재추출
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.pipeline import Pipeline  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
REPORT_DIR = PROJECT_ROOT / "reports"
CACHE = PROJECT_ROOT / "data" / "interim" / "g_eval_answers.json"
REVIEW_MD = EVAL_DIR / "g_eval_review.md"

N_ANSWERABLE = 17
N_REFUSAL = 5
N_BOUNDARY = 3
N_BLIND = 5

# 컬럼명이 평가셋마다 다를 수 있으므로 후보를 두고 찾는다.
COL_QUESTION = ("question", "query", "질문")
COL_DIFFICULTY = ("difficulty", "level", "난이도")
COL_DOCTYPE = ("doc_type", "source_doc_type", "문서유형", "target_doc_type")

# 코퍼스에 없는 상품군 질문. 어느 평가셋에도 없어 직접 작성했다.
# 런타임 규칙이 아니라 평가 데이터이므로 수기 작성이 문제되지 않는다.
SEED_OUT_OF_CORPUS = [
    ("정기예금을 중도해지하면 이자는 어떻게 계산되나요?", "정기예금"),
    ("자유적금 만기 전에 해지하면 우대금리는 어떻게 되나요?", "자유적금"),
    ("파킹통장 한도가 얼마까지인가요?", "파킹통장"),
]

CRITERIA = [
    ("환각", "답변이 근거에 없는 내용을 포함하는가", "pass=없음 / fail=있음"),
    ("상품일치", "근거가 질문과 같은 상품인가", "pass=일치 / fail=불일치 / na=상품무관"),
    ("수치정확", "금리·기간·금액이 근거와 일치하는가", "pass / fail / na=수치없음"),
    ("미확인신고", "확인 안 된 항목을 not_found 로 신고했는가", "pass / fail / na=해당없음"),
    ("실무활용", "직원이 이 답변으로 안내할 수 있는가", "pass / fail"),
]


# ==================================================================
# 로딩
# ==================================================================


def pick(row: dict, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c in row and (row[c] or "").strip():
            return row[c].strip()
    return ""


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_coverage() -> dict[str, dict]:
    """커버리지 측정 결과를 질문 텍스트로 색인한다."""
    files = sorted(REPORT_DIR.glob("coverage_measure_*.csv"))
    if not files:
        return {}
    out = {}
    for r in load_csv(files[-1]):
        q = (r.get("question") or "").strip()
        if q:
            out[q] = r
    return out


# ==================================================================
# 층화 표본
# ==================================================================


def stratified(rows: list[dict], n: int, coverage: dict, rng: random.Random) -> list[dict]:
    """난이도 × 문서유형으로 층을 나눠 고르게 뽑는다.

    커버리지 측정에서 covered 로 나온 질문을 우선한다. unmatched 질문은
    애초에 근거가 없어 생성 품질을 재기 어렵기 때문이다. 다만 경계 표본은
    따로 뽑으므로 여기서 제외해도 무방하다.
    """
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        q = pick(r, COL_QUESTION)
        if not q:
            continue
        cov = coverage.get(q, {})
        if (cov.get("match_status") or "") in {"unmatched", "ambiguous"}:
            continue
        key = (pick(r, COL_DIFFICULTY) or "?", pick(r, COL_DOCTYPE) or "?")
        buckets[key].append(r)

    if not buckets:
        return rng.sample(rows, min(n, len(rows)))

    for b in buckets.values():
        rng.shuffle(b)

    # 층을 돌아가며 하나씩 뽑아 균등하게 만든다.
    keys = sorted(buckets)
    out: list[dict] = []
    i = 0
    while len(out) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        i += 1
    return out


def sample_refusal(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """거절 유형별로 하나씩 뽑는다."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if (r.get("expected") or "").strip() != "refuse":
            continue
        by_type[(r.get("refusal_type") or "?").strip()].append(r)

    for v in by_type.values():
        rng.shuffle(v)

    out = []
    for t in sorted(by_type):
        if len(out) >= n:
            break
        out.append(by_type[t].pop())
    # 부족하면 아무 유형에서나 채운다.
    pool = [r for v in by_type.values() for r in v]
    rng.shuffle(pool)
    while len(out) < n and pool:
        out.append(pool.pop())
    return out


def sample_boundary(coverage: dict, n: int, rng: random.Random) -> list[dict]:
    """커버리지 경계 질문. 기존 평가셋의 unmatched/ambiguous + 신규 작성."""
    out = [{"question": q, "expected": "", "refusal_type": "",
            "_note": f"코퍼스 밖 상품군: {fam}"}
           for q, fam in SEED_OUT_OF_CORPUS[:max(0, n - 1)]]

    pool = [r for r in coverage.values()
            if (r.get("match_status") or "") in {"unmatched", "ambiguous"}]
    rng.shuffle(pool)
    while len(out) < n and pool:
        r = pool.pop()
        out.append({"question": r["question"], "expected": "",
                    "refusal_type": "",
                    "_note": f"매칭={r.get('match_status')} "
                             f"추출={r.get('extracted_product')}"})
    return out[:n]


# ==================================================================
# 초안 판정
# ==================================================================

DRAFT_SYSTEM = """당신은 은행 문서 기반 RAG 시스템의 답변 품질 평가기입니다.
질문, 시스템 답변, 근거 원문을 받아 항목별로 판정합니다.

항목:
- 환각: 답변에 근거로 뒷받침되지 않는 내용이 있으면 fail. 없으면 pass.
- 상품일치: 질문이 가리키는 상품과 근거의 상품이 다른데 답변이 그 사실을
  밝히지 않으면 fail. 같거나 상품 무관한 일반 규정이면 pass. 판단 불가면 na.
- 수치정확: 답변의 금리·기간·금액·비율이 근거와 다르면 fail. 일치하면 pass.
  수치가 없으면 na.
- 미확인신고: 질문이 물었으나 근거에 없는 항목을 답변이 언급하지 않고
  넘어갔으면 fail. 제대로 밝혔거나 해당 없으면 pass 또는 na.
- 실무활용: 은행 직원이 이 답변으로 고객에게 안내할 수 있으면 pass.

판정은 pass, fail, na 중 하나입니다.
근거에 없는 내용을 답변이 단정적으로 말하면 반드시 fail 입니다.

JSON으로만 답하십시오:
{"환각": "pass", "상품일치": "pass", "수치정확": "na", "미확인신고": "pass", "실무활용": "pass", "근거": "판정 이유 40자 이내"}"""

DRAFT_TEMPLATE = """[질문]
{question}

[시스템 답변]
{answer}

[시스템이 신고한 미확인 항목]
{not_found}

[근거 원문]
{evidence}"""


def draft_judge(client, model: str, item: dict) -> dict:
    ev = "\n\n".join(
        f"[{e['rank']}] {e['citation']}\n{e['text'][:700]}"
        for e in item["evidences"]
    ) or "(근거 없음)"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DRAFT_SYSTEM},
                {"role": "user", "content": DRAFT_TEMPLATE.format(
                    question=item["question"],
                    answer=item["answer"],
                    not_found=", ".join(item["not_found"]) or "(없음)",
                    evidence=ev,
                )},
            ],
            temperature=0,
            max_tokens=250,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        data["_tokens"] = resp.usage.prompt_tokens + resp.usage.completion_tokens
        return data
    except Exception as exc:
        return {"_error": str(exc)[:80]}


# ==================================================================
# 마크다운
# ==================================================================


def render_md(items: list[dict]) -> str:
    L: list[str] = []
    L.append("# G 평가셋 검수 — 답변 생성 품질")
    L.append("")
    L.append(f"생성일: {date.today().isoformat()} / 총 {len(items)}건")
    L.append("")
    L.append("## 검수 방법")
    L.append("")
    L.append("각 항목의 `판정:` 뒤 값을 고쳐 주세요. 값은 `pass`, `fail`, `na` 중 하나입니다.")
    L.append("초안이 있는 건은 동의하면 그대로 두시고, 다르면 고치시면 됩니다.")
    L.append("`(초안없음)` 표시가 붙은 건은 직접 판정해 주세요. 초안 편향을 재기 위한 대조군입니다.")
    L.append("")
    L.append("메모는 자유롭게 적어 주세요. 특히 `fail` 판정에는 이유를 한 줄 남겨 주시면")
    L.append("자동 평가기를 고칠 때 근거가 됩니다.")
    L.append("")
    for name, desc, scale in CRITERIA:
        L.append(f"- **{name}** — {desc} ({scale})")
    L.append("")
    L.append("---")
    L.append("")

    for it in items:
        L.append(f"## [{it['id']}] {it['group']}")
        L.append("")
        meta = []
        if it.get("difficulty"):
            meta.append(f"난이도 {it['difficulty']}")
        if it.get("doc_type"):
            meta.append(it["doc_type"])
        if it.get("note"):
            meta.append(it["note"])
        if meta:
            L.append(f"`{' / '.join(meta)}`")
            L.append("")

        L.append(f"**질문**  {it['question']}")
        L.append("")
        L.append(f"**시스템 판정**  `{it['decision']}` (stage={it['stage']}"
                 + (f", 사유={it['refusal_reason']}" if it.get("refusal_reason") else "")
                 + f", {it['latency_ms']}ms)")
        L.append("")
        L.append("**답변**")
        L.append("")
        for line in (it["answer"] or "").split("\n"):
            L.append(f"> {line}")
        L.append("")

        if it["not_found"]:
            L.append("**시스템이 신고한 미확인 항목**")
            L.append("")
            for x in it["not_found"]:
                L.append(f"- {x}")
            L.append("")

        if it["evidences"]:
            L.append("<details><summary><b>근거 원문 "
                     f"{len(it['evidences'])}건 (펼치기)</b></summary>")
            L.append("")
            for e in it["evidences"]:
                L.append(f"**[{e['rank']}] {e['citation']}**  `score={e['score']}`")
                L.append("")
                L.append("```")
                L.append(e["text"][:1200])
                L.append("```")
                L.append("")
            L.append("</details>")
            L.append("")

        L.append(f"<!-- ITEM {it['id']} -->")
        blind = " (초안없음)" if it["blind"] else ""
        for name, _, scale in CRITERIA:
            val = "?" if it["blind"] else it["draft"].get(name, "?")
            L.append(f"- {name} 판정: {val}   <!-- {scale}{blind} -->")
        if not it["blind"] and it["draft"].get("근거"):
            L.append(f"- 초안근거: {it['draft']['근거']}")
        L.append("- 메모: ")
        L.append("")
        L.append("---")
        L.append("")

    return "\n".join(L)


# ==================================================================
# 메인
# ==================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-draft", action="store_true")
    ap.add_argument("--llm-model", default="gpt-4.1-mini")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("=" * 76)
    print("G 평가셋 구축")
    print("=" * 76)

    retrieval = load_csv(EVAL_DIR / "retrieval_eval.csv")
    refusal = load_csv(EVAL_DIR / "refusal_eval.csv")
    coverage = load_coverage()
    print(f"  retrieval_eval {len(retrieval)}건 / refusal_eval {len(refusal)}건 "
          f"/ coverage {len(coverage)}건")

    if not retrieval:
        sys.exit("retrieval_eval.csv 를 찾지 못했습니다.")

    # --- 표본 ---
    picked: list[dict] = []
    for r in stratified(retrieval, N_ANSWERABLE, coverage, rng):
        picked.append({"question": pick(r, COL_QUESTION), "group": "정상 답변",
                       "difficulty": pick(r, COL_DIFFICULTY),
                       "doc_type": pick(r, COL_DOCTYPE), "note": ""})
    for r in sample_refusal(refusal, N_REFUSAL, rng):
        picked.append({"question": pick(r, COL_QUESTION), "group": "거절 대상",
                       "difficulty": "", "doc_type": "",
                       "note": f"기대 거절유형 {r.get('refusal_type', '?')}"})
    for r in sample_boundary(coverage, N_BOUNDARY, rng):
        picked.append({"question": r["question"], "group": "커버리지 경계",
                       "difficulty": "", "doc_type": "", "note": r.get("_note", "")})

    print(f"\n  표본 {len(picked)}건")
    for g, c in Counter(p["group"] for p in picked).most_common():
        print(f"    {g:<14} {c:>3}건")
    strata = Counter((p["difficulty"] or "-", p["doc_type"] or "-")
                     for p in picked if p["group"] == "정상 답변")
    if len(strata) > 1:
        print("\n  [정상 답변 층 분포]")
        for (d, t), c in sorted(strata.items()):
            print(f"    난이도 {d:<8} {t:<10} {c:>2}건")

    # --- 파이프라인 실행 ---
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    client = OpenAI()

    print("\n  인덱스 로딩 중...")
    pipeline = Pipeline.build(PROJECT_ROOT, client=client, llm_model=args.llm_model)

    print(f"  답변 생성 {len(picked)}건...")
    items = []
    for i, p in enumerate(picked, 1):
        ans = pipeline.answer(p["question"])
        d = ans.to_dict()
        items.append({
            "id": f"{i:02d}", "group": p["group"],
            "difficulty": p["difficulty"], "doc_type": p["doc_type"],
            "note": p["note"], **d,
        })
        if i % 5 == 0:
            print(f"    {i}/{len(picked)}")

    # --- 초안 판정 ---
    blind_ids = set(rng.sample([it["id"] for it in items], min(N_BLIND, len(items))))
    tokens = 0
    if not args.no_draft:
        print(f"\n  초안 판정 {len(items) - len(blind_ids)}건 "
              f"(백지 {len(blind_ids)}건)...")
        for it in items:
            if it["id"] in blind_ids:
                it["blind"], it["draft"] = True, {}
                continue
            it["blind"] = False
            it["draft"] = draft_judge(client, args.llm_model, it)
            tokens += int(it["draft"].get("_tokens", 0))
    else:
        for it in items:
            it["blind"], it["draft"] = True, {}

    errs = [it for it in items if it["draft"].get("_error")]
    if errs:
        print(f"  초안 오류 {len(errs)}건")

    # --- 저장 ---
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    REVIEW_MD.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_MD.write_text(render_md(items), encoding="utf-8")

    print("\n" + "=" * 76)
    print("완료")
    print("=" * 76)
    decisions = Counter(it["decision"] for it in items)
    print(f"  시스템 판정: " + " / ".join(f"{k} {v}" for k, v in decisions.items()))
    if not args.no_draft:
        drafted = [it for it in items if not it["blind"] and not it["draft"].get("_error")]
        for name, _, _ in CRITERIA:
            c = Counter(it["draft"].get(name, "?") for it in drafted)
            print(f"  초안 {name:<10} " + " / ".join(f"{k} {v}" for k, v in c.most_common()))
        print(f"  초안 토큰 {tokens:,}")

    print(f"\n저장 → {REVIEW_MD.relative_to(PROJECT_ROOT)}   <- 이 파일을 검수하세요")
    print(f"저장 → {CACHE.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
