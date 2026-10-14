"""59 — 초안 수준 범위 검사(scope.py) 개발 평가. 개발 셋만 사용한다.

사용 데이터 (모두 이미 사용한 개발 셋 — 새 held-out 은 읽지 않는다)
-------------------------------------------------------------
    정상 초안 25건 (clean_drafts_v1, Y)              → 경고가 나면 오탐
    개발 43건 (heldout_review_v0 → 이제 개발 셋)       → Y 는 오탐 후보, N 은 탐지 대상
    자연 오류 8건 (natural_errors_v0)
    개발 표적: B_01, B_03, q043, q055 (범위 혼합·축소 사례)

정렬 모드
--------
    --align llm    (기본) ClaimVerifier 실제 경로. LLM 캐시만 사용하고 API 는 호출하지 않는다.
                   54·58번을 cv-v1 으로 돌린 적이 있으면 개발 셋 전부가 캐시에 있다 → 0콜.
    --align proxy  LLM 없이 문장 분해 + 글자 2-gram 최다 겹침 청크를 근거로 쓴다 (설계 초기 확인용)

실행
----
    python scripts/59_dev_scope.py
    python scripts/59_dev_scope.py --align proxy

산출물
-----
    reports/v2_dev_scope_<날짜>.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.product_catalog import ProductCatalog  # noqa: E402
from finguide_rag.generation.product_match import ProductMatcher  # noqa: E402
from finguide_rag.verification.claim_verifier import CachedLLM, ClaimVerifier, EvidenceChunk  # noqa: E402
from finguide_rag.verification.schema import Claim, EvidenceRef, Verification, Verdict  # noqa: E402
from finguide_rag.verification.scope import DocIndex, check_draft_scope  # noqa: E402

V2 = PROJECT_ROOT / "data" / "eval" / "v2"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
CATALOG = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"
DOCS = PROJECT_ROOT / "data" / "interim" / "parsed" / "documents.jsonl"
CACHE = PROJECT_ROOT / "data" / "interim" / "v2_verifier_cache.json"
REPORT_DIR = PROJECT_ROOT / "reports"
TARGETS = {"B_01", "B_03", "q043", "q055"}

_SENT = re.compile(r"(?:[^.!?]|\.(?=\d))+[.!?]?")
_CH = re.compile(r"(?<=[.!?])\s+|\n+")


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def items() -> list[dict]:
    out = []
    for r in read_csv(V2 / "clean_drafts_v1.csv"):
        if r["human_is_correct"] == "Y":
            out.append({"id": r["draft_id"], "group": "정상25", "label": "Y",
                        "question": r["question"], "text": r["draft_text"]})
    for r in read_csv(V2 / "heldout_review_v0.csv"):
        if r["human_is_correct"] in ("Y", "N"):
            out.append({"id": r["draft_id"], "group": "개발43", "label": r["human_is_correct"],
                        "question": r["question"], "text": r["draft_text"]})
    for r in read_csv(V2 / "natural_errors_v0.csv"):
        out.append({"id": r["nat_id"], "group": "자연8", "label": "N",
                    "question": r["question"], "text": r["text"]})
    return out


def bigrams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def proxy_verify(question, text, chunks):
    """LLM 없이: 문장 = claim, 근거 = 2-gram 겹침 최대 청크 1개 (겹침 0.3 이상)."""
    claims, vers = [], []
    for i, m in enumerate(_SENT.finditer(text), 1):
        s = m.group(0).strip()
        if len(s) < 5:
            continue
        cb = bigrams(s)
        best, score = None, 0.0
        for ch in chunks:
            sc = max((len(cb & bigrams(x)) / (len(cb) or 1) for x in _CH.split(ch.text) if x.strip()), default=0)
            if sc > score:
                best, score = ch, sc
        ev = [EvidenceRef(best.doc_id, best.chunk_id, best.text, doc_name=best.doc_name,
                          doc_type=best.doc_type)] if best is not None and score >= 0.3 else []
        cid = f"c{i}"
        claims.append(Claim(cid, s, (m.start(), m.end())))
        vers.append(Verification(cid, Verdict.OK, [], evidence=ev))
    return claims, vers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", choices=["llm", "proxy"], default="llm")
    args = ap.parse_args()

    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    matcher = ProductMatcher.from_catalog(CATALOG)
    known = [e.canonical_name for e in ProductCatalog.load(CATALOG).all_named if e.canonical_name]
    index = DocIndex.build(DOCS, CATALOG, matcher)
    llm = CachedLLM(None, "gpt-4.1-mini", CACHE)        # API 호출 없음
    cv = ClaimVerifier(llm, matcher, known_products=known, prompt_version="cv-v1")

    lines: list[str] = []

    def out(x=""):
        print(x)
        lines.append(x)

    rows, misses = [], 0
    for it in items():
        chunks = [EvidenceChunk(e["chunk_id"], e.get("text", ""), e.get("citation", ""), e.get("doc_type", ""))
                  for e in (answers.get(it["question"]) or {}).get("evidences", [])]
        if args.align == "llm":
            res = cv.verify(it["question"], it["text"], chunks)
            if res.errors:
                misses += 1
            claims, vers = res.claims, res.verifications
            base = res.final.final.value
        else:
            claims, vers = proxy_verify(it["question"], it["text"], chunks)
            base = "-"
        hits = check_draft_scope(it["question"], claims, vers, index, matcher)
        rows.append({**it, "base": base, "hits": hits})

    out("=" * 72)
    out(f"초안 수준 범위 검사 — 개발 셋 (정렬={args.align})")
    out("=" * 72)
    if args.align == "llm" and misses:
        out(f"※ LLM 캐시에 없는 항목 {misses}건 — 대체 claim 으로 처리됨 (54·58번 cv-v1 실행 후 다시 돌릴 것)")

    ys = [r for r in rows if r["label"] == "Y"]
    fp = [r for r in ys if r["hits"]]
    out(f"\n[오탐] 사람=Y {len(ys)}건 중 범위 경고 {len(fp)}건")
    for r in fp:
        out(f"  {r['group']:<6}{r['id']:<6} {' / '.join(h.code.value + ': ' + h.detail for h in r['hits'])[:150]}")

    out("\n[개발 표적 4건]")
    for r in rows:
        if r["id"] in TARGETS:
            got = " / ".join(h.code.value + ": " + h.detail for h in r["hits"]) or "(경고 없음)"
            out(f"  {'✔' if r['hits'] else '✘'} {r['id']:<6} {got[:150]}")

    ns = [r for r in rows if r["label"] == "N"]
    out(f"\n[사람=N {len(ns)}건 전체]  범위 경고 {sum(bool(r['hits']) for r in ns)}건")
    if args.align == "llm":
        before = sum(r["base"] != "OK" for r in ns)
        after = sum(r["base"] != "OK" or bool(r["hits"]) for r in ns)
        out(f"  최종 판정 OK 아님: 범위 검사 전 {before}/{len(ns)} → 후 {after}/{len(ns)}")
        ybefore = sum(r["base"] != "OK" for r in ys)
        yafter = sum(r["base"] != "OK" or bool(r["hits"]) for r in ys)
        out(f"  정상(Y) OK 아님:   범위 검사 전 {ybefore}/{len(ys)} → 후 {yafter}/{len(ys)}")
    for r in ns:
        if r["hits"]:
            out(f"  {r['group']:<6}{r['id']:<6} {' / '.join(h.code.value for h in r['hits'])}")
    reasons_path = V2 / "dev_reason_labels_v0.csv"
    if reasons_path.exists():
        rmap = {r["id"]: r["reasons"].split("|") for r in read_csv(reasons_path)}
        out("\n[사유별 범위 경고]  (61번 사유 분류 기준)")
        for cls in ("MIXED_SCOPE", "SCOPE_MISMATCH", "WRONG", "UNSUPPORTED", "MISSING_CONDITION",
                    "OVERSTATED", "OMITTED", "OUTDATED"):
            rs = [r for r in ns if cls in rmap.get(r["id"].removeprefix("A_"), [])]
            if rs:
                ids = ", ".join(r["id"] for r in rs if r["hits"])
                out(f"  {cls:<18} {sum(bool(r['hits']) for r in rs)}/{len(rs)}   {ids}")
    out("\n※ 개발 셋 수치다. 공식 효과는 새 held-out(60번)으로만 보고한다.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"v2_dev_scope_{date.today().isoformat()}_{args.align}.md"
    path.write_text("# 초안 수준 범위 검사 — 개발 셋\n\n```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
    print(f"→ {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
