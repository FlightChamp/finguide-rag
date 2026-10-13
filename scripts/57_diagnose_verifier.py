"""57 — 54번 결과 진단: 정상 초안에서 경고가 왜 났는가 (API 호출 없음).

54번에서 정상 초안 claim 의 57%(soft)가 OK 가 아니었고, 10%가 PROHIBITED 였다.
자연 오류 셋 '탐지'도 상당수가 기대 코드가 아닌 NEEDS_CUSTOMER_DATA 로 걸렸다.
이 스크립트는 그 경고들을 사유 코드 × 발급 주체 × 세부 사유로 쪼개 보여준다.

입력
----
    data/interim/v2_verifier_results.jsonl   (54번 산출, 주입 셋 샘플별 원자료)
    data/eval/v2/injected_v0.csv              (샘플 유형)
    data/eval/v2/natural_errors_v0.csv        (자연 오류 셋 — 캐시로 재실행, API 0콜)

실행
----
    python scripts/57_diagnose_verifier.py
    python scripts/57_diagnose_verifier.py --examples 8     # 코드별 예시 수

산출물
-----
    reports/v2_verifier_diagnosis_<날짜>.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RAW = PROJECT_ROOT / "data" / "interim" / "v2_verifier_results.jsonl"
INJECTED = PROJECT_ROOT / "data" / "eval" / "v2" / "injected_v0.csv"
NATURAL = PROJECT_ROOT / "data" / "eval" / "v2" / "natural_errors_v0.csv"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
CATALOG = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"
CACHE = PROJECT_ROOT / "data" / "interim" / "v2_verifier_cache.json"
REPORT_DIR = PROJECT_ROOT / "reports"


def detail_key(d: str) -> str:
    """세부 사유를 묶기 좋은 형태로 (수치·따옴표 안 내용 제거)."""
    d = re.sub(r"'[^']*'", "'…'", d)
    d = re.sub(r"\[[^\]]*\]", "[…]", d)
    return d[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()
    if not RAW.exists():
        sys.exit("v2_verifier_results.jsonl 이 없습니다. 54번을 먼저 실행하세요.")

    with INJECTED.open(encoding="utf-8-sig", newline="") as f:
        kind = {r["sample_id"]: r["error_type"] for r in csv.DictReader(f)}
    recs = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    clean = [r for r in recs if kind.get(r["sample_id"]) == "NONE"]

    lines: list[str] = []

    def out(x=""):
        print(x)
        lines.append(x)

    # ---------- 정상 초안 claim 의 경고 분해 ----------
    n_claims = sum(len(r["verifications"]) for r in clean)
    verdicts = Counter(v["verdict"] for r in clean for v in r["verifications"])
    out("=" * 78)
    out(f"[1] 정상 초안 {len(clean)}건 · claim {n_claims}개 판정 분포: {dict(verdicts)}")
    out("=" * 78)
    by_code = Counter()
    by_detail = defaultdict(Counter)
    examples = defaultdict(list)
    only_reason = Counter()          # 그 claim 의 유일한 비-OK 사유 (이것만 고치면 OK 가 되는 것)
    for r in clean:
        claims = {c["claim_id"]: c for c in r["claims"]}
        for v in r["verifications"]:
            if v["verdict"] == "OK":
                continue
            bad = [h for h in v["reasons"] if h["code"] != "SUPPORTED"]
            for h in bad:
                key = f"{h['code']} ({h['source']})"
                by_code[key] += 1
                by_detail[key][detail_key(h["detail"])] += 1
                if len(examples[key]) < args.examples:
                    examples[key].append((r["sample_id"], claims[v["claim_id"]]["text"], h["detail"],
                                          [e["quote"] for e in v["evidence"]], v["verdict"]))
            codes = {h["code"] for h in bad}
            if len(codes) == 1:
                only_reason[codes.pop()] += 1

    out(f"\n{'사유 코드 (발급 주체)':<40}{'건수':>6}{'단독 원인':>10}")
    for key, n in by_code.most_common():
        out(f"  {key:<38}{n:>6}{only_reason[key.split(' ')[0]]:>10}")
    out("  ※ '단독 원인' = 그 코드 하나 때문에 OK 가 아니게 된 claim 수. 이 코드를 고치면 바로 줄어드는 양")

    for key, _ in by_code.most_common():
        out(f"\n── {key}")
        for d, n in by_detail[key].most_common(4):
            out(f"   세부 {n:>3}  {d}")
        for sid, text, detail, quotes, verdict in examples[key]:
            out(f"   예) [{sid}] {verdict}  claim: {text[:70]}")
            out(f"       사유: {detail[:110]}")
            out(f"       근거: {(' / '.join(quotes))[:150] or '(없음)'}")

    # ---------- PROHIBITED 정상 claim 전수 ----------
    out("\n" + "=" * 78)
    out("[2] 정상 초안인데 PROHIBITED 가 난 claim 전수")
    out("=" * 78)
    for r in clean:
        claims = {c["claim_id"]: c for c in r["claims"]}
        for v in r["verifications"]:
            if v["verdict"] == "PROHIBITED":
                hs = [f"{h['code']}:{h['detail'][:70]}" for h in v["reasons"] if h["code"] != "SUPPORTED"]
                out(f"  [{r['sample_id']}] {claims[v['claim_id']]['text'][:80]}")
                for h in hs:
                    out(f"      - {h}")
                out(f"      근거: {(' / '.join(e['quote'] for e in v['evidence']))[:160] or '(없음)'}")

    # ---------- 자연 오류 셋: 캐시로 재실행해 claim 단위로 ----------
    if NATURAL.exists():
        from finguide_rag.generation.product_match import ProductMatcher
        from finguide_rag.verification.claim_verifier import CachedLLM, ClaimVerifier, EvidenceChunk

        answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
        llm = CachedLLM(None, "gpt-4.1-mini", CACHE)       # client=None → 캐시만 사용, API 0콜
        from finguide_rag.generation.product_catalog import ProductCatalog
        known = [e.canonical_name for e in ProductCatalog.load(CATALOG).all_named if e.canonical_name]
        cv = ClaimVerifier(llm, ProductMatcher.from_catalog(CATALOG), known_products=known)
        out("\n" + "=" * 78)
        out("[3] 자연 오류 셋 — claim 단위 (캐시 재실행, API 0콜)")
        out("=" * 78)
        with NATURAL.open(encoding="utf-8-sig", newline="") as f:
            for n in csv.DictReader(f):
                chunks = [EvidenceChunk(e["chunk_id"], e.get("text", ""), e.get("citation", ""),
                                        e.get("doc_type", ""))
                          for e in (answers.get(n["question"]) or {}).get("evidences", [])]
                res = cv.verify(n["question"], n["text"], chunks)
                expected = set(n["expected_codes"].split("|"))
                fired = {h.code.value for v in res.verifications for h in v.reasons} - {"SUPPORTED"}
                out(f"\n  {n['nat_id']}  gold={n['expected_codes']}  → 기대 코드 적중: "
                    f"{'예' if fired & expected else '아니오'}   ({n['gold_note'][:40]})")
                for c, v in zip(res.claims, res.verifications):
                    codes = [f"{h.code.value}({h.source.value})" for h in v.reasons if h.code.value != "SUPPORTED"]
                    out(f"    {v.verdict.value:<10} {c.text[:60]}  {', '.join(codes)}")
        if llm.calls == 0 and llm.hits == 0:
            out("  (캐시가 비어 있어 재실행하지 못함)")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"v2_verifier_diagnosis_{date.today().isoformat()}.md"
    path.write_text("# v2 검증기 진단\n\n```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
    print(f"\n→ {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
