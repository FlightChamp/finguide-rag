"""53 — 규칙층 단독 baseline: 오류 주입 평가셋에서 규칙만으로 얼마나 잡는가.

왜 LLM 없이 먼저 재는가
--------------------
v2 검증기는 'LLM(추출·정렬·함의) + 규칙(수치·상품·기관·단정)' 구조다.
LLM 을 붙이기 전에 규칙층만의 성능을 재 두면, 나중에 LLM 을 붙였을 때
'무엇이 LLM 덕분에 좋아졌는가'를 분리해 말할 수 있다 (ablation 의 출발점).
v1 거절 로직에서 0단계 → 0+1단계 → 0+1+2단계 기여도를 분해했던 것과 같은 구조다.

이 측정의 한계 (리포트에도 그대로 적는다)
------------------------------------
1. LLM 정렬이 없어서 근거 span 은 '청크 문장 중 글자 2-gram 이 가장 많이 겹치는 2문장'
   (51번과 같은 proxy)으로 대신한다.
2. claim 분해 대신 문장 단위로 본다. 주입 위치(span)를 포함한 문장이 '그 claim'이다.
3. claim 의 상품명 슬롯(LLM 추출 예정)은 문장에 등장한 카탈로그 개별 상품명으로 대신한다.
4. 기반 초안 30건이 아직 사람 검수 전(provisional)이면 결과도 잠정치다.
5. 규칙 v0.3 은 개발 중 test 30건을 보고 두 번 조정됐다 → Clean False Alarm 은 낙관 편향.

지표
----
    탐지(Detected)        주입 문장의 규칙 판정이 OK 가 아님 (사유 코드가 하나라도 있음)
    코드 정확(Code-correct) 기대 사유 코드가 실제로 발급됨
    Clean False Alarm     대조군(NONE) 문장 중 경고가 난 비율 / 초안 단위 비율
모든 비율에 n 과 95% Wilson 신뢰구간을 붙인다.

실행
----
    python scripts/53_rule_baseline.py
    python scripts/53_rule_baseline.py --misses      # 놓친 주입 전부 출력

산출물
-----
    reports/v2_rule_baseline_<날짜>.md     요약표
    reports/v2_rule_baseline_<날짜>.csv    샘플별 결과
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.product_catalog import ProductCatalog  # noqa: E402
from finguide_rag.generation.product_match import ProductMatcher  # noqa: E402
from finguide_rag.verification.rules import run_claim_rules  # noqa: E402
from finguide_rag.verification.schema import EvidenceRef  # noqa: E402

INJECTED = PROJECT_ROOT / "data" / "eval" / "v2" / "injected_v0.csv"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
CATALOG = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"
REPORT_DIR = PROJECT_ROOT / "reports"

_SENT = re.compile(r"(?:[^.!?]|\.(?=\d))+[.!?]?")   # 1.5% 의 소수점에서 자르지 않는다
_CHUNK_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|\s(?=\d+\.\s)|\s(?=[·•\-]\s)")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "   -  (n=0)"
    lo, hi = wilson(k, n)
    return f"{k / n:5.2f} [{lo:.2f}–{hi:.2f}] (n={n})"


def _bigrams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def proxy_span(claim: str, chunks: list[str], k: int = 2) -> str:
    cb = _bigrams(claim)
    sents = [s for c in chunks for s in _CHUNK_SPLIT.split(c) if len(s.strip()) > 4]
    sents.sort(key=lambda s: len(cb & _bigrams(s)), reverse=True)
    return "\n".join(sents[:k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()
    if not INJECTED.exists():
        sys.exit("injected_v0.csv 가 없습니다. 52번을 먼저 실행하세요.")

    with INJECTED.open(encoding="utf-8-sig", newline="") as f:
        samples = list(csv.DictReader(f))
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    matcher = ProductMatcher.from_catalog(CATALOG)
    products = sorted((e.canonical_name for e in ProductCatalog.load(CATALOG).covered_products
                       if e.canonical_name), key=len, reverse=True)

    def claim_product(sentence: str) -> str | None:
        return next((p for p in products if p in sentence), None)

    def evaluate(sentence: str, evid: list[dict]) -> list[str]:
        chunks = [e.get("text", "") for e in evid]
        quote = proxy_span(sentence, chunks)
        refs = [EvidenceRef(doc_id=e["chunk_id"].rsplit("_c", 1)[0], chunk_id=e["chunk_id"],
                            quote=quote if i == 0 else "", doc_name=e.get("citation", ""),
                            doc_type=e.get("doc_type", ""))
                for i, e in enumerate(evid)] or [EvidenceRef("", "", quote)]
        r = run_claim_rules(sentence, refs, claim_product=claim_product(sentence), matcher=matcher)
        return [h.code.value for h in r.hits]

    rows = []
    det = defaultdict(lambda: [0, 0, 0])       # subtype → [n, detected, code_correct]
    clean_sent = [0, 0]                        # [n, flagged]
    clean_draft = [0, 0]
    provisional = sum(1 for s in samples if s["label_status"] != "confirmed")

    for s in samples:
        evid = (answers.get(s["question"]) or {}).get("evidences", [])
        text = s["text"]
        if s["error_type"] == "NONE":
            any_flag = False
            for m in _SENT.finditer(text):
                sent = m.group(0).strip()
                if len(sent) < 5:
                    continue
                codes = evaluate(sent, evid)
                clean_sent[0] += 1
                clean_sent[1] += bool(codes)
                any_flag |= bool(codes)
                rows.append({**s, "eval_sentence": sent, "fired": "|".join(codes),
                             "detected": "", "code_correct": "", "false_alarm": int(bool(codes))})
            clean_draft[0] += 1
            clean_draft[1] += any_flag
            continue

        a = int(s["span_start"])
        m = next((m for m in _SENT.finditer(text) if m.start() <= a < m.end()), None)
        sent = (m.group(0) if m else text).strip()
        codes = evaluate(sent, evid)
        expected = [c for c in s["expected_codes"].split("|") if c]
        detected = bool(codes)
        correct = any(c in codes for c in expected)
        key = f"{s['error_type']}/{s['subtype']}"
        if s["subtype"] == "intensifier":
            key += f" (탐지 목록 {'안' if s['in_detector_vocab'] == 'Y' else '밖'})"
        det[key][0] += 1
        det[key][1] += detected
        det[key][2] += correct
        rows.append({**s, "eval_sentence": sent, "fired": "|".join(codes),
                     "detected": int(detected), "code_correct": int(correct), "false_alarm": ""})

    # ---------- 출력 ----------
    lines = []
    def out(x=""):
        print(x)
        lines.append(x)

    out("=" * 72)
    out("규칙층 단독 baseline (LLM 없음, 근거=유사문장 proxy)")
    out("=" * 72)
    if provisional:
        out(f"※ 사람 검수 전 샘플 {provisional}/{len(samples)}건 — 잠정치")
    out(f"\n{'유형/하위유형':<38}{'탐지':<26}{'코드 정확'}")
    for k in sorted(det):
        n, d, c = det[k]
        out(f"  {k:<36}{fmt_rate(d, n):<26}{fmt_rate(c, n)}")
    inj_n = sum(v[0] for v in det.values())
    inj_d = sum(v[1] for v in det.values())
    out(f"  {'(전체 주입)':<36}{fmt_rate(inj_d, inj_n)}")
    out(f"\nClean False Alarm  문장 {fmt_rate(clean_sent[1], clean_sent[0])}"
        f"   초안 {fmt_rate(clean_draft[1], clean_draft[0])}")

    misses = [r for r in rows if r["detected"] == 0]
    if args.misses and misses:
        out("\n[놓친 주입]")
        for r in misses:
            out(f"  - {r['error_type']}/{r['subtype']} [{r['base_id']}] "
                f"'{r['original_fragment']}' → '{r['injected_fragment']}' | {r['eval_sentence'][:70]}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"v2_rule_baseline_{date.today().isoformat()}"
    with (REPORT_DIR / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        cols = ["sample_id", "base_id", "error_type", "subtype", "in_detector_vocab",
                "original_fragment", "injected_fragment", "expected_codes", "eval_sentence",
                "fired", "detected", "code_correct", "false_alarm", "label_status"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (REPORT_DIR / f"{stem}.md").write_text(
        "# v2 규칙층 단독 baseline\n\n```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
    print(f"\n→ reports/{stem}.md / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
