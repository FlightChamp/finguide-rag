"""
31_measure_coverage.py

목적
----
기존 평가셋에 product_mismatch 유형이 몇 건이나 숨어 있는지 측정한다.
아무것도 수정하지 않는다. refusal.py 는 건드리지 않는다.

왜 게이트 구현보다 측정이 먼저인가
------------------------------
게이트를 먼저 붙이면 FAR 0.030 과 과잉거절 0.100 이 왜 변했는지
설명할 수 없다. 기존 53건에 이 유형이 0건이면 게이트는 기존 수치를
전혀 바꾸지 않아야 하고, 그것이 "새 유형이라 기존 평가셋이 못 잡았다"의
정량적 근거가 된다. 반대로 몇 건 숨어 있으면 기존 수치의 해석 자체가
달라진다.

확정된 로직을 건드리기 전에 영향 범위를 아는 것이 먼저다.

비용
----
질문당 LLM 1회. 결과는 디스크에 캐시되므로 규칙을 바꿔 재측정해도
호출은 늘지 않는다.

사용법
-----
    python scripts/31_measure_coverage.py
        카탈로그 생성 + 평가셋 전수 분석

    python scripts/31_measure_coverage.py --catalog-only
        카탈로그만 만들고 출력한다 (LLM 미사용, 무료).
        상품명 추출이 제대로 됐는지 눈으로 확인할 때 쓴다.

    python scripts/31_measure_coverage.py -q "정기예금 중도해지하면 이자는?"
        질문 하나만 분석한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.catalog_matcher import (  # noqa: E402
    match_product,
    should_refuse_product_mismatch,
)
from finguide_rag.generation.product_catalog import ProductCatalog  # noqa: E402
from finguide_rag.generation.query_analyzer import QueryAnalyzer  # noqa: E402

REGISTRY = PROJECT_ROOT / "data" / "registry" / "document_registry.csv"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
CATALOG_OUT = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"
CACHE_OUT = PROJECT_ROOT / "data" / "interim" / "query_analysis_cache.json"
REPORT_DIR = PROJECT_ROOT / "reports"

QUESTION_KEYS = ("question", "query", "질문")


def find_eval_files() -> list[Path]:
    """data/eval 아래에서 질문 컬럼을 가진 CSV 를 찾는다."""
    out = []
    for p in sorted(EVAL_DIR.glob("*.csv")):
        try:
            with p.open(encoding="utf-8-sig", newline="") as f:
                header = next(csv.reader(f), [])
        except Exception:
            continue
        if any(k in header for k in QUESTION_KEYS):
            out.append(p)
    return out


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    key = next((k for k in QUESTION_KEYS if rows and k in rows[0]), None)
    if key is None:
        return []
    out = []
    for r in rows:
        q = (r.get(key) or "").strip()
        if not q:
            continue
        out.append({
            "question": q,
            "expected": (r.get("expected") or "").strip(),
            "refusal_type": (r.get("refusal_type") or "").strip(),
        })
    return out


def print_catalog(cat: ProductCatalog) -> None:
    print("\n" + "=" * 76)
    print("상품 카탈로그")
    print("=" * 76)
    print(f"  특정 상품   {len(cat.covered_products):>4}건")
    print(f"  상품군      {len(cat.covered_families):>4}건")
    print(f"  일반 약관   {len(cat.general_policy_docs):>4}건")
    print(f"  제외        {len(cat.skipped):>4}건")

    print("\n  [특정 상품]")
    for e in cat.covered_products:
        print(f"    {e.category:<18} {e.canonical_name}")

    print("\n  [상품군]")
    for e in cat.covered_families:
        print(f"    {e.category:<18} {e.canonical_name}")

    print("\n  [일반 약관 — 상품설명서 없이도 답변 가능한 근거]")
    seen = set()
    for e in cat.general_policy_docs:
        tag = (e.canonical_name, e.doc_title)
        if tag in seen:
            continue
        seen.add(tag)
        print(f"    {e.canonical_name:<10} {e.doc_title[:52]}")

    if cat.skipped:
        print(f"\n  [카탈로그 제외 {len(cat.skipped)}건]")
        why = Counter(s["why"] for s in cat.skipped)
        for k, v in why.most_common():
            print(f"    {k:<24} {v:>3}건")


def analyze_one(analyzer: QueryAnalyzer, catalog: ProductCatalog,
                question: str) -> dict:
    a = analyzer.analyze(question)
    m = match_product(a.extracted_product, catalog)
    refuse, reason = should_refuse_product_mismatch(a, m)
    return {
        "question": question,
        "extracted_product": a.extracted_product,
        "granularity": a.product_granularity,
        "intent": a.intent,
        "needs_specific": a.requires_product_specific_doc,
        "general_ok": a.can_answer_with_general_terms,
        "confidence": round(a.confidence, 3),
        "match_status": m.status,
        "matched_product": m.matched_product,
        "candidates": "|".join(m.candidates or []),
        "gate_refuse": refuse,
        "gate_reason": reason or "",
        "error": a.error,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-only", action="store_true")
    ap.add_argument("-q", "--question", default=None)
    ap.add_argument("--llm-model", default="gpt-4.1-mini")
    args = ap.parse_args()

    print("=" * 76)
    print("코퍼스 커버리지 측정 — product_mismatch 후보 탐색")
    print("=" * 76)

    catalog = ProductCatalog.from_registry(REGISTRY)
    catalog.save(CATALOG_OUT)
    print_catalog(catalog)
    print(f"\n저장 → {CATALOG_OUT.relative_to(PROJECT_ROOT)}")

    if args.catalog_only:
        print("\n  카탈로그만 생성했습니다. 상품명 추출이 올바른지 위 목록을")
        print("  확인한 뒤 LLM 분석을 실행하십시오.")
        return

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    analyzer = QueryAnalyzer(OpenAI(), args.llm_model, CACHE_OUT)

    # --- 단건 ---
    if args.question:
        r = analyze_one(analyzer, catalog, args.question)
        analyzer.flush()
        print("\n" + "=" * 76)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    # --- 평가셋 전수 ---
    files = find_eval_files()
    if not files:
        sys.exit(f"{EVAL_DIR} 에서 질문 컬럼을 가진 CSV 를 찾지 못했습니다.")

    print("\n" + "=" * 76)
    print("평가셋 분석")
    print("=" * 76)
    for p in files:
        print(f"  발견: {p.name}")

    all_rows: list[dict] = []
    for path in files:
        rows = load_questions(path)
        if not rows:
            continue
        print(f"\n  [{path.name}] {len(rows)}건 분석 중...")

        results = []
        for i, r in enumerate(rows, 1):
            out = analyze_one(analyzer, catalog, r["question"])
            out["source"] = path.name
            out["expected"] = r["expected"]
            out["refusal_type"] = r["refusal_type"]
            results.append(out)
            all_rows.append(out)
            if i % 20 == 0:
                print(f"    {i}/{len(rows)}")
                analyzer.flush()
        analyzer.flush()

        # 요약
        status = Counter(r["match_status"] for r in results)
        gate = [r for r in results if r["gate_refuse"]]
        print(f"\n    카탈로그 대조: " + " / ".join(
            f"{k} {v}" for k, v in status.most_common()))
        print(f"    게이트 거절 후보: {len(gate)}건 "
              f"({len(gate) / len(results):.1%})")

        if gate:
            by_reason = Counter(r["gate_reason"] for r in gate)
            for k, v in by_reason.most_common():
                print(f"      {k:<20} {v:>3}건")

            # 기존 정답 라벨과 대조한다.
            # expected=answer 인데 게이트가 거절하면 새로운 과잉 거절이다.
            conflict = [r for r in gate if r["expected"] == "answer"]
            agree = [r for r in gate if r["expected"] == "refuse"]
            if r"expected" in results[0] and (conflict or agree):
                print(f"\n      기존 라벨 대조")
                print(f"        expected=refuse 와 일치   {len(agree):>3}건")
                print(f"        expected=answer 와 충돌   {len(conflict):>3}건 "
                      f"<- 새로운 과잉 거절 위험")

            print(f"\n    [게이트 거절 후보 목록]")
            for r in gate[:20]:
                lab = f" (라벨:{r['expected']}/{r['refusal_type']})" if r["expected"] else ""
                print(f"      [{r['gate_reason']:<18}] {r['question'][:44]}{lab}")
                print(f"         추출={r['extracted_product']} "
                      f"매칭={r['match_status']} conf={r['confidence']}")

    # --- 전체 요약 ---
    print("\n" + "=" * 76)
    print("전체 요약")
    print("=" * 76)
    print(f"  분석 질문        {len(all_rows)}건")
    print(f"  LLM 호출(캐시)   {analyzer.cached_count}건 / "
          f"{analyzer.total_tokens:,} 토큰")

    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_src[r["source"]].append(r)

    print(f"\n  {'평가셋':<28}{'질문':>6}{'게이트 거절':>12}{'비율':>8}")
    print("  " + "-" * 54)
    for src, rows in by_src.items():
        g = sum(1 for r in rows if r["gate_refuse"])
        print(f"  {src:<28}{len(rows):>6}{g:>12}{g / len(rows):>8.1%}")

    errs = [r for r in all_rows if r["error"]]
    if errs:
        print(f"\n  분석 오류 {len(errs)}건")
        for r in errs[:5]:
            print(f"    {r['question'][:40]} -> {r['error']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"coverage_measure_{date.today().isoformat()}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n저장 → {out.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
