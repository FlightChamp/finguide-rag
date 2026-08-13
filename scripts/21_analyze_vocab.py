"""
21_analyze_vocab.py

목적
----
BM25 색인 어휘의 문서빈도(DF)를 전수 조사해 불용어 후보를 찾는다.

왜 필요한가
---------
하이브리드 실험에서 BM25의 기여가 기대보다 작았다. 표본 400개 기준
어휘 통계를 보니 '상품', '설명서'가 100% 문서에 등장했다. 이는
indexable_text 에 문서명을 접두어로 붙였기 때문이다.

    "『주택청약예금』 요약 상품설명서\n합산) 보호됩니다..."
     -> 주택 · 청약 · 예금 · 요약 · 상품 · 설명서 · 합산 · 보호 ...

모든 청크에 '상품', '설명서'가 들어가므로 변별력이 0이다. BM25의 IDF가
가중치를 낮추긴 하지만, 질의에 "상품설명서"가 포함되면 노이즈로 작용한다.

판단 기준
--------
DF 비율이 높다고 무조건 빼면 안 된다. '대출'은 40% 문서에 나오지만
질의에서 중요한 변별 요소다. 따라서 두 가지를 함께 본다.

1. DF 비율 — 얼마나 흔한가
2. 문서 유형 간 분포 편차 — 특정 유형에만 몰려 있는가

'대출'은 약관·설명서에 몰려 있어 유형을 가르는 신호가 되지만,
'상품'은 모든 유형에 고르게 퍼져 있어 아무 정보도 주지 않는다.

사용법
-----
    python scripts/21_analyze_vocab.py
    python scripts/21_analyze_vocab.py --top 60
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.retrieval import KiwiTokenizer  # noqa: E402

CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"
EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
OUT_JSON = PROJECT_ROOT / "reports" / "vocab_analysis.json"

# 이 비율 이상 문서에 등장하면 불용어 후보로 본다.
DF_THRESHOLD = 0.35


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} 없음.")
    return [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]


def build_indexable_text(row: dict) -> str:
    parts = [row.get("doc_display_name", ""), row.get("section", "")]
    prefix = " ".join(p for p in parts if p and p != "FAQ")
    text = row.get("text", "")
    return f"{prefix}\n{text}" if prefix else text


def load_query_terms(tok: KiwiTokenizer) -> Counter:
    """평가셋 질의에 등장하는 어휘를 센다.

    질의에 자주 나오는 어휘를 불용어로 빼면 검색이 망가진다.
    DF가 높아도 질의에서 쓰이면 신중히 판단해야 한다.
    """
    if not EVAL_CSV.exists():
        return Counter()

    import csv
    with EVAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    counter: Counter = Counter()
    for r in rows:
        counter.update(set(tok.tokenize(r["question"])))
    return counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40, help="출력할 어휘 수")
    ap.add_argument("--threshold", type=float, default=DF_THRESHOLD)
    args = ap.parse_args()

    rows = load_chunks()
    tok = KiwiTokenizer()

    print("=" * 76)
    print("BM25 어휘 분포 분석")
    print("=" * 76)
    print(f"  청크 {len(rows):,}개 형태소 분석 중...")

    # 문서빈도(DF)와 문서유형별 분포를 함께 센다
    df: Counter = Counter()
    df_by_type: dict[str, Counter] = defaultdict(Counter)
    type_totals: Counter = Counter()

    for i, row in enumerate(rows, 1):
        tokens = set(tok.tokenize(build_indexable_text(row)))
        df.update(tokens)
        dt = row.get("doc_type", "?")
        df_by_type[dt].update(tokens)
        type_totals[dt] += 1
        if i % 500 == 0:
            print(f"    {i:,}/{len(rows):,}")

    n = len(rows)
    query_terms = load_query_terms(tok)

    print(f"\n  고유 어휘 {len(df):,}개")
    print(f"  질의 어휘 {len(query_terms):,}개 (평가셋 96건 기준)")

    # --- 고빈도 어휘 ---
    print("\n" + "=" * 76)
    print(f"고빈도 어휘 상위 {args.top}개")
    print("=" * 76)
    print("  DF     : 전체 문서 중 등장 비율")
    print("  유형편차: 문서 유형별 등장 비율의 표준편차. 낮을수록 어디에나 있다는 뜻")
    print("  질의   : 평가셋 질의에 등장한 횟수")
    print()
    print(f"  {'어휘':<14}{'DF':>8}  {'유형편차':>8}  {'질의':>5}   판정")
    print("  " + "-" * 62)

    candidates: list[str] = []
    keep: list[str] = []

    for word, count in df.most_common(args.top):
        ratio = count / n

        # 문서 유형별 등장 비율의 편차
        type_ratios = [
            df_by_type[dt][word] / type_totals[dt]
            for dt in type_totals if type_totals[dt] > 0
        ]
        mean = sum(type_ratios) / len(type_ratios)
        std = math.sqrt(sum((r - mean) ** 2 for r in type_ratios) / len(type_ratios))

        q_count = query_terms.get(word, 0)

        # 판정
        # 흔하면서 유형 간 편차도 작으면 아무 정보를 주지 않는다.
        # 다만 질의에 자주 쓰이는 어휘는 제외하지 않는다.
        if ratio >= args.threshold and std < 0.15 and q_count <= 2:
            verdict = "제외 후보"
            candidates.append(word)
        elif ratio >= args.threshold and q_count > 2:
            verdict = "유지 (질의 빈출)"
            keep.append(word)
        elif ratio >= args.threshold:
            verdict = "유지 (유형 편중)"
            keep.append(word)
        else:
            verdict = "유지"
            keep.append(word)

        print(f"  {word:<14}{ratio:>7.1%}  {std:>8.3f}  {q_count:>5}   {verdict}")

    # --- 유형별 특징 어휘 ---
    print("\n" + "=" * 76)
    print("문서 유형별 특징 어휘 (해당 유형에만 몰려 있는 어휘)")
    print("=" * 76)
    print("  이런 어휘는 유형을 구별하는 신호이므로 절대 제외하면 안 된다.")

    for dt in sorted(type_totals):
        scores: list[tuple[float, str, float]] = []
        for word, cnt in df_by_type[dt].items():
            if cnt < 10:
                continue
            in_ratio = cnt / type_totals[dt]
            out_cnt = df[word] - cnt
            out_total = n - type_totals[dt]
            out_ratio = out_cnt / out_total if out_total else 0
            if in_ratio > 0.15 and in_ratio > out_ratio * 3:
                scores.append((in_ratio - out_ratio, word, in_ratio))
        scores.sort(reverse=True)
        top = ", ".join(f"{w}({r:.0%})" for _, w, r in scores[:8])
        print(f"\n  [{dt}] {top}")

    # --- 결과 ---
    print("\n" + "=" * 76)
    print("불용어 추가 권고")
    print("=" * 76)
    if candidates:
        print(f"  {len(candidates)}개: {', '.join(candidates)}")
        print()
        print("  이 어휘들은 전체 문서의 {:.0%} 이상에 등장하면서".format(args.threshold))
        print("  문서 유형 간 분포 차이도 작아 검색에 기여하지 못한다.")
        print("  대부분 indexable_text 의 문서명 접두어에서 유입된 것이다.")
    else:
        print("  없음. 현재 불용어 목록으로 충분하다.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "n_chunks": n,
        "vocab_size": len(df),
        "df_threshold": args.threshold,
        "stopword_candidates": candidates,
        "top_terms": [
            {"word": w, "df": c, "ratio": round(c / n, 4),
             "in_queries": query_terms.get(w, 0)}
            for w, c in df.most_common(args.top)
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n저장 → {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
