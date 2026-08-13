"""
18_analyze_failures.py

목적
----
평가에서 실패한 질의를 분석해 개선 수단을 정한다.

무엇을 판단하는가
---------------
1. 문서 유형별 실패 분포
   약관이 유독 약한지, 그렇다면 어떤 성격의 질문에서 실패하는지 본다.

2. BM25로 잡을 수 있는가
   질의의 단어가 정답 청크에 실제로 등장하는지 확인한다.
   등장한다면 어휘 매칭(BM25)으로 잡을 수 있고, 등장하지 않는다면
   표현 자체가 달라 질의 확장이나 재순위화가 필요하다.

3. 검색된 문서가 왜 올라왔는가
   엉뚱한 문서가 상위에 오는 이유를 보면 개선 방향이 잡힌다.

사용법
-----
    python scripts/18_analyze_failures.py
    python scripts/18_analyze_failures.py --type 약관
    python scripts/18_analyze_failures.py --detail 5     # 상세 5건
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"

# 어휘 매칭 판정에서 제외할 일반어.
# 이런 단어는 어느 문서에나 있어 BM25 가능성 판단에 도움이 안 된다.
STOPWORDS = {
    "하나은행", "하나", "은행", "경우", "어떻게", "무엇", "어떤", "언제", "얼마",
    "있나요", "하나요", "되나요", "인가요", "합니까", "됩니까", "가능", "필요",
    "제가", "저는", "내가", "우리", "그거", "이거", "때문", "위해", "대해",
}


def find_latest(pattern: str) -> Path | None:
    files = sorted(REPORT_DIR.glob(pattern))
    return files[-1] if files else None


def load_chunks() -> dict[str, dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} 없음.")
    out = {}
    for line in CHUNKS_PATH.open(encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            out[c["chunk_id"]] = c
    return out


def extract_terms(text: str) -> list[str]:
    """질의에서 검색에 의미 있는 어절을 뽑는다.

    조사를 떼고 2자 이상 명사만 남긴다. 완벽한 형태소 분석은 아니지만
    어휘 매칭 가능성을 가늠하는 데는 충분하다.
    """
    terms = []
    for raw in re.findall(r"[가-힣A-Za-z]{2,}", text):
        token = re.sub(
            r"(으로|에서|에게|까지|부터|이나|라도|든지|에는|은|는|이|가|을|를|의|에|도|만|과|와|로)$",
            "", raw,
        )
        if len(token) >= 2 and token not in STOPWORDS:
            terms.append(token)
    return terms


def lexical_overlap(question: str, chunk_text: str) -> tuple[float, list[str]]:
    """질의 어휘가 정답 청크에 얼마나 등장하는지.

    비율이 높으면 BM25 같은 어휘 기반 검색으로 잡을 수 있다는 신호다.
    낮으면 질의와 문서가 서로 다른 말로 같은 개념을 표현하고 있다는 뜻이라,
    어휘 매칭으로는 해결되지 않는다.
    """
    terms = extract_terms(question)
    if not terms:
        return 0.0, []
    matched = [t for t in terms if t in chunk_text]
    return len(matched) / len(terms), matched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="실패 CSV 경로 (기본: 최신)")
    ap.add_argument("--type", default=None, help="문서 유형 필터 (약관/설명서/FAQ)")
    ap.add_argument("--detail", type=int, default=6, help="상세 출력 건수")
    args = ap.parse_args()

    path = Path(args.file) if args.file else find_latest("failures_*.csv")
    if not path or not path.exists():
        sys.exit("실패 CSV가 없습니다. 먼저 17_evaluate.py 를 실행하세요.")

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    chunks = load_chunks()

    print("=" * 76)
    print(f"실패 케이스 분석  ({path.name})")
    print("=" * 76)
    print(f"  전체 실패 {len(rows)}건")

    # --- 문서 유형 x 실패 유형 ---
    cross: Counter = Counter()
    for r in rows:
        cross[(r["doc_type"], r["failure_type"])] += 1

    print("\n  [문서유형 x 실패유형]")
    print(f"    {'':<10} {'순위낮음':>8} {'20위밖':>8}   계")
    for dt in sorted({r["doc_type"] for r in rows}):
        low = cross[(dt, "순위 낮음")]
        out = cross[(dt, "20위 밖")]
        print(f"    {dt:<10} {low:>8} {out:>8}   {low + out:>3}")

    # --- 난이도별 ---
    print("\n  [난이도별 실패]")
    for d, n in Counter(r["difficulty"] for r in rows).most_common():
        print(f"    {d:<10} {n:>3}건")

    # --- 어휘 매칭 가능성 ---
    if args.type:
        rows = [r for r in rows if r["doc_type"] == args.type]
        print(f"\n  ({args.type} 만 필터링: {len(rows)}건)")

    print("\n" + "=" * 76)
    print("어휘 매칭 가능성 (BM25 적용 시 기대 효과)")
    print("=" * 76)
    print("  질의의 단어가 정답 청크에 실제로 등장하는 비율을 본다.")
    print("  높으면 BM25로 잡을 수 있고, 낮으면 표현 격차라 질의 확장이 필요하다.")
    print()

    analyzed = []
    for r in rows:
        rel_ids = [c for c in r["relevant_chunk_ids"].split("|") if c]
        best_ratio, best_terms, best_id = 0.0, [], ""
        for cid in rel_ids:
            text = chunks.get(cid, {}).get("text", "")
            if not text:
                continue
            ratio, matched = lexical_overlap(r["question"], text)
            if ratio > best_ratio:
                best_ratio, best_terms, best_id = ratio, matched, cid
        analyzed.append((r, best_ratio, best_terms, best_id))

    buckets = {"높음 (0.5+)": 0, "중간 (0.25~0.5)": 0, "낮음 (0.25 미만)": 0}
    for _, ratio, _, _ in analyzed:
        if ratio >= 0.5:
            buckets["높음 (0.5+)"] += 1
        elif ratio >= 0.25:
            buckets["중간 (0.25~0.5)"] += 1
        else:
            buckets["낮음 (0.25 미만)"] += 1

    for k, v in buckets.items():
        share = v / len(analyzed) if analyzed else 0
        note = ""
        if k.startswith("높음"):
            note = "  <- BM25로 개선 가능성 큼"
        elif k.startswith("낮음"):
            note = "  <- 표현 격차. 질의 확장/재순위화 필요"
        print(f"    {k:<18} {v:>3}건 ({share:.0%}){note}")

    # --- 상세 ---
    analyzed.sort(key=lambda x: -x[1])

    print("\n" + "=" * 76)
    print(f"어휘 매칭이 높은 사례 (BM25 유력) — 상위 {args.detail}건")
    print("=" * 76)
    for r, ratio, terms, cid in analyzed[:args.detail]:
        print(f"\n  [{r['query_id']}] {r['doc_type']} / {r['difficulty']} / "
              f"{'20위 밖' if r['hit_rank'] == '미발견' else r['hit_rank'] + '위'}")
        print(f"    Q: {r['question'][:66]}")
        print(f"    어휘 일치율 {ratio:.0%}  일치 어휘: {', '.join(terms[:6])}")
        src = chunks.get(cid, {})
        if src:
            print(f"    정답: {src.get('doc_display_name', '')[:52]}")

    print("\n" + "=" * 76)
    print(f"어휘 매칭이 낮은 사례 (표현 격차) — 하위 {args.detail}건")
    print("=" * 76)
    for r, ratio, terms, cid in analyzed[-args.detail:]:
        print(f"\n  [{r['query_id']}] {r['doc_type']} / {r['difficulty']} / "
              f"{'20위 밖' if r['hit_rank'] == '미발견' else r['hit_rank'] + '위'}")
        print(f"    Q: {r['question'][:66]}")
        print(f"    어휘 일치율 {ratio:.0%}")
        src = chunks.get(cid, {})
        if src:
            print(f"    정답: {src.get('doc_display_name', '')[:52]}")
            body = src.get("text", "")[:130].replace("\n", " ")
            print(f"    내용: {body}...")
        # 실제로 검색된 것과 비교
        top1 = r["top5_retrieved"].split("|")[0] if r.get("top5_retrieved") else ""
        got = chunks.get(top1, {})
        if got:
            print(f"    1위로 검색된 것: {got.get('doc_display_name', '')[:48]}")

    print("\n" + "=" * 76)


if __name__ == "__main__":
    main()
