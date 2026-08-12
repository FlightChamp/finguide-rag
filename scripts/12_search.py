"""
12_search.py

목적
----
구축한 인덱스로 검색해 결과를 눈으로 확인한다.

정량 지표(Recall@5 등)를 재기 전에 이 단계가 필요하다. "이 질문에 이게
나오네, 이건 좀 이상한데?" 하는 감각이 이후 개선의 실마리가 되고,
평가셋을 만들 때 어떤 질문이 어려운지 판단하는 기준도 여기서 생긴다.

사용법
-----
    python scripts/12_search.py                        # 대화형
    python scripts/12_search.py -q "발행어음 중도해지"    # 단발 질의
    python scripts/12_search.py --model bge-m3         # 모델 지정
    python scripts/12_search.py --demo                 # 준비된 질문 일괄 실행
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.embedding import MODELS, Embedder, FaissStore  # noqa: E402

INDEX_ROOT = PROJECT_ROOT / "data" / "indexes" / "faiss"

# 검색 품질을 가늠하기 위한 표본 질문.
# 실제 영업점 직원이 던질 법한 형태로 구성했다.
DEMO_QUERIES = [
    # 상품명이 명시된 질의 (BM25가 강한 유형)
    "주택청약종합저축 중도해지하면 이자가 어떻게 되나요",
    "마이데이터 서비스는 몇 세부터 가입할 수 있나요",
    # 상품명 없이 상황만 설명한 질의 (Dense가 강한 유형)
    "예금을 만기 전에 깨면 손해가 큰가요",
    "보이스피싱 피해를 막으려면 계좌에 어떤 제한을 걸 수 있나요",
    # 조건·절차를 묻는 질의
    "대출 신청할 때 어떤 서류를 준비해야 하나요",
    "환율이 오르면 선물환 계약에서 손실이 나나요",
    # 근거가 없을 가능성이 높은 질의 (거절 대상 후보)
    "타행 대비 금리 경쟁력은 어떤가요",
    "한도약정수수료 요율은 몇 퍼센트인가요",
]


def show_hits(query: str, hits, verbose: bool = False) -> None:
    print()
    print("=" * 74)
    print(f"질의: {query}")
    print("=" * 74)

    if not hits:
        print("  결과 없음")
        return

    for h in hits:
        # 점수 구간을 시각적으로 표시한다. 임계값 감각을 잡는 데 도움이 된다.
        bar = "█" * int(h.score * 20)
        print(f"\n  [{h.rank}] {h.score:.4f} {bar}")
        print(f"      {h.citation}")
        print(f"      ({h.doc_type} / {h.category} / {h.chunk_id})")

        body = h.text.replace("\n", " ")
        limit = 400 if verbose else 180
        print(f"      {body[:limit]}{'...' if len(body) > limit else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small", choices=list(MODELS))
    ap.add_argument("-q", "--query", default=None, help="단발 질의")
    ap.add_argument("-k", "--top-k", type=int, default=5)
    ap.add_argument("--demo", action="store_true", help="표본 질문 일괄 실행")
    ap.add_argument("-v", "--verbose", action="store_true", help="본문을 길게 표시")
    args = ap.parse_args()

    index_dir = INDEX_ROOT / args.model
    if not (index_dir / "config.json").exists():
        sys.exit(
            f"{index_dir} 에 인덱스가 없습니다.\n"
            f"먼저 실행하세요: python scripts/11_build_index.py --model {args.model}"
        )

    print(f"인덱스 로딩: {args.model}")
    store = FaissStore.load(index_dir)
    embedder = Embedder(args.model)
    print(f"  {store}")

    def run(query: str) -> None:
        qv = embedder.encode_queries([query])
        hits = store.search(qv, top_k=args.top_k)
        show_hits(query, hits, args.verbose)

    # --- 표본 질문 일괄 ---
    if args.demo:
        for q in DEMO_QUERIES:
            run(q)
        print()
        print("=" * 74)
        print("점수 해석 참고")
        print("=" * 74)
        print("  1위 점수가 낮게 나온 질의는 근거가 없거나 검색이 실패한 경우다.")
        print("  이 분포가 나중에 거절(refusal) 임계값을 정하는 근거가 된다.")
        return

    # --- 단발 질의 ---
    if args.query:
        run(args.query)
        return

    # --- 대화형 ---
    print("\n질문을 입력하세요. 종료: 빈 입력 또는 Ctrl+C\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            break
        if not query:
            print("종료")
            break
        run(query)
        print()


if __name__ == "__main__":
    main()
