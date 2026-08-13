"""
19_build_bm25.py

목적
----
청크를 형태소 분석해 BM25 인덱스를 구축한다.

무엇을 색인하는가
---------------
dense 인덱스와 동일하게 indexable_text(문서명 + 조항명 + 본문)를 쓴다.
조항 본문은 "이 예금은..." 처럼 시작해 상품명이 등장하지 않는 경우가
많으므로, 접두어가 없으면 "발행어음 중도해지" 같은 질의와 매칭되지 않는다.

두 인덱스가 같은 텍스트를 보도록 맞춰야 하이브리드 결합이 일관된다.

출력
----
data/indexes/bm25/bm25.pkl

사용법
-----
    python scripts/19_build_bm25.py
    python scripts/19_build_bm25.py --k1 1.5 --b 0.6   # 파라미터 조정
    python scripts/19_build_bm25.py --inspect          # 토큰화 결과 확인
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.retrieval import BM25Store, KiwiTokenizer, explain_tokens  # noqa: E402

CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "bm25"


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} 없음. 먼저 09_build_chunks.py 를 실행하세요.")
    return [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]


def build_indexable_text(row: dict) -> str:
    """dense 인덱스와 동일한 규칙으로 색인 텍스트를 만든다."""
    parts = [row.get("doc_display_name", ""), row.get("section", "")]
    prefix = " ".join(p for p in parts if p and p != "FAQ")
    text = row.get("text", "")
    return f"{prefix}\n{text}" if prefix else text


def inspect(rows: list[dict], n: int = 5) -> None:
    """토큰화 결과를 눈으로 확인한다.

    형태소 분석이 잘못되면 검색이 조용히 나빠진다. 색인 전에 표본을
    확인해 조사가 제대로 떨어지는지, 상품명이 쪼개지지 않는지 본다.
    """
    tok = KiwiTokenizer()
    print("\n" + "=" * 72)
    print("토큰화 표본")
    print("=" * 72)

    import random
    for row in random.Random(42).sample(rows, min(n, len(rows))):
        text = build_indexable_text(row)
        print(f"\n  [{row['chunk_id']}] {row['doc_type']}")
        print(f"    원문: {text[:100].replace(chr(10), ' ')}...")
        tokens = tok.tokenize(text)
        print(f"    토큰({len(tokens)}개): {' · '.join(tokens[:24])}")

    print("\n  [질의 토큰화]")
    for q in [
        "기업 인터넷뱅킹에서 급여 이체는 언제까지 할 수 있나요?",
        "하나은행 입출금 예금 이율은 바뀌면 언제부터 적용돼요?",
        "돈 빌린 사람이 약속을 안 지켜서 은행이 보증인에게 알릴 때",
    ]:
        print(f"    {q[:44]}")
        print(f"      -> {explain_tokens(tok, q)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", type=float, default=1.2, help="용어 빈도 포화 계수")
    ap.add_argument("--b", type=float, default=0.75, help="문서 길이 정규화 강도")
    ap.add_argument("--inspect", action="store_true", help="토큰화 표본만 확인")
    args = ap.parse_args()

    print("=" * 72)
    print("BM25 인덱스 구축")
    print("=" * 72)

    rows = load_chunks()
    print(f"  청크 {len(rows):,}개 로드")

    if args.inspect:
        inspect(rows)
        print("\n  --inspect 이므로 인덱스를 만들지 않았습니다.")
        return

    texts = [build_indexable_text(r) for r in rows]
    chunk_ids = [r["chunk_id"] for r in rows]

    print(f"  파라미터: k1={args.k1}, b={args.b}")

    store = BM25Store(k1=args.k1, b=args.b)

    t0 = time.perf_counter()
    store.build(chunk_ids, texts, show_progress=True)
    elapsed = time.perf_counter() - t0

    print(f"  소요: {elapsed:.1f}초")

    store.save(INDEX_DIR)

    # --- 자체 검증 ---
    # 저장한 인덱스를 다시 읽어 검색이 되는지 확인한다.
    print("\n  저장된 인덱스 검증 중...")
    reloaded = BM25Store.load(INDEX_DIR)
    print(f"    로드 성공: {reloaded}")

    # 첫 청크의 본문 일부로 검색하면 자기 자신이 상위에 나와야 한다
    probe = rows[0]["text"][:80]
    hits = reloaded.search(probe, top_k=5)
    found = next((h.rank for h in hits if h.chunk_id == chunk_ids[0]), 0)
    print(f"    자기 검색 테스트: {'통과' if found == 1 else f'{found}위 (확인 필요)'}")

    # --- 어휘 통계 ---
    # 색인된 어휘의 분포를 본다. 특정 단어가 지나치게 흔하면 변별력이 없다.
    tok = KiwiTokenizer()
    sample = [tok.tokenize(t) for t in texts[:400]]
    vocab: Counter = Counter()
    for tokens in sample:
        vocab.update(set(tokens))   # 문서 빈도(DF)

    print(f"\n  [표본 400개 기준 어휘 통계]")
    print(f"    고유 어휘: {len(vocab):,}개")
    print(f"    문서당 평균 토큰: {sum(len(t) for t in sample) / len(sample):.0f}개")

    print("\n    가장 흔한 어휘 (변별력 낮음)")
    for word, df in vocab.most_common(12):
        print(f"      {word:<12} {df:>3}개 문서 ({df / len(sample):.0%})")

    print(f"\n저장 → {INDEX_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 72)
    print("다음: python scripts/20_tune_hybrid.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
