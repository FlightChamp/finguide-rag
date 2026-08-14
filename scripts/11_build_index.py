"""
11_build_index.py

목적
----
청크를 임베딩해 FAISS 인덱스를 구축한다.

인덱스는 "실험 이름" 단위로 분리해 보관한다. 청킹 전략이나 임베딩
모델을 바꿀 때 기존 인덱스를 덮어쓰면 비교 실험이 불가능해지기 때문이다.
결과가 나쁠 때 되돌릴 수도 없다.

    data/indexes/faiss/e5-small/              기본 (길이 기반 청킹)
    data/indexes/faiss/e5-small_structural/   구조별 청킹
    data/indexes/faiss/bge-m3/                모델 비교용

실험 이름은 --name 으로 지정하며, 생략하면 청크 파일명에서 자동 유추한다.

무엇을 임베딩하는가
----------------
청크의 text 가 아니라 indexable_text 를 임베딩한다. 후자는 문서명과
조항명이 접두어로 붙은 형태다. 조항 본문은 보통 "이 예금은..." 처럼
시작해 상품명이 등장하지 않으므로, 접두어가 없으면 "발행어음 중도해지"
같은 질의와 매칭되지 않는다.

사용법
-----
    python scripts/11_build_index.py                  # e5-small (기본)
    python scripts/11_build_index.py --model bge-m3   # bge-m3
    python scripts/11_build_index.py --limit 100      # 소량 시험
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.embedding import MODELS, Embedder, FaissStore  # noqa: E402

CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"
INDEX_ROOT = PROJECT_ROOT / "data" / "indexes" / "faiss"


def load_chunks(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} 없음. 먼저 09_build_chunks.py 를 실행하세요.")

    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return rows[:limit] if limit else rows


def resolve_index_name(model_key: str, chunks_path: Path, explicit: str | None) -> str:
    """인덱스 디렉토리 이름을 정한다.

    명시하지 않으면 청크 파일명에서 유추한다.
        chunks.jsonl            -> e5-small
        chunks_structural.jsonl -> e5-small_structural

    이렇게 하면 실험을 늘려도 인덱스가 서로 덮어쓰지 않는다.
    """
    if explicit:
        return explicit

    stem = chunks_path.stem            # chunks_structural
    suffix = stem.removeprefix("chunks").lstrip("_")
    return f"{model_key}_{suffix}" if suffix else model_key


def build_indexable_text(row: dict) -> str:
    """인덱싱 대상 텍스트를 만든다.

    Chunk.indexable_text 와 같은 규칙이다. JSONL에서 복원하므로
    여기서 다시 구성한다.
    """
    parts = [row.get("doc_display_name", ""), row.get("section", "")]
    prefix = " ".join(p for p in parts if p and p != "FAQ")
    text = row.get("text", "")
    return f"{prefix}\n{text}" if prefix else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small", choices=list(MODELS), help="임베딩 모델")
    ap.add_argument("--batch", type=int, default=16, help="배치 크기")
    ap.add_argument("--limit", type=int, default=None, help="청크 수 제한(시험용)")
    ap.add_argument("--chunks", default=None,
                    help="청크 JSONL 경로 (기본: data/interim/chunks.jsonl)")
    ap.add_argument("--name", default=None,
                    help="인덱스 이름. 생략 시 모델명 + 청크 파일명으로 자동 결정")
    ap.add_argument("--force", action="store_true",
                    help="기존 인덱스가 있어도 덮어쓴다")
    args = ap.parse_args()

    spec = MODELS[args.model]

    print("=" * 66)
    print("인덱스 구축")
    print("=" * 66)
    print(f"  모델   : {spec.model_id}")
    print(f"  차원   : {spec.dim} / 최대 {spec.max_tokens}토큰")
    if spec.passage_prefix:
        print(f"  접두어 : 문서 '{spec.passage_prefix}' / 질의 '{spec.query_prefix}'")
    else:
        print("  접두어 : 없음")

    chunks_path = Path(args.chunks) if args.chunks else CHUNKS_PATH
    index_name = resolve_index_name(args.model, chunks_path, args.name)
    out_dir = INDEX_ROOT / index_name

    print(f"  청크   : {chunks_path.name}")
    print(f"  인덱스 : {index_name}")

    # 기존 인덱스를 실수로 덮어쓰면 비교 실험이 불가능해진다.
    # 베이스라인을 잃고 나면 청킹부터 다시 돌려야 한다.
    if (out_dir / "config.json").exists() and not args.force:
        sys.exit(
            f"\n{out_dir} 에 이미 인덱스가 있습니다.\n"
            f"덮어쓰려면 --force, 다른 이름으로 만들려면 --name 을 쓰세요."
        )

    # --- 청크 로드 ---
    rows = load_chunks(chunks_path, args.limit)
    print(f"\n  청크 {len(rows):,}개 로드")

    texts = [build_indexable_text(r) for r in rows]

    # 접두어가 붙은 상태에서 토큰 한계를 넘는지 확인한다.
    # 청킹 단계에서 쟀지만 'passage: ' 가 추가되므로 다시 본다.
    tok_field = f"tokens_{args.model}"
    if tok_field in rows[0]:
        over = [r for r in rows if r.get(tok_field, 0) > spec.max_tokens - 10]
        if over:
            print(f"  !! 토큰 한계 근접 청크 {len(over)}개 — 뒷부분이 잘릴 수 있습니다.")

    # --- 임베딩 ---
    embedder = Embedder(args.model, batch_size=args.batch)
    print(f"\n  임베딩 중... (첫 실행 시 모델 다운로드가 발생합니다)")

    t0 = time.perf_counter()
    vectors = embedder.encode_passages(texts, show_progress=True)
    elapsed = time.perf_counter() - t0

    print(f"\n  완료: {elapsed:.1f}초 ({elapsed / len(texts) * 1000:.0f}ms/개)")
    print(f"  벡터 shape: {vectors.shape}")

    # 정규화가 제대로 됐는지 확인한다.
    # L2 노름이 1이 아니면 내적이 코사인 유사도가 아니게 된다.
    norms = (vectors ** 2).sum(axis=1) ** 0.5
    print(f"  L2 노름 범위: {norms.min():.4f} ~ {norms.max():.4f} (1.0이어야 정상)")

    # --- 인덱스 구축 ---
    store = FaissStore(dim=spec.dim, model_key=args.model)
    store.build(vectors, rows)

    store.save(out_dir)

    # --- 자체 검증 ---
    # 저장한 인덱스를 다시 읽어 정상 동작하는지 확인한다.
    # 매핑이 깨진 채로 다음 단계로 넘어가는 것을 막는다.
    print("\n  저장된 인덱스 검증 중...")
    reloaded = FaissStore.load(out_dir)
    print(f"    로드 성공: {reloaded}")

    # 첫 청크로 자기 자신을 검색한다. 1위로 나와야 정상이다.
    probe = embedder.encode_passages([texts[0]], show_progress=False)
    hits = reloaded.search(probe, top_k=1)
    ok = hits and hits[0].chunk_id == rows[0]["chunk_id"]
    print(f"    자기 검색 테스트: {'통과' if ok else '실패'} "
          f"(score={hits[0].score:.4f})" if hits else "    실패")

    if not ok:
        print("    !! 매핑이 어긋났을 수 있습니다. 인덱스를 재구축하세요.")

    print(f"\n저장 → {out_dir.relative_to(PROJECT_ROOT)}")
    print("=" * 66)


if __name__ == "__main__":
    main()
