"""
08_benchmark_embedding.py

목적
----
임베딩 모델을 실제 문서 텍스트로 돌려 CPU 처리 속도를 측정한다.
전체 인덱스 구축에 몇 분이 걸리는지 미리 알아야, 반복 실험이 현실적인지
판단하고 필요하면 더 작은 모델로 교체할 수 있다.

첫 실행 시 모델 파일(약 2.2GB)을 내려받는다.
캐시 위치: C:\\Users\\<사용자>\\.cache\\huggingface

사용법
-----
    python scripts/08_benchmark_embedding.py
    python scripts/08_benchmark_embedding.py --small    # 경량 모델도 함께 비교
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = PROJECT_ROOT / "data" / "interim" / "parsed" / "documents.jsonl"

# 청킹 후 예상되는 청크 수. 문서 304건 기준 대략치이며,
# 실제 파이프라인 규모를 가늠하기 위한 값이다.
ESTIMATED_CHUNKS = 4000

MODELS = {
    "bge-m3": "BAAI/bge-m3",                                   # 568M, 8192토큰, 1024차원
    "e5-small": "intfloat/multilingual-e5-small",              # 118M, 512토큰, 384차원
}


def show_environment() -> None:
    import torch

    print("=" * 66)
    print("실행 환경")
    print("=" * 66)
    print(f"  OS          : {platform.system()} {platform.release()}")
    print(f"  CPU         : {platform.processor()[:50]}")
    print(f"  논리 코어    : {__import__('os').cpu_count()}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA 사용   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    else:
        print("  -> CPU로 실행합니다. GPU가 있다면 CUDA 빌드 설치로 가속 가능합니다.")
    print(f"  스레드      : {torch.get_num_threads()}")


def load_samples(n: int = 64) -> list[str]:
    """실제 문서에서 청크 크기의 텍스트 조각을 뽑는다.

    무작위 문자열로 재면 토큰 수가 실제와 달라 측정이 왜곡된다.
    한국어 금융 문서의 실제 텍스트로 재야 의미가 있다.
    """
    if not DOCS_PATH.exists():
        sys.exit(f"{DOCS_PATH} 없음. 먼저 06_build_documents.py 를 실행하세요.")

    rows = [json.loads(line) for line in DOCS_PATH.open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r["is_parsable"] and r.get("text")]

    samples: list[str] = []
    for r in rows:
        text = r["text"]
        # 목표 청크 크기(300~600자)에 맞춰 잘라낸다
        for start in range(0, min(len(text), 3000), 500):
            piece = text[start:start + 500].strip()
            if len(piece) > 100:
                samples.append(piece)
            if len(samples) >= n:
                return samples
    return samples


def benchmark(model_name: str, samples: list[str], batch_size: int = 8) -> dict:
    from sentence_transformers import SentenceTransformer

    print(f"\n{'=' * 66}")
    print(f"모델: {model_name}")
    print("=" * 66)

    print("  로딩 중... (첫 실행 시 다운로드가 발생합니다)")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_time = time.perf_counter() - t0
    print(f"  로딩 완료: {load_time:.1f}초")

    dim = model.get_sentence_embedding_dimension()
    max_seq = model.max_seq_length
    print(f"  차원: {dim} / 최대 입력: {max_seq} 토큰")

    # 첫 호출은 초기화 비용이 섞이므로 워밍업 후 측정한다
    model.encode(samples[:2], batch_size=2, show_progress_bar=False)

    print(f"\n  {len(samples)}개 인코딩 중...")
    t0 = time.perf_counter()
    vectors = model.encode(
        samples,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0

    per_item = elapsed / len(samples)
    per_sec = len(samples) / elapsed
    projected = ESTIMATED_CHUNKS * per_item

    print(f"  소요       : {elapsed:.1f}초")
    print(f"  개당       : {per_item * 1000:.0f}ms")
    print(f"  처리량     : {per_sec:.1f}개/초")
    print(f"  벡터 shape : {vectors.shape}")
    print(f"\n  >> 청크 {ESTIMATED_CHUNKS:,}개 예상 소요: {projected / 60:.1f}분")

    # 검색 시점의 질의 임베딩 지연도 측정한다.
    # 사용자가 체감하는 것은 이쪽이다.
    query = "발행어음을 중도해지하면 이자는 어떻게 되나요?"
    t0 = time.perf_counter()
    for _ in range(5):
        model.encode([query], show_progress_bar=False)
    query_ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"  >> 질의 1건 임베딩 지연: {query_ms:.0f}ms")

    return {
        "model": model_name,
        "dim": dim,
        "max_seq": max_seq,
        "per_item_ms": per_item * 1000,
        "projected_min": projected / 60,
        "query_ms": query_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="경량 모델도 함께 측정")
    ap.add_argument("--n", type=int, default=64, help="측정에 쓸 샘플 수")
    ap.add_argument("--batch", type=int, default=8, help="배치 크기")
    args = ap.parse_args()

    show_environment()

    samples = load_samples(args.n)
    lengths = [len(s) for s in samples]
    print(f"\n  샘플 {len(samples)}개 / 평균 {sum(lengths) // len(lengths)}자")

    results = [benchmark(MODELS["bge-m3"], samples, args.batch)]

    if args.small:
        results.append(benchmark(MODELS["e5-small"], samples, args.batch))

    if len(results) > 1:
        print(f"\n{'=' * 66}")
        print("비교")
        print("=" * 66)
        print(f"  {'모델':<38} {'차원':>6} {'개당':>9} {'전체':>9}")
        for r in results:
            name = r["model"].split("/")[-1]
            print(f"  {name:<38} {r['dim']:>6} {r['per_item_ms']:>7.0f}ms {r['projected_min']:>7.1f}분")
        print("\n  경량 모델은 빠르지만 최대 입력이 512토큰이라")
        print("  긴 약관 조항이 잘릴 수 있다. 속도가 감당 가능하면 bge-m3 를 쓴다.")

    print(f"\n{'=' * 66}")
    print("판단 기준")
    print("=" * 66)
    print("  전체 5분 이내  : 그대로 진행")
    print("  5~20분         : 감당 가능. 캐시를 두고 재실행을 줄인다")
    print("  20분 초과      : 경량 모델 검토 또는 GPU 환경 고려")
    print("=" * 66)


if __name__ == "__main__":
    main()
