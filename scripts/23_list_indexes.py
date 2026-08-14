"""
23_list_indexes.py

목적
----
구축된 인덱스 목록을 보여주고, 필요하면 기존 인덱스를 새 경로 규칙으로
옮긴다.

배경
----
초기에는 인덱스를 모델명 단위로만 저장했다.

    data/indexes/faiss/e5-small/
    data/indexes/bm25/bm25.pkl        <- 이름 구분 없음

청킹 전략을 바꾸는 실험을 시작하면서, 인덱스를 실험 단위로 분리해야
비교가 가능해졌다. BM25는 경로 자체가 바뀌었으므로 한 번 옮겨야 한다.

    data/indexes/bm25/default/bm25.pkl

사용법
-----
    python scripts/23_list_indexes.py            # 목록 확인
    python scripts/23_list_indexes.py --migrate  # 기존 BM25 인덱스 이동
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAISS_ROOT = PROJECT_ROOT / "data" / "indexes" / "faiss"
BM25_ROOT = PROJECT_ROOT / "data" / "indexes" / "bm25"
INTERIM = PROJECT_ROOT / "data" / "interim"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}TB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--migrate", action="store_true",
                    help="기존 BM25 인덱스를 default/ 로 이동")
    args = ap.parse_args()

    # --- 마이그레이션 ---
    legacy = BM25_ROOT / "bm25.pkl"
    if args.migrate:
        if not legacy.exists():
            print("이동할 기존 BM25 인덱스가 없습니다.")
        else:
            target_dir = BM25_ROOT / "default"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "bm25.pkl"
            if target.exists():
                print(f"{target} 이 이미 있습니다. 이동을 건너뜁니다.")
            else:
                shutil.move(str(legacy), str(target))
                print(f"이동 완료: {legacy.name} → default/{target.name}")
        print()

    # --- 청크 파일 ---
    print("=" * 70)
    print("청크 파일")
    print("=" * 70)
    chunk_files = sorted(INTERIM.glob("chunks*.jsonl"))
    if not chunk_files:
        print("  없음")
    for f in chunk_files:
        n = sum(1 for line in f.open(encoding="utf-8") if line.strip())
        print(f"  {f.name:<32} {n:>6,}청크  {human(f.stat().st_size):>8}")

    # --- 평가셋 ---
    print("\n" + "=" * 70)
    print("평가셋")
    print("=" * 70)
    eval_files = sorted(EVAL_DIR.glob("retrieval_eval*.csv"))
    for f in eval_files:
        n = sum(1 for _ in f.open(encoding="utf-8-sig")) - 1
        note = " (검수 초안)" if "draft" in f.name else ""
        print(f"  {f.name:<40} {n:>4}건{note}")

    # --- Dense 인덱스 ---
    print("\n" + "=" * 70)
    print("Dense 인덱스 (FAISS)")
    print("=" * 70)
    if not FAISS_ROOT.exists():
        print("  없음")
    else:
        found = False
        for d in sorted(FAISS_ROOT.iterdir()):
            cfg = d / "config.json"
            if not cfg.exists():
                continue
            found = True
            c = json.loads(cfg.read_text(encoding="utf-8"))
            print(f"  {d.name:<28} {c['n_vectors']:>6,}벡터  "
                  f"{c['dim']:>4}차원  {c.get('model_key', '?'):<10} "
                  f"{human(dir_size(d)):>8}")
        if not found:
            print("  없음")

    # --- Sparse 인덱스 ---
    print("\n" + "=" * 70)
    print("Sparse 인덱스 (BM25)")
    print("=" * 70)
    if legacy.exists():
        print(f"  !! {legacy.name} 이 옛 경로에 있습니다.")
        print("     --migrate 로 default/ 로 옮기세요.")
    if BM25_ROOT.exists():
        found = False
        for d in sorted(BM25_ROOT.iterdir()):
            if not d.is_dir() or not (d / "bm25.pkl").exists():
                continue
            found = True
            print(f"  {d.name:<28} {human(dir_size(d)):>8}")
        if not found and not legacy.exists():
            print("  없음")

    # --- 실험 조합 안내 ---
    print("\n" + "=" * 70)
    print("실험 실행 예시")
    print("=" * 70)
    print("""  베이스라인 (길이 기반 청킹)
    python scripts/17_evaluate.py --tag baseline
    python scripts/20_tune_hybrid.py --bm25 default --tag baseline

  구조별 청킹
    python scripts/11_build_index.py --chunks data/interim/chunks_structural.jsonl
    python scripts/19_build_bm25.py  --chunks data/interim/chunks_structural.jsonl
    python scripts/17_evaluate.py --index e5-small_structural \\
        --eval data/eval/retrieval_eval_structural.csv --tag structural
    python scripts/20_tune_hybrid.py --index e5-small_structural --bm25 structural \\
        --eval data/eval/retrieval_eval_structural.csv --tag structural""")
    print("=" * 70)


if __name__ == "__main__":
    main()
