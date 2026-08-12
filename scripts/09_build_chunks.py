"""
09_build_chunks.py

목적
----
정규화된 문서 304건을 검색 단위(Chunk)로 분할하고, 각 청크의 실제
토큰 수를 측정한다.

토큰 수를 함께 재는 이유
---------------------
multilingual-e5-small의 최대 입력은 512토큰이다. 한국어는 대략 1자당
1~1.5토큰이므로 목표 청크 크기 500자가 512토큰을 넘길 수 있다. 초과한
청크는 뒷부분이 잘린 채 임베딩되므로 검색에서 누락된다.

몇 %가 초과하는지 실측해야 청크 크기를 조정할지, 모델을 바꿀지
판단할 수 있다. 추측으로 정할 문제가 아니다.

출력
----
data/interim/chunks.jsonl        전체 청크
data/interim/chunk_stats.csv     문서별 청킹 통계

사용법
-----
    python scripts/09_build_chunks.py
    python scripts/09_build_chunks.py --no-tokens    # 토큰 측정 생략 (빠름)
    python scripts/09_build_chunks.py --target 400   # 청크 크기 조정
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.chunking import ChunkerFactory  # noqa: E402
from finguide_rag.schema import Chunk, Document, DocType, Structure  # noqa: E402

DOCS_PATH = PROJECT_ROOT / "data" / "interim" / "parsed" / "documents.jsonl"
OUT_CHUNKS = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"
OUT_STATS = PROJECT_ROOT / "data" / "interim" / "chunk_stats.csv"

# 검증 대상 모델의 입력 한계
E5_SMALL_LIMIT = 512
BGE_M3_LIMIT = 8192


# ------------------------------------------------------------------
# 문서 로드
# ------------------------------------------------------------------


def load_documents() -> list[Document]:
    """documents.jsonl 을 Document 객체로 복원한다."""
    if not DOCS_PATH.exists():
        sys.exit(f"{DOCS_PATH} 없음. 먼저 06_build_documents.py 를 실행하세요.")

    from datetime import date

    docs: list[Document] = []
    for line in DOCS_PATH.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)

        doc = Document(
            doc_id=row["doc_id"],
            bank_code=row["bank_code"],
            source_path=row["source_path"],
            title=row.get("title", ""),
            subtitle=row.get("subtitle", ""),
            text=row.get("text", ""),
            doc_type=DocType(row["doc_type"]),
            category=row.get("category", ""),
            structure=Structure(row["structure"]),
            compliance_no=row.get("compliance_no", ""),
            is_latest=row.get("is_latest", True),
            superseded_by=row.get("superseded_by") or None,
            is_parsable=row.get("is_parsable", True),
            exclusion_reason=row.get("exclusion_reason", ""),
            page_count=row.get("page_count", 0),
            char_count=row.get("char_count", 0),
            source_url=row.get("source_url", ""),
            orig_filename=row.get("orig_filename", ""),
        )
        if row.get("effective_date"):
            doc.effective_date = date.fromisoformat(row["effective_date"])

        docs.append(doc)

    return docs


# ------------------------------------------------------------------
# 토큰 측정
# ------------------------------------------------------------------


def measure_tokens(chunks: list[Chunk]) -> dict[str, list[int]]:
    """두 모델의 토크나이저로 실제 토큰 수를 잰다.

    모델 전체가 아니라 토크나이저만 로드하므로 가볍고 빠르다.
    임베딩에 들어가는 것은 text가 아니라 indexable_text(문서명+조항명이
    접두어로 붙은 것)이므로 그것을 기준으로 측정한다.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  !! transformers 없음 — 토큰 측정을 건너뜁니다.")
        return {}

    result: dict[str, list[int]] = {}
    targets = {
        "e5-small": "intfloat/multilingual-e5-small",
        "bge-m3": "BAAI/bge-m3",
    }

    texts = [c.indexable_text for c in chunks]

    for label, model_id in targets.items():
        print(f"  {label} 토크나이저 로딩...")
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception as exc:
            print(f"    실패: {exc}")
            continue

        counts: list[int] = []
        for i in range(0, len(texts), 256):
            batch = texts[i:i + 256]
            encoded = tok(batch, add_special_tokens=True, truncation=False)
            counts.extend(len(ids) for ids in encoded["input_ids"])
        result[label] = counts
        print(f"    {len(counts)}개 측정 완료")

    return result


# ------------------------------------------------------------------
# 저장
# ------------------------------------------------------------------


def save_chunks(chunks: list[Chunk], token_counts: dict[str, list[int]]) -> None:
    OUT_CHUNKS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CHUNKS.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks):
            d = ch.to_dict()
            for label, counts in token_counts.items():
                if i < len(counts):
                    d[f"tokens_{label}"] = counts[i]
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def save_stats(docs: list[Document], by_doc: dict[str, list[Chunk]]) -> None:
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_STATS.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "doc_id", "doc_type", "structure", "title",
            "doc_chars", "n_chunks", "avg_chunk_chars", "max_chunk_chars",
        ])
        for doc in docs:
            chunks = by_doc.get(doc.doc_id, [])
            if not chunks:
                continue
            sizes = [c.char_count for c in chunks]
            w.writerow([
                doc.doc_id,
                doc.doc_type.value,
                doc.structure.value,
                doc.title[:40],
                doc.char_count,
                len(chunks),
                sum(sizes) // len(sizes),
                max(sizes),
            ])


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500, help="목표 청크 크기(자)")
    ap.add_argument("--max", type=int, default=900, help="최대 청크 크기(자)")
    ap.add_argument("--overlap", type=int, default=80, help="겹침 크기(자)")
    ap.add_argument("--no-tokens", action="store_true", help="토큰 측정 생략")
    args = ap.parse_args()

    print("=" * 68)
    print("청킹")
    print("=" * 68)
    print(f"  목표 {args.target}자 / 최대 {args.max}자 / 겹침 {args.overlap}자")

    docs = load_documents()
    parsable = [d for d in docs if d.is_parsable]
    print(f"  문서 {len(parsable)}건 (전체 {len(docs)}건 중)")

    factory = ChunkerFactory(
        target_chars=args.target,
        max_chars=args.max,
        overlap_chars=args.overlap,
    )

    all_chunks: list[Chunk] = []
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    empty_docs: list[str] = []

    print("\n  분할 중...")
    for doc in parsable:
        chunks = factory.chunk(doc)
        if not chunks:
            empty_docs.append(doc.orig_filename)
            continue
        all_chunks.extend(chunks)
        by_doc[doc.doc_id] = chunks

    print(f"  청크 {len(all_chunks):,}개 생성")

    if empty_docs:
        print(f"\n  !! 청크가 생성되지 않은 문서 {len(empty_docs)}건:")
        for name in empty_docs[:5]:
            print(f"     - {name[:56]}")

    # --- 토큰 측정 ---
    token_counts: dict[str, list[int]] = {}
    if not args.no_tokens:
        print("\n  토큰 수 측정 중...")
        token_counts = measure_tokens(all_chunks)

    # --- 저장 ---
    save_chunks(all_chunks, token_counts)
    save_stats(parsable, by_doc)

    # --- 요약 ---
    sizes = [c.char_count for c in all_chunks]
    sizes.sort()

    print("\n" + "=" * 68)
    print("요약")
    print("=" * 68)
    print(f"  총 청크        : {len(all_chunks):,}개")
    print(f"  문서당 평균     : {len(all_chunks) / len(by_doc):.1f}개")

    print("\n  [청크 크기(자)]")
    print(f"    최소 / 중앙 / 최대 : {sizes[0]} / {sizes[len(sizes)//2]} / {sizes[-1]}")
    print(f"    평균              : {sum(sizes)//len(sizes)}")

    print("\n  [유형별 청크 수]")
    by_type: Counter = Counter()
    for ch in all_chunks:
        by_type[ch.doc_type] += 1
    for k, v in by_type.most_common():
        print(f"    {k:<10} : {v:>6,}개")

    print("\n  [구조별 청크 수]")
    by_struct: Counter = Counter()
    for ch in all_chunks:
        by_struct[ch.metadata.get("structure", "?")] += 1
    for k, v in by_struct.most_common():
        print(f"    {k:<10} : {v:>6,}개")

    # --- 토큰 초과 분석 (핵심) ---
    if token_counts:
        print("\n" + "=" * 68)
        print("토큰 수 실측 — 모델 입력 한계 검증")
        print("=" * 68)

        for label, counts in token_counts.items():
            limit = E5_SMALL_LIMIT if label == "e5-small" else BGE_M3_LIMIT
            over = [c for c in counts if c > limit]
            srt = sorted(counts)
            ratio_over = len(over) / len(counts) * 100

            print(f"\n  [{label}]  한계 {limit} 토큰")
            print(f"    중앙값 / 평균 : {srt[len(srt)//2]} / {sum(srt)//len(srt)}")
            print(f"    상위 95% / 최대: {srt[int(len(srt)*0.95)]} / {srt[-1]}")
            print(f"    한계 초과      : {len(over):,}개 ({ratio_over:.1f}%)")

            if label == "e5-small":
                # 한국어 문자당 토큰 비율. 청크 크기 조정의 근거가 된다.
                ratio = sum(counts) / sum(sizes)
                print(f"    문자당 토큰비  : {ratio:.2f}")
                if ratio_over > 5:
                    safe = int(limit / ratio * 0.9)
                    print(f"    -> 초과 비율이 높습니다. --target {safe} 로 재실행을 권장합니다.")
                else:
                    print("    -> 초과 비율이 낮아 현재 설정을 유지해도 됩니다.")

    print(f"\n저장 → {OUT_CHUNKS.relative_to(PROJECT_ROOT)}")
    print(f"저장 → {OUT_STATS.relative_to(PROJECT_ROOT)}")
    print("=" * 68)


if __name__ == "__main__":
    main()
