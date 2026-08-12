"""
10_inspect_chunks.py

09_build_chunks.py 결과에서 이상 청크를 확인한다.

점검 항목
--------
1. 최대 크기(--max)를 초과한 청크 — force_split이 동작하지 않은 경우
2. 최소 크기 미만 청크 — 필터가 새는 경우
3. e5-small 512토큰 초과 청크 — 임베딩 시 잘리는 대상
4. 문서당 청크 수 이상치 — 청킹이 과하거나 부족한 문서

사용법
-----
    python scripts/10_inspect_chunks.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"

MAX_CHARS = 900
MIN_CHARS = 100
E5_LIMIT = 512

if not CHUNKS_PATH.exists():
    sys.exit(f"{CHUNKS_PATH} 없음. 먼저 09_build_chunks.py 를 실행하세요.")

chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]
print(f"청크 {len(chunks):,}개 로드\n")


def preview(text: str, n: int = 100) -> str:
    """줄바꿈을 기호로 바꿔 한 줄로 보여준다."""
    return text[:n].replace("\n", " ⏎ ")


# ------------------------------------------------------------------
print("=" * 70)
print(f"1. 최대 크기({MAX_CHARS}자) 초과 청크")
print("=" * 70)

over = sorted(
    [c for c in chunks if c["char_count"] > MAX_CHARS],
    key=lambda c: -c["char_count"],
)
print(f"  {len(over)}개\n")

for c in over[:5]:
    print(f"  [{c['chunk_id']}] {c['char_count']}자 / {c.get('tokens_e5-small', '?')}토큰")
    print(f"    문서: {c['doc_display_name'][:50]}")
    print(f"    구조: {c['metadata'].get('structure')}")
    print(f"    본문: {preview(c['text'], 150)}")
    print()

if len(over) > 5:
    print(f"  ... 외 {len(over) - 5}개\n")


# ------------------------------------------------------------------
print("=" * 70)
print(f"2. 최소 크기({MIN_CHARS}자) 미만 청크")
print("=" * 70)

under = sorted([c for c in chunks if c["char_count"] < MIN_CHARS], key=lambda c: c["char_count"])
print(f"  {len(under)}개\n")

for c in under[:8]:
    print(f"  [{c['chunk_id']}] {c['char_count']}자")
    print(f"    문서: {c['doc_display_name'][:50]}")
    print(f"    본문: {preview(c['text'])}")
    print()


# ------------------------------------------------------------------
print("=" * 70)
print(f"3. e5-small 한계({E5_LIMIT}토큰) 초과 청크")
print("=" * 70)

tok_over = sorted(
    [c for c in chunks if c.get("tokens_e5-small", 0) > E5_LIMIT],
    key=lambda c: -c.get("tokens_e5-small", 0),
)
print(f"  {len(tok_over)}개 — 임베딩 시 뒷부분이 잘린다\n")

for c in tok_over:
    print(f"  [{c['chunk_id']}] {c['char_count']}자 / {c['tokens_e5-small']}토큰")
    print(f"    문서: {c['doc_display_name'][:50]}")
    print(f"    본문: {preview(c['text'], 120)}")
    print()


# ------------------------------------------------------------------
print("=" * 70)
print("4. 문서당 청크 수 분포")
print("=" * 70)

per_doc: Counter = Counter()
doc_name: dict[str, str] = {}
for c in chunks:
    per_doc[c["doc_id"]] += 1
    doc_name[c["doc_id"]] = c["doc_display_name"]

counts = sorted(per_doc.values())
print(f"  최소 / 중앙 / 최대 : {counts[0]} / {counts[len(counts)//2]} / {counts[-1]}")

print("\n  [청크가 가장 많은 문서]")
for doc_id, n in per_doc.most_common(5):
    print(f"    {n:>3}개  {doc_name[doc_id][:52]}")

print("\n  [청크가 1개뿐인 문서 (FAQ 제외)]")
single = [
    (d, doc_name[d]) for d, n in per_doc.items()
    if n == 1 and not d.startswith("hana_faq")
]
print(f"    {len(single)}건")
for _, name in single[:8]:
    print(f"    - {name[:56]}")


# ------------------------------------------------------------------
print()
print("=" * 70)
print("5. 검색 품질 표본 확인")
print("=" * 70)
print("  실제 인덱싱에 들어가는 텍스트(문서명 접두어 포함)를 확인한다.\n")

import random

random.seed(42)
for c in random.sample(chunks, 3):
    print(f"  [{c['chunk_id']}] {c['char_count']}자")
    print(f"    출처: {c['citation']}")
    print(f"    ---")
    text = c["text"][:220].replace("\n", "\n         ")
    print(f"         {text}")
    print()

print("=" * 70)
