"""
07_inspect_issues.py

06_build_documents.py 가 보고한 두 가지 문제를 상세히 확인한다.

1. 본문 해시 중복 — 파일명은 다른데 내용이 완전히 같은 문서
   크롤링 시 잘못된 파일이 저장됐을 가능성이 있다.

2. 버전 판정 결과 — is_latest=False 로 표시된 문서
   제목이 같다는 이유로 별개 문서가 잘못 묶였을 수 있다.

사용법
-----
    python scripts/07_inspect_issues.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = PROJECT_ROOT / "data" / "interim" / "parsed" / "documents.jsonl"

if not DOCS_PATH.exists():
    sys.exit(f"{DOCS_PATH} 없음. 먼저 06_build_documents.py 를 실행하세요.")

rows = [json.loads(line) for line in DOCS_PATH.open(encoding="utf-8") if line.strip()]
pdfs = [r for r in rows if r["doc_type"] != "FAQ" and r["is_parsable"]]

print("=" * 70)
print("1. 본문 해시 중복")
print("=" * 70)

by_hash: dict[str, list[dict]] = defaultdict(list)
for r in pdfs:
    if r["content_hash"]:
        by_hash[r["content_hash"]].append(r)

dups = {h: v for h, v in by_hash.items() if len(v) > 1}

if not dups:
    print("  없음")
else:
    for i, (h, group) in enumerate(dups.items(), 1):
        print(f"\n  [중복 {i}] 해시 {h[:16]}...  ({len(group)}건)")
        for r in group:
            print(f"    파일   : {r['orig_filename']}")
            print(f"    제목   : {r['title']}")
            print(f"    부제   : {r['subtitle']}")
            print(f"    분량   : {r['char_count']}자 / {r['page_count']}쪽")
            print(f"    시행일 : {r['effective_date']}  심의필: {r['compliance_no']}")
            print()

print("=" * 70)
print("2. 구버전으로 표시된 문서 (is_latest=False)")
print("=" * 70)

old = [r for r in rows if not r["is_latest"]]

if not old:
    print("  없음")
else:
    # 어떤 문서로 대체됐는지 함께 보여준다
    by_id = {r["doc_id"]: r for r in rows}
    for r in old:
        newer = by_id.get(r["superseded_by"] or "", {})
        print(f"\n  [구버전] {r['orig_filename']}")
        print(f"    제목   : {r['title']}")
        print(f"    부제   : {r['subtitle']}")
        print(f"    시행일 : {r['effective_date']}")
        print(f"    -> 대체: {newer.get('orig_filename', '?')}")
        print(f"       제목: {newer.get('title', '?')}")
        print(f"       부제: {newer.get('subtitle', '?')}")
        print(f"       시행일: {newer.get('effective_date', '?')}")
        same = r["content_hash"] == newer.get("content_hash")
        print(f"       본문 동일 여부: {'같음 (중복 의심)' if same else '다름 (개정으로 보임)'}")

print()
print("=" * 70)
print("3. 부제가 비어 있는 문서 (버전 오판 위험)")
print("=" * 70)

no_sub = [r for r in pdfs if not r["subtitle"]]
print(f"  {len(no_sub)}건 / 전체 {len(pdfs)}건")

# 제목이 같은데 부제가 없는 문서들은 잘못 묶일 위험이 있다
title_groups: dict[str, list[dict]] = defaultdict(list)
for r in no_sub:
    title_groups[r["title"]].append(r)

risky = {t: v for t, v in title_groups.items() if len(v) > 1}
if risky:
    print("\n  제목이 같고 부제가 없어 구별이 어려운 묶음:")
    for title, group in risky.items():
        print(f"\n    [{title}] {len(group)}건")
        for r in group:
            print(f"      - {r['orig_filename'][:62]}")
            print(f"        {r['effective_date']} / {r['char_count']}자")
else:
    print("  위험 묶음 없음")

print()
print("=" * 70)
