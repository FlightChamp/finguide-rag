"""
06_build_documents.py

목적
----
원본 소스(PDF 108건 + FAQ 200건)를 전부 파싱해 공통 Document 스키마로
정규화하고, 문서 대장을 갱신한다.

수행 내용
--------
1. data/raw/hana/{desc,terms}/*.pdf  ->  Document
2. data/raw/hana/faq/faq_hana.jsonl  ->  Document (200건)
3. 파일명 날짜와 준법감시인 심의필 날짜를 대조해 불일치를 보고한다
   (마이데이터 문서에서 잘못된 파일이 저장된 사례가 있었으므로 상시 검증한다)
4. 동일 제목 문서를 탐지해 중복 수집 가능성을 보고한다
5. 결과를 JSONL과 CSV로 저장한다

출력
----
data/interim/parsed/documents.jsonl   전체 Document (본문 포함)
data/registry/document_registry.csv   갱신된 대장 (본문 제외)

사용법
-----
    python scripts/06_build_documents.py
    python scripts/06_build_documents.py --limit 10     # 앞 10건만 (빠른 확인)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.parsing import ParserFactory  # noqa: E402
from finguide_rag.schema import Document, DocType, Structure  # noqa: E402


# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hana"
OUT_JSONL = PROJECT_ROOT / "data" / "interim" / "parsed" / "documents.jsonl"
OUT_REGISTRY = PROJECT_ROOT / "data" / "registry" / "document_registry.csv"

# 대장에 기록할 컬럼. 본문(text)은 용량이 크므로 제외하고 JSONL에만 남긴다.
REGISTRY_COLUMNS = [
    "doc_id", "bank_code", "doc_type", "category",
    "title", "subtitle", "display_name",
    "structure", "effective_date",
    "compliance_no", "compliance_date",
    "is_parsable", "exclusion_reason", "title_confidence",
    "page_count", "char_count", "n_tables",
    "content_hash", "is_latest", "superseded_by",
    "source_url", "source_path", "orig_filename", "parsed_at",
]

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(message)s",
)


# ------------------------------------------------------------------
# 수집
# ------------------------------------------------------------------


def build_doc_id(pdf_path: Path, doc_type: str, category: str, seq: int) -> str:
    """대장용 doc_id를 만든다.

    기존 대장의 규칙(hana_desc_deposit_001)을 유지해 호환성을 지킨다.
    """
    return f"hana_{doc_type}_{category}_{seq:03d}"


def collect_pdfs(factory: ParserFactory, limit: int | None = None) -> list[Document]:
    """설명서·약관 PDF를 파싱한다."""
    documents: list[Document] = []
    counters: dict[tuple[str, str], int] = defaultdict(int)

    for kind in ("desc", "terms"):
        directory = RAW_DIR / kind
        if not directory.exists():
            print(f"  !! {directory} 없음 — 건너뜁니다.")
            continue

        pdfs = sorted(directory.glob("*.pdf"))
        if limit:
            pdfs = pdfs[:limit]

        print(f"\n[{kind}] {len(pdfs)}건")
        for i, path in enumerate(pdfs, 1):
            parser = factory.get_parser(path)
            if parser is None:
                print(f"  [{i:>3}] {path.name[:50]:<50} 처리 가능한 파서 없음")
                continue

            # doc_id에 카테고리가 필요하므로 파일명에서 먼저 뽑는다
            category = parser._detect_category(path.name)  # type: ignore[attr-defined]
            counters[(kind, category)] += 1
            doc_id = build_doc_id(path, kind, category, counters[(kind, category)])

            docs = parser.safe_parse(path, doc_id=doc_id)
            if not docs:
                print(f"  [{i:>3}] {path.name[:50]:<50} 파싱 실패")
                continue

            doc = docs[0]
            documents.append(doc)

            mark = ""
            if not doc.is_parsable:
                mark = f"  <-- 제외: {doc.exclusion_reason}"
            print(f"  [{i:>3}] {path.name[:50]:<50}{mark}")

    return documents


def collect_faq(factory: ParserFactory) -> list[Document]:
    """FAQ JSONL을 파싱한다."""
    path = RAW_DIR / "faq" / "faq_hana.jsonl"
    if not path.exists():
        print(f"\n  !! {path} 없음 — 건너뜁니다.")
        return []

    print(f"\n[faq] {path.name}")
    documents = factory.parse(path)
    print(f"  {len(documents)}건 파싱")
    return documents


# ------------------------------------------------------------------
# 검증
# ------------------------------------------------------------------


def verify_dates(documents: list[Document]) -> list[tuple[str, str, str]]:
    """파일명 날짜와 심의필 날짜를 대조한다.

    마이데이터 문서에서 잘못된 PDF가 저장된 사례가 있었으므로 상시 검증한다.
    심의필 날짜가 파일명 날짜보다 뒤이면 명백한 이상이다.
    """
    issues: list[tuple[str, str, str]] = []
    for doc in documents:
        if doc.doc_type == DocType.FAQ or not doc.is_parsable:
            continue
        if not doc.effective_date or not doc.compliance_date:
            continue

        gap_days = (doc.effective_date - doc.compliance_date).days
        # 심의를 거친 뒤 시행하므로 심의필 <= 시행일이 정상이다.
        # 다만 게시일 기준 파일명일 수 있어 여유를 둔다.
        if gap_days < -30:
            issues.append((
                doc.orig_filename,
                f"파일명 {doc.effective_date}",
                f"심의필 {doc.compliance_date} (시행일보다 {-gap_days}일 뒤)",
            ))
        elif gap_days > 365 * 2:
            issues.append((
                doc.orig_filename,
                f"파일명 {doc.effective_date}",
                f"심의필 {doc.compliance_date} ({gap_days // 365}년 차이)",
            ))
    return issues


def verify_duplicates(documents: list[Document]) -> dict[str, list[str]]:
    """동일한 본문 해시를 가진 문서를 찾아 중복분을 인덱싱에서 제외한다.

    제목 중복은 정상일 수 있으나(장외파생상품설명서 14건), 본문 해시가
    같으면 같은 파일이 두 번 저장된 것이다. 실제로 크롤링 시 링크가 잘못
    매핑된 사례가 확인됐다(NDF 자리에 FX SWAP 문서 등).

    같은 내용이 두 번 검색되면 Recall/MRR 지표가 왜곡되므로, 파일명 순으로
    첫 번째만 남기고 나머지는 제외한다.
    """
    by_hash: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        if doc.is_parsable and doc.content_hash and doc.doc_type != DocType.FAQ:
            by_hash[doc.content_hash].append(doc)

    result: dict[str, list[str]] = {}
    for h, group in by_hash.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda d: d.orig_filename)
        for dup in group[1:]:
            dup.mark_unparsable(f"본문 중복 (원본: {group[0].orig_filename})")
        result[h] = [d.orig_filename for d in group]
    return result


def mark_versions(documents: list[Document]) -> int:
    """같은 문서의 여러 버전을 찾아 is_latest / superseded_by 를 채운다.

    보수적으로 판정한다. 잘못 묶으면 멀쩡한 문서가 검색에서 빠지므로,
    확실한 경우가 아니면 판정하지 않는다.

    실제로 관찰된 오탐 사례:
      - '가계대출 상품설명서' 2건은 시행일이 같은 별개 상품이었다
        (일반 / 1Q 오토론)
      - '대출거래약정서' 2건은 보금자리론용(2018)과 주택도시기금용(2026)으로
        전혀 다른 상품이었다

    두 사례 모두 부제 추출에 실패해 제목만으로 묶인 것이 원인이다.
    따라서 부제가 없으면 버전 판정을 보류한다.
    """
    groups: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        if not doc.is_parsable or doc.doc_type == DocType.FAQ:
            continue
        # 부제가 없으면 동일 문서인지 확신할 수 없다.
        # PDF 104건 중 54건이 부제 없음이므로 이 조건이 중요하다.
        if not doc.subtitle:
            continue
        if not doc.effective_date:
            continue
        key = re.sub(r"\s+", "", f"{doc.title}|{doc.subtitle}")
        groups[key].append(doc)

    superseded = 0
    for docs in groups.values():
        if len(docs) < 2:
            continue

        # 시행일이 모두 같으면 개정 관계가 아니다. 별개 문서이거나 중복이다.
        dates = {d.effective_date for d in docs}
        if len(dates) < 2:
            continue

        docs.sort(key=lambda d: d.effective_date)  # type: ignore[arg-type,return-value]
        latest = docs[-1]
        for old in docs[:-1]:
            # 본문이 완전히 같으면 개정이 아니라 중복 저장이다.
            if old.content_hash == latest.content_hash:
                continue
            old.is_latest = False
            old.superseded_by = latest.doc_id
            superseded += 1
    return superseded


# ------------------------------------------------------------------
# 저장
# ------------------------------------------------------------------


def save_jsonl(documents: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def save_registry(documents: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Excel에서 한글이 깨지지 않도록 BOM 포함 UTF-8로 저장한다
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for doc in documents:
            writer.writerow(doc.to_dict())


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="유형별 처리 건수 제한 (빠른 확인용)")
    ap.add_argument("--skip-faq", action="store_true", help="FAQ 파싱 생략")
    args = ap.parse_args()

    print("=" * 68)
    print("문서 파싱 및 대장 구축")
    print("=" * 68)

    factory = ParserFactory(bank_code="hana")

    documents = collect_pdfs(factory, limit=args.limit)
    if not args.skip_faq:
        documents.extend(collect_faq(factory))

    if not documents:
        sys.exit("파싱된 문서가 없습니다.")

    # --- 검증 및 정리 ---
    # 중복 제외를 먼저 해야 버전 판정이 중복 문서에 오염되지 않는다.
    dups = verify_duplicates(documents)
    n_superseded = mark_versions(documents)
    date_issues = verify_dates(documents)

    # --- 저장 ---
    save_jsonl(documents, OUT_JSONL)
    save_registry(documents, OUT_REGISTRY)

    # --- 요약 ---
    parsable = [d for d in documents if d.is_parsable]
    excluded = [d for d in documents if not d.is_parsable]

    print("\n" + "=" * 68)
    print("요약")
    print("=" * 68)
    print(f"  전체 문서        : {len(documents)}건")
    print(f"  인덱싱 대상      : {len(parsable)}건")
    print(f"  제외             : {len(excluded)}건")
    if n_superseded:
        print(f"  구버전으로 표시   : {n_superseded}건")

    print("\n  [유형별]")
    by_type: dict[str, int] = defaultdict(int)
    for d in parsable:
        by_type[d.doc_type.value] += 1
    for k, v in sorted(by_type.items()):
        print(f"    {k:<10} : {v:>4}건")

    print("\n  [구조별] (PDF만)")
    by_struct: dict[str, int] = defaultdict(int)
    for d in parsable:
        if d.doc_type != DocType.FAQ:
            by_struct[d.structure.value] += 1
    for k, v in sorted(by_struct.items(), key=lambda x: -x[1]):
        print(f"    {k:<10} : {v:>4}건")

    print("\n  [제목 신뢰도] (PDF만)")
    by_conf: dict[str, int] = defaultdict(int)
    for d in parsable:
        if d.doc_type != DocType.FAQ:
            by_conf[d.title_confidence.value] += 1
    for k, v in sorted(by_conf.items(), key=lambda x: -x[1]):
        print(f"    {k:<10} : {v:>4}건")

    if excluded:
        print("\n  [제외 문서]")
        for d in excluded[:10]:
            print(f"    - {d.orig_filename[:48]:<48} {d.exclusion_reason}")
        if len(excluded) > 10:
            print(f"    ... 외 {len(excluded) - 10}건")

    # --- 검증 결과 ---
    if date_issues:
        print("\n  [날짜 불일치 — 확인 필요]")
        for name, a, b in date_issues[:10]:
            print(f"    - {name[:44]}")
            print(f"        {a} / {b}")
    else:
        print("\n  [날짜 검증] 이상 없음")

    if dups:
        print("\n  [본문 중복 — 첫 번째만 유지하고 나머지 제외]")
        for names in list(dups.values())[:5]:
            print(f"    유지: {names[0][:60]}")
            for n in names[1:]:
                print(f"    제외: {n[:60]}")
            print()
        print("    ! 파일명과 내용이 불일치하므로 재수집이 필요합니다.")
    else:
        print("  [중복 검증] 이상 없음")

    print(f"\n저장 → {OUT_JSONL.relative_to(PROJECT_ROOT)}")
    print(f"저장 → {OUT_REGISTRY.relative_to(PROJECT_ROOT)}")
    print("=" * 68)


if __name__ == "__main__":
    main()
