"""
04_inspect_pdfs.py

목적
----
Document 스키마와 파서를 설계하기 **전에** 원본 PDF의 실제 구조를 관찰한다.
스키마를 먼저 정하고 데이터를 끼워맞추면 반드시 어긋나므로, 관찰이 선행되어야 한다.

확인 항목
--------
1. 한글 상품명이 본문 어디에, 어떤 형태로 인쇄되어 있는가
   → 대장(document_registry.csv)의 로마자 product 필드를 복원할 근거를 찾는다
2. 약관의 조항 패턴(제N조)이 일관적인가
   → 조항 단위 청킹이 가능한지 판단한다
3. 상품설명서의 표가 어떻게 추출되는가
   → 표 보존 전략(pdfplumber vs camelot)을 결정한다
4. 머리말/꼬리말 등 반복 노이즈가 있는가
   → 전처리에서 제거할 대상을 파악한다

사용법
-----
    python scripts/04_inspect_pdfs.py

출력
----
- 터미널: 요약 리포트
- data/samples/inspect_*.txt : 샘플 문서의 전체 추출 텍스트 (육안 검토용)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    sys.exit("pdfplumber가 없습니다.  pip install pdfplumber 를 먼저 실행하세요.")

# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hana"
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"

# 유형별로 관찰할 문서 수. 너무 많으면 육안 검토가 불가능해진다.
N_SAMPLES = 3

# 조항 패턴 후보. 약관마다 표기가 다를 수 있어 여러 개를 시험한다.
ARTICLE_PATTERNS = {
    "제N조": re.compile(r"제\s*\d+\s*조"),
    "제N조(제목)": re.compile(r"제\s*\d+\s*조\s*\("),
    "N. 항목": re.compile(r"^\s*\d+\.\s", re.MULTILINE),
    "①②③": re.compile(r"[①-⑮]"),
}

# 상품명이 들어있을 가능성이 높은 표현
PRODUCT_NAME_HINTS = [
    "상품설명서",
    "상품명",
    "약관",
    "여신거래",
    "예금거래",
]


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------


def pick_samples(directory: Path, n: int) -> list[Path]:
    """디렉토리에서 균등하게 n개를 고른다.

    앞에서 n개만 자르면 파일명 정렬 순서상 비슷한 문서만 뽑히므로,
    전체에 걸쳐 균등 간격으로 추출한다.
    """
    files = sorted(directory.glob("*.pdf"))
    if not files:
        return []
    if len(files) <= n:
        return files
    step = len(files) // n
    return [files[i * step] for i in range(n)]


def extract_pages(pdf_path: Path, max_pages: int | None = None) -> list[str]:
    """페이지별 텍스트를 리스트로 반환한다."""
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        targets = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for page in targets:
            pages.append(page.extract_text() or "")
    return pages


def count_tables(pdf_path: Path, max_pages: int = 5) -> int:
    """앞쪽 페이지에서 추출되는 표의 개수를 센다."""
    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            total += len(page.extract_tables())
    return total


def find_korean_lines(text: str, limit: int = 12) -> list[str]:
    """한글이 포함된 줄만 추려서 반환한다.

    상품명이 어느 줄에 있는지 육안으로 찾기 위한 용도.
    """
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"[가-힣]", stripped):
            out.append(stripped)
        if len(out) >= limit:
            break
    return out


def detect_repeated_lines(pages: list[str], min_ratio: float = 0.6) -> list[str]:
    """여러 페이지에 반복 등장하는 줄을 찾는다 (머리말/꼬리말 후보)."""
    if len(pages) < 3:
        return []
    counter: dict[str, int] = {}
    for page in pages:
        seen = set()
        for line in page.splitlines():
            s = line.strip()
            if 2 < len(s) < 60 and s not in seen:
                counter[s] = counter.get(s, 0) + 1
                seen.add(s)
    threshold = len(pages) * min_ratio
    return [line for line, cnt in counter.items() if cnt >= threshold]


# ------------------------------------------------------------------
# 관찰 로직
# ------------------------------------------------------------------


def inspect_one(pdf_path: Path, doc_type: str) -> None:
    """문서 1건을 관찰하고 리포트를 출력한다."""
    print()
    print("-" * 70)
    print(f"[{doc_type}] {pdf_path.name}")
    print("-" * 70)

    try:
        pages = extract_pages(pdf_path)
    except Exception as exc:  # 손상 PDF 대응
        print(f"  !! 추출 실패: {exc}")
        return

    if not pages:
        print("  !! 페이지가 없습니다.")
        return

    full_text = "\n".join(pages)
    n_chars = len(full_text)

    print(f"  페이지 수      : {len(pages)}")
    print(f"  추출 문자 수   : {n_chars:,}")

    # 텍스트가 거의 없으면 스캔 이미지 PDF일 가능성이 높다 → OCR 필요
    if n_chars < 200:
        print("  !! 텍스트가 거의 없습니다. 스캔본일 수 있으므로 OCR 검토 필요.")

    # --- 1) 상품명 후보 (1페이지 상단 한글 줄) ---
    print("\n  [1페이지 상단 한글 줄 — 상품명 후보]")
    for i, line in enumerate(find_korean_lines(pages[0]), 1):
        print(f"    {i:>2}. {line[:60]}")

    # --- 2) 조항 패턴 ---
    print("\n  [구조 패턴 검출]")
    for name, pattern in ARTICLE_PATTERNS.items():
        hits = pattern.findall(full_text)
        if hits:
            sample = hits[0] if isinstance(hits[0], str) else str(hits[0])
            print(f"    {name:<12} : {len(hits):>4}건   예) {sample.strip()[:20]}")
        else:
            print(f"    {name:<12} : 없음")

    # --- 3) 표 ---
    try:
        n_tables = count_tables(pdf_path)
        print(f"\n  [표] 앞 5페이지에서 {n_tables}개 검출")
    except Exception as exc:
        print(f"\n  [표] 검출 실패: {exc}")

    # --- 4) 반복 줄 (머리말/꼬리말) ---
    repeated = detect_repeated_lines(pages)
    if repeated:
        print("\n  [반복 등장 줄 — 제거 대상 후보]")
        for line in repeated[:5]:
            print(f"    - {line[:60]}")

    # --- 5) 전체 텍스트 저장 ---
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / f"inspect_{doc_type}_{pdf_path.stem[:40]}.txt"
    out_path.write_text(full_text, encoding="utf-8")
    print(f"\n  전체 텍스트 저장 → {out_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print("=" * 70)
    print("PDF 구조 관찰")
    print("=" * 70)

    targets = {
        "terms": RAW_DIR / "terms",
        "desc": RAW_DIR / "desc",
    }

    for doc_type, directory in targets.items():
        if not directory.exists():
            print(f"\n!! {directory} 없음 — 건너뜁니다.")
            continue

        samples = pick_samples(directory, N_SAMPLES)
        print(f"\n\n{'=' * 70}")
        print(f"{doc_type.upper()}  (전체 {len(list(directory.glob('*.pdf')))}건 중 {len(samples)}건 관찰)")
        print("=" * 70)

        for pdf_path in samples:
            inspect_one(pdf_path, doc_type)

    print()
    print("=" * 70)
    print("완료. data/samples/ 의 txt 파일을 열어 직접 확인하세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()
