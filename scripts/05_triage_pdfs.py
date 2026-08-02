"""
05_triage_pdfs.py

목적
----
원본 PDF 108건(설명서 80 + 약관 28)을 전수조사하여, 파서 설계에 필요한
사실관계를 CSV로 확보한다.

관찰(04) 단계에서 드러난 문제
---------------------------
1. 텍스트 레이어가 없는 스캔본이 섞여 있다 (추출 문자 2개인 문서 발견)
   → 몇 건인지 모르면 OCR 도입 여부를 결정할 수 없다
2. 한글 제목의 위치가 문서마다 다르다 (1번째 줄 ~ 7번째 줄)
   → 위치 기반 규칙 대신 폰트 크기 기반 추출이 필요하다
3. 조항(제N조) 구조가 없는 문서가 있다 (추가약정서 등 1페이지 서식)
   → 문서 유형이 아니라 실제 구조에 따라 청킹 전략을 골라야 한다

출력
----
data/registry/pdf_triage.csv
    파일별 진단 결과. 이후 대장(document_registry.csv) 보강의 근거가 된다.

사용법
-----
    python scripts/05_triage_pdfs.py
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict, fields
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
OUT_CSV = PROJECT_ROOT / "data" / "registry" / "pdf_triage.csv"

# 페이지당 이 글자 수 미만이면 스캔본으로 의심한다.
# 정상 문서는 페이지당 대략 500~2,000자가 나온다.
SCAN_SUSPECT_CHARS_PER_PAGE = 100

# 제목에 나타나는 문서 종류 키워드. 긴 것부터 검사해야 오탐이 없다.
TITLE_KEYWORDS = [
    "상품설명서",
    "설명서",
    "기본약관",
    "이용약관",
    "약관",
    "추가약정서",
    "약정서",
    "동의서",
    "신청서",
    "확인서",
]

# 제목 후보에서 제외할 줄. 준법감시인 심의필은 상단에 크게 찍히는 경우가 있어
# 폰트 크기 기준만으로는 제목으로 오인될 수 있다.
TITLE_EXCLUDE = re.compile(r"준법감시인|심의필|보존년한|페이지|Page")

RE_ARTICLE = re.compile(r"제\s*\d+\s*조")
RE_CIRCLED = re.compile(r"[\u2460-\u246f]")  # ①~⑮
RE_NUMBERED = re.compile(r"^\s*\d+\.\s", re.MULTILINE)
RE_COMPLIANCE = re.compile(r"(\d{4})\s*-\s*약관\s*-\s*(\d+)\s*호")
RE_COMPLIANCE_DATE = re.compile(r"\((\d{4})[.\-](\d{1,2})[.\-](\d{1,2})\)")


# ------------------------------------------------------------------
# 결과 스키마
# ------------------------------------------------------------------


@dataclass
class TriageResult:
    """PDF 1건의 진단 결과."""

    doc_type: str  # desc / terms
    filename: str
    n_pages: int
    n_chars: int
    chars_per_page: int
    is_scanned: bool  # 텍스트 레이어 없음 → OCR 필요
    title_by_font: str  # 폰트 크기 기반 제목 후보
    title_by_keyword: str  # 키워드 기반 제목 후보
    n_articles: int  # 제N조 출현 수
    n_circled: int  # ①②③ 출현 수
    n_numbered: int  # "1. " 출현 수
    n_tables: int  # 추출된 표 개수 (전 페이지)
    structure: str  # article / clause / numbered / flat
    compliance_no: str  # 준법감시인 심의필 번호
    compliance_date: str  # 심의필 일자
    error: str


# ------------------------------------------------------------------
# 추출 로직
# ------------------------------------------------------------------


def normalize(text: str) -> str:
    """PDF 추출 텍스트의 흔한 잡음을 정리한다.

    - 유니코드 정규화(NFC): 한글 자모 분리 방지
    - 전각 공백 → 반각
    - 연속 공백 축약
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_title_by_font(page) -> str:
    """1페이지에서 가장 큰 폰트로 쓰인 줄을 제목 후보로 반환한다.

    제목의 '위치'는 문서마다 다르지만 '크기'는 대체로 본문보다 크다.
    따라서 위치 기반보다 폰트 기반이 견고하다.

    구현: 문자들을 y좌표(top)로 묶어 줄을 만들고, 줄별 평균 폰트 크기를
    구한 뒤 가장 큰 줄을 고른다.
    """
    chars = page.chars
    if not chars:
        return ""

    # y좌표를 정수로 반올림해 같은 줄끼리 묶는다
    lines: dict[int, list] = {}
    for ch in chars:
        key = round(ch["top"])
        lines.setdefault(key, []).append(ch)

    scored: list[tuple[float, int, str]] = []
    for top, group in lines.items():
        group_sorted = sorted(group, key=lambda c: c["x0"])
        text = normalize("".join(c["text"] for c in group_sorted))
        if len(text) < 3 or TITLE_EXCLUDE.search(text):
            continue
        if not re.search(r"[가-힣A-Za-z]", text):
            continue
        avg_size = sum(c["size"] for c in group) / len(group)
        scored.append((avg_size, -top, text))

    if not scored:
        return ""

    # 폰트가 크고, 같으면 페이지 위쪽에 있는 줄을 우선한다
    scored.sort(reverse=True)
    return scored[0][2][:80]


def extract_title_by_keyword(first_page_text: str) -> str:
    """앞부분 줄들 중 문서종류 키워드를 포함한 줄을 제목 후보로 반환한다.

    폰트 기반이 실패할 때를 대비한 교차 검증용이다.
    두 방식의 결과가 일치하면 신뢰도가 높다고 볼 수 있다.
    """
    lines = [normalize(ln) for ln in first_page_text.splitlines()]
    lines = [ln for ln in lines if ln and not TITLE_EXCLUDE.search(ln)]

    for line in lines[:25]:  # 제목이 7번째 줄에 있던 사례가 있어 넉넉히 본다
        for kw in TITLE_KEYWORDS:
            if kw in line and len(line) < 60:
                return line[:80]
    return ""


def classify_structure(n_articles: int, n_circled: int, n_numbered: int, n_pages: int) -> str:
    """문서의 내부 구조를 분류한다. 청킹 전략 선택의 근거가 된다.

    article  : 제N조 단위로 나눌 수 있음 (약관 표준형)
    clause   : 조는 없으나 ①②③ 항 구조가 뚜렷함
    numbered : "1. 2. 3." 번호 목록 중심
    flat     : 구조 표지가 없음 (서식, 표 중심 문서) → 페이지/길이 기반 분할 필요
    """
    if n_articles >= 5:
        return "article"
    if n_circled >= 10:
        return "clause"
    if n_numbered >= 5:
        return "numbered"
    return "flat"


def triage_one(pdf_path: Path, doc_type: str) -> TriageResult:
    """PDF 1건을 진단한다. 실패해도 예외를 던지지 않고 error 필드에 기록한다."""
    result = TriageResult(
        doc_type=doc_type,
        filename=pdf_path.name,
        n_pages=0,
        n_chars=0,
        chars_per_page=0,
        is_scanned=False,
        title_by_font="",
        title_by_keyword="",
        n_articles=0,
        n_circled=0,
        n_numbered=0,
        n_tables=0,
        structure="",
        compliance_no="",
        compliance_date="",
        error="",
    )

    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages
            result.n_pages = len(pages)
            if result.n_pages == 0:
                result.error = "페이지 없음"
                return result

            texts = []
            n_tables = 0
            for page in pages:
                texts.append(page.extract_text() or "")
                try:
                    n_tables += len(page.extract_tables())
                except Exception:
                    pass  # 표 추출 실패는 치명적이지 않다

            full_text = normalize("\n".join(texts))
            result.n_chars = len(full_text)
            result.chars_per_page = result.n_chars // max(result.n_pages, 1)
            result.n_tables = n_tables
            result.is_scanned = result.chars_per_page < SCAN_SUSPECT_CHARS_PER_PAGE

            # 스캔본이면 아래 분석은 무의미하므로 여기서 종료
            if result.is_scanned:
                result.structure = "scanned"
                return result

            result.title_by_font = extract_title_by_font(pages[0])
            result.title_by_keyword = extract_title_by_keyword(texts[0])

            result.n_articles = len(RE_ARTICLE.findall(full_text))
            result.n_circled = len(RE_CIRCLED.findall(full_text))
            result.n_numbered = len(RE_NUMBERED.findall(full_text))
            result.structure = classify_structure(
                result.n_articles, result.n_circled, result.n_numbered, result.n_pages
            )

            # 준법감시인 심의필 정보는 문서 버전을 특정하는 공식 근거가 되므로
            # 노이즈로 버리지 않고 메타데이터로 확보한다
            m = RE_COMPLIANCE.search(full_text)
            if m:
                result.compliance_no = f"{m.group(1)}-약관-{m.group(2)}"
                tail = full_text[m.end(): m.end() + 30]
                d = RE_COMPLIANCE_DATE.search(tail)
                if d:
                    result.compliance_date = f"{d.group(1)}-{int(d.group(2)):02d}-{int(d.group(3)):02d}"

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:120]

    return result


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------


def main() -> None:
    targets = {"desc": RAW_DIR / "desc", "terms": RAW_DIR / "terms"}

    results: list[TriageResult] = []

    for doc_type, directory in targets.items():
        if not directory.exists():
            print(f"!! {directory} 없음 — 건너뜁니다.")
            continue

        pdfs = sorted(directory.glob("*.pdf"))
        print(f"\n[{doc_type}] {len(pdfs)}건 처리 중...")

        for i, pdf_path in enumerate(pdfs, 1):
            res = triage_one(pdf_path, doc_type)
            results.append(res)
            flag = ""
            if res.error:
                flag = "  <-- 오류"
            elif res.is_scanned:
                flag = "  <-- 스캔본 의심"
            print(f"  [{i:>3}/{len(pdfs)}] {pdf_path.name[:52]:<52}{flag}")

    if not results:
        sys.exit("처리된 파일이 없습니다.")

    # --- CSV 저장 ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Excel에서 한글이 깨지지 않도록 BOM 포함 UTF-8로 저장한다
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(TriageResult)])
        writer.writeheader()
        for res in results:
            writer.writerow(asdict(res))

    # --- 요약 ---
    total = len(results)
    scanned = [r for r in results if r.is_scanned]
    errored = [r for r in results if r.error]
    no_title = [r for r in results if not r.is_scanned and not r.title_by_font and not r.title_by_keyword]
    title_agree = [
        r for r in results
        if r.title_by_font and r.title_by_keyword and r.title_by_font == r.title_by_keyword
    ]

    print("\n" + "=" * 64)
    print("요약")
    print("=" * 64)
    print(f"  전체                : {total}건")
    print(f"  스캔본 의심 (OCR 필요): {len(scanned)}건")
    print(f"  처리 오류            : {len(errored)}건")
    print(f"  제목 추출 실패        : {len(no_title)}건")
    print(f"  두 방식 제목 일치     : {len(title_agree)}건")

    print("\n  [구조 분류]")
    counts: dict[str, int] = {}
    for r in results:
        counts[r.structure] = counts.get(r.structure, 0) + 1
    for structure, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {structure:<10} : {cnt:>3}건")

    if scanned:
        print("\n  [스캔본 목록]")
        for r in scanned[:20]:
            print(f"    - {r.filename[:60]}")
        if len(scanned) > 20:
            print(f"    ... 외 {len(scanned) - 20}건")

    if errored:
        print("\n  [오류 목록]")
        for r in errored:
            print(f"    - {r.filename[:45]} : {r.error}")

    print(f"\n결과 저장 → {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
