"""PDF 파서.

전수조사(108건)에서 확인된 사실을 그대로 반영한다.

1. 제목 위치가 일정하지 않다 (1번째 줄 ~ 7번째 줄)
   -> 위치가 아니라 폰트 크기로 찾는다.

2. 제목만으로는 문서를 구별할 수 없다
   -> 장외파생상품설명서 14건은 title 이 모두 같고 부제로만 구별된다.
      폰트 크기 상위 2줄을 각각 title, subtitle 로 잡는다.

3. 문자 단위로 이어붙이면 공백이 사라진다
   -> 05_triage 의 불일치 35건 중 대부분이 이 문제였다.
      extract_words() 를 써서 공백을 보존한다.

4. 스캔본이 4건 있다
   -> 페이지당 문자 수가 임계값 미만이면 파싱 불가로 표시하고 레코드는 남긴다.

5. 심의필 번호의 유형이 여러 가지다
   -> 2022년판은 '제2022-상품-170호', 2026년판은 '제2026-설명서-078호'.
      약관/상품/설명서를 모두 받는다.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import pdfplumber

from ..schema import (
    Document,
    DocType,
    Structure,
    TitleConfidence,
    normalize_text,
    parse_date_from_filename,
)
from .base import BaseParser, ParseError

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

# 페이지당 이 글자 수 미만이면 텍스트 레이어가 없는 것으로 본다.
# 정상 문서는 페이지당 1,200~2,200자가 나왔다.
SCAN_THRESHOLD_CHARS_PER_PAGE = 100

# 제목 후보에서 배제할 줄. 준법감시인 심의필은 상단에 배치되어
# 폰트 크기만으로는 제목으로 오인될 수 있다.
TITLE_EXCLUDE = re.compile(
    r"준법감시인|심의필|심사필|보존년한|페이지|Page|\d+\s*/\s*\d+|^\(cid:"
)

# 부제로 인정하지 않을 패턴.
# 상품설명서 표지에는 제목 바로 아래 안내 문구가 큰 글씨로 인쇄되는
# 경우가 많다. 이를 부제로 잡으면 출처 표시가 문장으로 오염된다.
# 예: "이 설명서는 이용자의 서비스에 대한 이해를 돕고 약관의 중요 내용을..."
SUBTITLE_EXCLUDE = re.compile(
    r"이\s*(설명서|약정서|계약서)는"
    r"|참고자료|알려드리기\s*위한|이해를\s*돕|내부통제기준"
    r"|(합니다|입니다|한다|됩니다|바랍니다)\s*[.]?\s*$"
)

# 부제 최대 길이. 파생상품 부제가 실제로 50자를 넘는 경우가 있어
# 넉넉히 잡는다. 예: "Target Redemption Forward – 고객이 조기종료 조건부로 외화 매도"
SUBTITLE_MAX_LEN = 60

# 문서 종류를 나타내는 키워드. 폰트 기반 결과를 교차 검증하는 데 쓴다.
TITLE_KEYWORDS = [
    "상품설명서",
    "설명서",
    "기본약관",
    "이용약관",
    "약관",
    "추가약정서",
    "약정서",
    "계약권유문서",
    "동의서",
]

# 심의필: 제2026-설명서-078호(2026.06.15)
RE_COMPLIANCE = re.compile(
    r"제?\s*(\d{4})\s*-\s*(약관|상품|설명서)\s*-\s*(\d+)\s*호"
)
RE_COMPLIANCE_DATE = re.compile(r"\((\d{4})[.\-](\d{1,2})[.\-](\d{1,2})\)")

RE_ARTICLE = re.compile(r"제\s*\d+\s*조")
RE_CIRCLED = re.compile(r"[\u2460-\u246f]")
RE_NUMBERED = re.compile(r"^\s*\d+\.\s", re.MULTILINE)

# 파일명: hana_{유형}_{카테고리}_{상품명}_{YYYYMMDD}.pdf
# 카테고리에 언더스코어가 포함된 경우가 있어(digital_banking, retirement_pension)
# 알려진 카테고리를 명시적으로 매칭한다.
KNOWN_CATEGORIES = [
    "digital_banking",
    "retirement_pension",
    "derivatives",
    "deposit",
    "trust",
    "loan",
    "fx",
    "others",
]


def collapse_letter_spacing(text: str) -> str:
    """자간을 넓혀 인쇄한 제목의 공백을 제거한다.

    PDF에서 '대 출 거 래 추 가 약 정 서' 처럼 글자마다 공백을 넣어
    조판한 제목이 그대로 추출된다. 이 상태로는 검색어와 매칭되지 않고
    출처 표시도 어색하다.

    한 글자 단위로 공백이 반복되는 구간만 압축하고, 일반적인 단어
    사이 공백은 유지한다. 예를 들어 '은행여신거래 기본약관 (가계용)'은
    그대로 둔다.
    """
    # 한글 한 글자 + 공백 패턴이 3회 이상 연속되는 구간을 찾는다
    pattern = re.compile(r"(?:[가-힣]\s){2,}[가-힣]")

    def _compress(m: re.Match) -> str:
        return m.group(0).replace(" ", "")

    return pattern.sub(_compress, text)


class PDFParser(BaseParser):
    """상품설명서·약관 PDF를 Document로 변환한다."""

    def __init__(self, bank_code: str = "hana", scan_threshold: int = SCAN_THRESHOLD_CHARS_PER_PAGE):
        self.bank_code = bank_code
        self.scan_threshold = scan_threshold

    # --------------------------------------------------------------
    # 인터페이스
    # --------------------------------------------------------------

    def can_parse(self, source: Path) -> bool:
        return source.suffix.lower() == ".pdf" and not source.name.startswith("INVALID_")

    def parse(self, source: Path, doc_id: str = "", **kwargs) -> list[Document]:
        if not source.exists():
            raise ParseError(f"파일 없음: {source}")

        doc = Document(
            doc_id=doc_id or source.stem,
            bank_code=self.bank_code,
            source_path=str(source).replace("\\", "/"),
            orig_filename=source.name,
            parsed_at=datetime.now(),
        )

        doc.doc_type = self._detect_doc_type(source.name)
        doc.category = self._detect_category(source.name)
        doc.effective_date = parse_date_from_filename(source.name)

        with pdfplumber.open(source) as pdf:
            pages = pdf.pages
            doc.page_count = len(pages)
            if doc.page_count == 0:
                doc.mark_unparsable("페이지 없음")
                return [doc]

            page_texts: list[str] = []
            for page in pages:
                page_texts.append(page.extract_text() or "")
                try:
                    doc.n_tables += len(page.extract_tables())
                except Exception:
                    pass  # 표 추출 실패는 치명적이지 않다

            doc.text = normalize_text("\n".join(page_texts))
            doc.char_count = len(doc.text)

            # 스캔본 판정. 이후 분석이 무의미하므로 여기서 종료한다.
            if doc.char_count // max(doc.page_count, 1) < self.scan_threshold:
                doc.mark_unparsable("텍스트 레이어 없음(스캔본)")
                return [doc]

            title, subtitle, confidence = self._extract_titles(pages[0], page_texts[0])
            doc.title = title
            doc.subtitle = subtitle
            doc.title_confidence = confidence

        doc.structure = self._detect_structure(doc.text)
        self._extract_compliance(doc)
        doc.compute_hash()

        return [doc]

    # --------------------------------------------------------------
    # 제목 추출
    # --------------------------------------------------------------

    def _extract_titles(self, first_page, first_page_text: str) -> tuple[str, str, TitleConfidence]:
        """1페이지에서 제목과 부제를 뽑고 신뢰도를 판정한다.

        폰트 크기 상위 2줄을 title, subtitle 로 삼되, 키워드 기반 결과와
        대조해 신뢰도를 매긴다.
        """
        lines = self._lines_by_font(first_page)
        by_keyword = self._title_by_keyword(first_page_text)

        if not lines:
            return (by_keyword, "", TitleConfidence.MEDIUM if by_keyword else TitleConfidence.MANUAL)

        title = lines[0][1]
        subtitle = lines[1][1] if len(lines) > 1 else ""

        # 자간을 넓혀 조판한 제목의 공백을 압축한다
        title = collapse_letter_spacing(title)
        subtitle = collapse_letter_spacing(subtitle)

        # 부제 검증
        if subtitle:
            # 폰트 크기가 비슷한 줄만 부제로 인정한다.
            # 제목이 20pt인데 다음 줄이 9pt면 그건 본문이지 부제가 아니다.
            ratio = lines[1][0] / lines[0][0] if lines[0][0] else 0
            if ratio < 0.7:
                subtitle = ""
            # 안내 문구를 부제로 잡은 경우 버린다
            elif SUBTITLE_EXCLUDE.search(subtitle) or len(subtitle) > SUBTITLE_MAX_LEN:
                subtitle = ""
            # 제목과 같은 내용이면 중복이므로 버린다
            elif re.sub(r"\s+", "", subtitle) == re.sub(r"\s+", "", title):
                subtitle = ""

        # 신뢰도 판정: 공백을 무시하고 비교한다.
        # 폰트 방식과 키워드 방식이 같은 답을 내면 신뢰할 수 있다.
        confidence = TitleConfidence.MEDIUM
        if by_keyword:
            squash = lambda s: re.sub(r"\s+", "", s)
            if squash(title) == squash(by_keyword):
                confidence = TitleConfidence.HIGH
            elif squash(by_keyword) in squash(title) or squash(title) in squash(by_keyword):
                confidence = TitleConfidence.HIGH
                # 키워드 방식이 더 구체적이면 그쪽을 채택한다.
                # 예: '대출거래약정서' vs '대출거래약정서 (주택담보노후연금대출용)'
                if len(squash(by_keyword)) > len(squash(title)):
                    title = by_keyword

        # 제목이 비었거나 잡음이면 키워드 결과로 대체한다.
        if not title or TITLE_EXCLUDE.search(title):
            title = by_keyword
            confidence = TitleConfidence.MEDIUM if by_keyword else TitleConfidence.MANUAL

        return (title.strip(), subtitle.strip(), confidence)

    def _lines_by_font(self, page, top_n: int = 2) -> list[tuple[float, str]]:
        """페이지의 줄을 폰트 크기 순으로 정렬해 상위 n개를 반환한다.

        extract_words() 를 쓰는 이유: 문자 단위로 이어붙이면 PDF에 공백
        문자가 없는 경우 단어가 모두 붙어버린다. 05_triage 에서 실제로
        '『청년주택드림청약통장』요약상품설명서' 처럼 추출됐다.
        """
        try:
            words = page.extract_words(extra_attrs=["size"])
        except Exception:
            return []

        if not words:
            return []

        # y좌표로 같은 줄끼리 묶는다
        rows: dict[int, list] = {}
        for w in words:
            rows.setdefault(round(w["top"]), []).append(w)

        candidates: list[tuple[float, float, str]] = []
        for top, group in rows.items():
            group.sort(key=lambda w: w["x0"])
            text = normalize_text(" ".join(w["text"] for w in group))

            if len(text) < 3 or len(text) > 60:
                continue
            if TITLE_EXCLUDE.search(text):
                continue
            if not re.search(r"[가-힣A-Za-z]", text):
                continue

            avg_size = sum(float(w.get("size", 0)) for w in group) / len(group)
            candidates.append((avg_size, -top, text))

        if not candidates:
            return []

        # 폰트가 크고, 같으면 위쪽에 있는 줄을 우선한다
        candidates.sort(reverse=True)
        return [(size, text) for size, _, text in candidates[:top_n]]

    def _title_by_keyword(self, first_page_text: str) -> str:
        """문서 종류 키워드를 포함한 줄을 제목 후보로 반환한다.

        폰트 기반 결과를 검증하는 용도. 제목이 7번째 줄에 있던 사례가
        있어 넉넉히 훑는다.
        """
        for raw in first_page_text.splitlines()[:25]:
            line = normalize_text(raw)
            if not line or len(line) > 60 or TITLE_EXCLUDE.search(line):
                continue
            if any(kw in line for kw in TITLE_KEYWORDS):
                return line
        return ""

    # --------------------------------------------------------------
    # 분류
    # --------------------------------------------------------------

    def _detect_doc_type(self, filename: str) -> DocType:
        return DocType.TERMS if "_terms_" in filename else DocType.DESCRIPTION

    def _detect_category(self, filename: str) -> str:
        """파일명에서 카테고리를 뽑는다.

        digital_banking 처럼 언더스코어를 포함한 카테고리가 있어
        단순 split 으로는 안 된다. 알려진 목록을 긴 것부터 매칭한다.
        """
        for cat in KNOWN_CATEGORIES:
            if f"_{cat}_" in filename:
                return cat
        return "unknown"

    def _detect_structure(self, text: str) -> Structure:
        """본문 구조를 판정한다. 청킹 전략 라우팅의 근거가 된다.

        임계값은 05_triage 의 108건 분포를 근거로 정했다.
        """
        n_articles = len(RE_ARTICLE.findall(text))
        n_circled = len(RE_CIRCLED.findall(text))
        n_numbered = len(RE_NUMBERED.findall(text))

        if n_articles >= 5:
            return Structure.ARTICLE
        if n_circled >= 10:
            return Structure.CLAUSE
        if n_numbered >= 5:
            return Structure.NUMBERED
        return Structure.FLAT

    # --------------------------------------------------------------
    # 버전 정보
    # --------------------------------------------------------------

    def _extract_compliance(self, doc: Document) -> None:
        """준법감시인 심의필 번호와 일자를 추출한다.

        이 번호는 하나은행이 문서 버전을 공식 식별하는 값이므로,
        파일명 날짜보다 신뢰도가 높은 개정 판별 근거가 된다.
        """
        m = RE_COMPLIANCE.search(doc.text)
        if not m:
            return

        year, kind, number = m.groups()
        doc.compliance_no = f"{year}-{kind}-{number}"

        tail = doc.text[m.end(): m.end() + 40]
        d = RE_COMPLIANCE_DATE.search(tail)
        if d:
            try:
                doc.compliance_date = datetime(
                    int(d.group(1)), int(d.group(2)), int(d.group(3))
                ).date()
            except ValueError:
                logger.warning("심의필 일자 파싱 실패: %s", doc.orig_filename)
