"""공통 데이터 스키마.

FAQ(JSONL) / 상품설명서(PDF) / 약관(PDF) 세 가지 이질적 소스를 하나의
Document 표현으로 수렴시킨다. 이후 청킹·임베딩·검색 단계는 원본 형식을
알 필요 없이 이 스키마만 다룬다.

설계 근거
--------
1. structure 를 doc_type 과 분리한다
   전수조사 결과 설명서에도 조항(제N조) 구조가 15건, 약관에도 구조 표지가
   없는 서식이 4건 있었다. 문서 유형으로 청킹 전략을 고르면 틀린다.

2. display_name 을 별도로 둔다
   장외파생상품설명서 14건은 title 이 모두 동일하다. 부제를 합친 이름이
   있어야 검색 결과와 출처 표시에서 문서를 구별할 수 있다.

3. 파싱 불가 문서도 레코드를 남긴다
   스캔본 4건을 대장에서 지우면 "왜 80건이 아니라 76건인가"에 답할 수 없다.
   is_parsable=False 와 exclusion_reason 으로 추적 가능하게 남긴다.

4. 버전 관리를 처음부터 넣는다
   하나은행 문서는 시행일 단위로 개정된다. 실제로 마이데이터 설명서는
   2022년판과 2026년판 사이에 이용 연령 요건이 바뀌었다(만 19세 미만 제한
   -> 만 14세 기준 3단계). 구버전을 근거로 답하면 오안내가 된다.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from enum import Enum
from typing import Any


# ==================================================================
# 열거형
# ==================================================================


class DocType(str, Enum):
    """문서 유형. 출처 표기와 필터링에 쓴다."""

    DESCRIPTION = "설명서"
    TERMS = "약관"
    FAQ = "FAQ"


class Structure(str, Enum):
    """문서의 내부 구조. 청킹 전략을 고르는 라우팅 키다.

    전수조사(108건) 분포:
        ARTICLE  37건 - 제N조 체계가 뚜렷한 표준 약관형
        FLAT     34건 - 구조 표지가 없는 짧은 서식·안내문
        CLAUSE   18건 - 조는 없으나 항(1)(2)(3) 구조가 뚜렷함
        NUMBERED 15건 - 번호 목록 중심
        SCANNED   4건 - 텍스트 레이어 없음. 인덱싱 제외
    """

    ARTICLE = "article"
    CLAUSE = "clause"
    NUMBERED = "numbered"
    FLAT = "flat"
    SCANNED = "scanned"
    QA_PAIR = "qa_pair"  # FAQ 전용


class TitleConfidence(str, Enum):
    """제목 추출 신뢰도.

    폰트 크기 기반과 키워드 기반, 두 방식의 결과를 대조해 판정한다.
    두 방식이 일치한 경우가 104건 중 91건(88%)이었다.
    """

    HIGH = "high"  # 두 방식 일치
    MEDIUM = "medium"  # 한 방식만 성공하거나 결과가 다름
    MANUAL = "manual"  # 자동 추출 실패. 수동 매핑으로 채움


# ==================================================================
# Document
# ==================================================================


@dataclass
class Document:
    """정규화된 문서 1건.

    PDF든 JSONL이든 파싱을 마치면 이 형태가 된다.
    """

    # --- 식별 ---
    doc_id: str  # 예: hana_desc_deposit_001
    bank_code: str  # 예: hana. 타행 확장 대비
    source_path: str  # 원본 파일 경로 (저장소 루트 기준 상대경로)

    # --- 내용 ---
    title: str = ""  # PDF에서 추출한 한글 제목
    subtitle: str = ""  # 부제. 동일 제목 문서를 구별한다
    text: str = ""  # 정규화된 본문 전체

    # --- 분류 ---
    doc_type: DocType = DocType.DESCRIPTION
    category: str = ""  # deposit / loan / fx / trust / derivatives ...
    structure: Structure = Structure.FLAT

    # --- 버전 관리 ---
    effective_date: date | None = None  # 파일명에서 파싱한 시행일
    compliance_no: str = ""  # 준법감시인 심의필 번호
    compliance_date: date | None = None  # 심의필 일자
    content_hash: str = ""  # 본문 SHA-256. 재수집 시 변경 감지용
    is_latest: bool = True
    superseded_by: str | None = None  # 대체된 경우 후속 문서의 doc_id

    # --- 품질 관리 ---
    is_parsable: bool = True  # False면 인덱싱에서 제외
    title_confidence: TitleConfidence = TitleConfidence.MEDIUM
    exclusion_reason: str = ""

    # --- 통계 (triage 단계에서 이미 계산되므로 추가 비용 없음) ---
    page_count: int = 0
    char_count: int = 0
    n_tables: int = 0

    # --- 추적 ---
    source_url: str = ""  # 원문 링크. 데모에서 "원문 보기"에 쓴다
    orig_filename: str = ""
    parsed_at: datetime | None = None

    # ----------------------------------------------------------
    # 파생 속성
    # ----------------------------------------------------------

    @property
    def display_name(self) -> str:
        """검색 결과와 출처 표시에 쓰는 이름.

        장외파생상품설명서 14건처럼 제목이 같은 문서들은 부제로만 구별된다.
        따라서 사용자에게 보여줄 때는 반드시 이 값을 쓴다.
        """
        if self.subtitle and self.subtitle != self.title:
            return f"{self.title} - {self.subtitle}"
        return self.title or self.orig_filename

    @property
    def citation(self) -> str:
        """답변에 붙일 출처 문자열.

        직원이 원문을 찾아갈 수 있도록 시행일까지 포함한다.
        """
        parts = [self.display_name]
        if self.effective_date:
            # FAQ에는 시행일 개념이 없다. 언제 수집된 정보인지를 밝힌다.
            label = "수집" if self.doc_type == DocType.FAQ else "시행"
            parts.append(f"({self.effective_date.isoformat()} {label})")
        return " ".join(parts)

    # ----------------------------------------------------------
    # 동작
    # ----------------------------------------------------------

    def compute_hash(self) -> str:
        """본문의 SHA-256을 계산해 content_hash에 채우고 반환한다.

        공백 차이만으로 해시가 달라지면 개정 감지가 무의미해지므로
        공백을 제거한 뒤 계산한다.
        """
        normalized = re.sub(r"\s+", "", self.text)
        self.content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return self.content_hash

    def mark_unparsable(self, reason: str) -> None:
        """인덱싱 제외 처리. 레코드 자체는 남긴다."""
        self.is_parsable = False
        self.exclusion_reason = reason
        self.structure = Structure.SCANNED

    def to_dict(self) -> dict[str, Any]:
        """CSV/JSON 저장용 평면 딕셔너리로 변환한다."""
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, (date, datetime)):
                d[key] = value.isoformat()
            elif isinstance(value, Enum):
                d[key] = value.value
        d["display_name"] = self.display_name
        return d


# ==================================================================
# Chunk
# ==================================================================


@dataclass
class Chunk:
    """검색 단위. 임베딩과 BM25 인덱싱의 대상이다."""

    chunk_id: str  # 예: hana_desc_deposit_001_c003
    doc_id: str
    text: str
    order: int  # 문서 내 순번

    # --- 위치 정보 (근거 제시용) ---
    section: str = ""  # 예: 제3조 (이자)
    page_start: int = 0
    page_end: int = 0

    # --- 상위 문서에서 상속 (검색 필터링·출처 표시용) ---
    # 청크만 있으면 답변을 구성할 수 있어야 하므로 필요한 메타데이터를 복사한다.
    doc_display_name: str = ""
    doc_type: str = ""
    category: str = ""
    effective_date: str = ""
    source_url: str = ""

    # --- 통계 ---
    char_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """이 청크를 근거로 제시할 때 쓰는 출처 문자열."""
        parts = [self.doc_display_name]
        if self.section and self.section != "FAQ":
            parts.append(self.section)
        if self.effective_date:
            # FAQ에는 시행일 개념이 없다. 수집 시점을 밝힌다.
            label = "수집" if self.doc_type == "FAQ" else "시행"
            parts.append(f"({self.effective_date} {label})")
        return " ".join(p for p in parts if p)

    @property
    def indexable_text(self) -> str:
        """인덱싱에 넣을 텍스트.

        본문만 넣으면 "발행어음 중도해지" 같은 질의에서 상품명이 청크 안에
        없을 때 매칭에 실패한다. 문서명과 섹션명을 앞에 붙여 보완한다.
        """
        prefix = " ".join(p for p in [self.doc_display_name, self.section] if p)
        return f"{prefix}\n{self.text}" if prefix else self.text

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["citation"] = self.citation
        return d


# ==================================================================
# 공용 유틸
# ==================================================================


def normalize_text(text: str) -> str:
    """PDF 추출 텍스트의 흔한 잡음을 정리한다.

    - NFC 정규화: 한글 자모가 분리되어 추출되는 경우를 합친다
    - 전각/비분리 공백을 일반 공백으로
    - 줄 끝 공백 제거, 3줄 이상 빈 줄은 2줄로 축약
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_date_from_filename(filename: str) -> date | None:
    """파일명 끝의 YYYYMMDD를 date로 파싱한다.

    파일명 규칙: hana_{유형}_{카테고리}_{상품명}_{YYYYMMDD}.pdf
    """
    m = re.search(r"_(\d{8})\.pdf$", filename, re.IGNORECASE)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None
