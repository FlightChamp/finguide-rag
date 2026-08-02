"""FAQ 파서.

크롤링한 JSONL 200건을 Document로 변환한다.
PDF와 달리 1파일이 여러 Document가 된다.

FAQ는 이미 질문-답변 쌍으로 구조화되어 있으므로 별도 구조 판정이 불필요하다.
structure 는 QA_PAIR 로 고정하고, 청킹 단계에서 쌍 단위로 처리한다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from ..schema import Document, DocType, Structure, TitleConfidence, normalize_text
from .base import BaseParser, ParseError

logger = logging.getLogger(__name__)


class FAQParser(BaseParser):
    """faq_hana.jsonl 을 Document 목록으로 변환한다."""

    def __init__(self, bank_code: str = "hana"):
        self.bank_code = bank_code

    def can_parse(self, source: Path) -> bool:
        return source.suffix.lower() == ".jsonl" and "faq" in source.name.lower()

    def parse(self, source: Path, **kwargs) -> list[Document]:
        if not source.exists():
            raise ParseError(f"파일 없음: {source}")

        documents: list[Document] = []
        skipped = 0

        with source.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("JSON 파싱 실패 (%s:%d): %s", source.name, lineno, exc)
                    skipped += 1
                    continue

                doc = self._build(row, source)
                if doc is None:
                    skipped += 1
                    continue
                documents.append(doc)

        if skipped:
            logger.warning("FAQ %d건 건너뜀 (전체 %d건 중)", skipped, len(documents) + skipped)

        return documents

    # --------------------------------------------------------------

    def _build(self, row: dict, source: Path) -> Document | None:
        """JSONL 한 줄을 Document로 변환한다."""
        question = normalize_text(str(row.get("question", "")))
        answer = normalize_text(str(row.get("answer", "")))

        # 질문이나 답변이 비면 검색 대상으로서 가치가 없다
        if not question or not answer:
            logger.warning("빈 질문/답변: %s", row.get("faq_id", "?"))
            return None

        doc = Document(
            doc_id=row.get("faq_id", ""),
            bank_code=row.get("bank_code", self.bank_code),
            source_path=str(source).replace("\\", "/"),
            orig_filename=source.name,
            parsed_at=datetime.now(),
            # 질문 자체가 제목 역할을 한다.
            # 검색 결과에 "Q. ..." 형태로 노출되어 직원이 즉시 판단할 수 있다.
            title=question,
            text=f"Q. {question}\nA. {answer}",
            doc_type=DocType.FAQ,
            category=row.get("category", ""),
            structure=Structure.QA_PAIR,
            source_url=row.get("source_url", ""),
            title_confidence=TitleConfidence.HIGH,  # 크롤링 원본이므로 확실하다
            page_count=1,
        )

        doc.char_count = len(doc.text)
        doc.compute_hash()

        # 크롤링 시점을 시행일 대신 사용한다.
        # FAQ에는 시행일 개념이 없지만, 언제 수집된 정보인지는 밝혀야 한다.
        collected = row.get("collected_at", "")
        if collected:
            try:
                doc.effective_date = datetime.strptime(collected, "%Y-%m-%d").date()
            except ValueError:
                pass

        # status 가 active 가 아니면 인덱싱에서 제외한다
        status = row.get("status", "active")
        if status != "active":
            doc.mark_unparsable(f"status={status}")

        return doc
