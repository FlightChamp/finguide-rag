"""파서 팩토리.

소스 파일을 받아 적절한 파서를 골라준다. 상위 코드는 어떤 파서가
쓰이는지 알 필요가 없다. 타행 문서나 새 형식(HTML, XML 등)이 추가되면
여기에 등록만 하면 된다.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import Document
from .base import BaseParser
from .faq_parser import FAQParser
from .pdf_parser import PDFParser


class ParserFactory:
    """등록된 파서 중 소스를 처리할 수 있는 것을 찾아 반환한다."""

    def __init__(self, bank_code: str = "hana"):
        # 순서가 중요하다. 먼저 등록된 파서부터 can_parse 를 묻는다.
        self._parsers: list[BaseParser] = [
            FAQParser(bank_code=bank_code),
            PDFParser(bank_code=bank_code),
        ]

    def register(self, parser: BaseParser, prepend: bool = False) -> None:
        """파서를 추가한다. 기존 파서보다 우선하려면 prepend=True."""
        if prepend:
            self._parsers.insert(0, parser)
        else:
            self._parsers.append(parser)

    def get_parser(self, source: Path) -> BaseParser | None:
        for parser in self._parsers:
            if parser.can_parse(source):
                return parser
        return None

    def parse(self, source: Path, **kwargs) -> list[Document]:
        """소스를 파싱한다. 처리할 파서가 없으면 빈 리스트를 반환한다."""
        parser = self.get_parser(source)
        if parser is None:
            return []
        return parser.safe_parse(source, **kwargs)
