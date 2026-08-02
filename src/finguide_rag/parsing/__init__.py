"""문서 파싱 계층.

PDF/JSONL 등 이질적 소스를 공통 Document 스키마로 정규화한다.
"""

from .base import BaseParser, ParseError
from .factory import ParserFactory
from .faq_parser import FAQParser
from .pdf_parser import PDFParser

__all__ = [
    "BaseParser",
    "ParseError",
    "ParserFactory",
    "FAQParser",
    "PDFParser",
]
