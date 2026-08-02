"""파서 추상 계층.

여러 소스 형식(PDF, JSONL)을 각각의 구현체가 처리하되, 상위 파이프라인은
BaseParser 인터페이스만 알면 되도록 한다. 이후 타행(KB, 신한 등) 문서나
새로운 형식이 추가되어도 상위 코드를 고칠 필요가 없다.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..schema import Document

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """파싱 실패. 개별 문서의 실패가 전체 배치를 중단시키지 않도록 사용한다."""


class BaseParser(ABC):
    """모든 파서의 공통 인터페이스.

    구현체는 can_parse 와 parse 두 가지만 책임진다.
    """

    @abstractmethod
    def can_parse(self, source: Path) -> bool:
        """이 파서가 해당 소스를 처리할 수 있는지 판단한다."""

    @abstractmethod
    def parse(self, source: Path, **kwargs) -> list[Document]:
        """소스를 Document 목록으로 변환한다.

        PDF는 1건이 Document 1개지만, FAQ JSONL은 1파일이 여러 Document가
        되므로 반환형을 리스트로 통일한다.
        """

    def safe_parse(self, source: Path, **kwargs) -> list[Document]:
        """예외를 삼키고 로깅한다.

        108건을 배치 처리할 때 1건의 실패로 전체가 멈추면 안 된다.
        """
        try:
            return self.parse(source, **kwargs)
        except Exception as exc:
            logger.error("파싱 실패: %s (%s: %s)", source.name, type(exc).__name__, exc)
            return []
