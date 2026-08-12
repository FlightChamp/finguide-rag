"""청커 팩토리.

문서의 structure 값에 따라 적절한 청커를 반환한다.

현재 상태 (1차 구현)
------------------
FAQ를 제외한 모든 구조에 FlatChunker를 사용한다. 파이프라인을 먼저
관통시켜 베이스라인 지표를 확보하는 것이 목적이기 때문이다.

구조별 청커(ArticleChunker 등)는 그 다음 단계에서 추가한다. 베이스라인이
있어야 "구조별 청킹이 Recall@5를 몇 %p 올렸는지" 측정할 수 있다.
지표 없이 정교하게 만들면 개선인지 아닌지 알 수 없다.

추가 시에는 _chunkers 딕셔너리에 등록만 하면 되고, 상위 코드는
고칠 필요가 없다.
"""

from __future__ import annotations

from ..schema import Chunk, Document, Structure
from .base import BaseChunker
from .flat_chunker import FAQChunker, FlatChunker


class ChunkerFactory:
    """structure 값으로 청커를 선택한다."""

    def __init__(self, **kwargs):
        flat = FlatChunker(**kwargs)
        faq = FAQChunker(**kwargs)

        self._chunkers: dict[Structure, BaseChunker] = {
            Structure.QA_PAIR: faq,
            # 아래는 모두 flat으로 처리한다(1차 구현).
            # 구조별 청커 도입 시 여기만 교체하면 된다.
            Structure.ARTICLE: flat,
            Structure.CLAUSE: flat,
            Structure.NUMBERED: flat,
            Structure.FLAT: flat,
        }
        self._default = flat

    def register(self, structure: Structure, chunker: BaseChunker) -> None:
        """청커를 교체하거나 추가한다."""
        self._chunkers[structure] = chunker

    def get_chunker(self, structure: Structure) -> BaseChunker:
        return self._chunkers.get(structure, self._default)

    def chunk(self, doc: Document) -> list[Chunk]:
        """문서를 청킹한다. 인덱싱 대상이 아니면 빈 목록을 반환한다."""
        if not doc.is_parsable:
            return []
        return self.get_chunker(doc.structure).chunk(doc)
