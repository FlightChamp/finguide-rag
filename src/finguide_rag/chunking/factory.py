"""청커 팩토리.

문서의 structure 값에 따라 적절한 청커를 반환한다.

두 가지 모드
-----------
structural=False (베이스라인)
    FAQ를 제외한 모든 구조에 FlatChunker를 쓴다. 길이 기반 분할이다.

structural=True (구조별 청킹)
    조항·항·번호 단위로 문서의 의미 경계를 따라 자른다.

두 모드를 모두 유지하는 이유는 비교 실험 때문이다. 구조별 청킹이
Recall@5를 실제로 몇 %p 올리는지 같은 평가셋으로 측정해야 개선을
주장할 수 있다. 지표 없이 정교하게 만들면 개선인지 알 수 없다.
"""

from __future__ import annotations

from ..schema import Chunk, Document, Structure
from .base import BaseChunker
from .flat_chunker import FAQChunker, FlatChunker
from .structural_chunker import ArticleChunker, ClauseChunker, NumberedChunker


class ChunkerFactory:
    """structure 값으로 청커를 선택한다."""

    def __init__(self, structural: bool = False, **kwargs):
        flat = FlatChunker(**kwargs)
        faq = FAQChunker(**kwargs)

        self.structural = structural

        if structural:
            self._chunkers: dict[Structure, BaseChunker] = {
                Structure.QA_PAIR: faq,
                Structure.ARTICLE: ArticleChunker(**kwargs),
                Structure.CLAUSE: ClauseChunker(**kwargs),
                Structure.NUMBERED: NumberedChunker(**kwargs),
                # 구조 표지가 없는 문서는 길이 기반으로 자를 수밖에 없다
                Structure.FLAT: flat,
            }
        else:
            self._chunkers = {
                Structure.QA_PAIR: faq,
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
