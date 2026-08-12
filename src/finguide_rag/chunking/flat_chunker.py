"""길이 기반 청커와 FAQ 청커.

FlatChunker
    구조 표지가 없는 문서를 길이로 자른다. 현재는 모든 PDF에 이 방식을
    적용한다(1차 구현). 구조별 청커는 베이스라인 지표를 확보한 뒤
    도입해 개선폭을 측정한다.

FAQChunker
    FAQ는 이미 Q&A 쌍으로 구조화되어 있어 분할이 불필요하다.
    문서 1건이 곧 청크 1개다.
"""

from __future__ import annotations

from ..schema import Chunk, Document
from .base import BaseChunker


class FlatChunker(BaseChunker):
    """길이 기반 재귀 분할.

    문장 경계를 지키면서 목표 길이에 맞춰 자르고, 인접 청크 사이에
    일부를 겹쳐 경계에 걸친 내용이 유실되지 않도록 한다.
    """

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self.split_by_length(doc.text)
        return [
            self.make_chunk(doc, text, order)
            for order, text in enumerate(pieces, 1)
        ]


class FAQChunker(BaseChunker):
    """FAQ 전용.

    Q&A 쌍은 그 자체로 완결된 검색 단위이므로 원칙적으로 분할하지 않는다.
    답변만 분리되면 무슨 질문에 대한 답인지 알 수 없게 되기 때문이다.

    다만 임베딩 모델의 입력 한계를 넘으면 뒷부분이 잘린 채 인덱싱되어
    검색에서 아예 누락된다. 실측 결과 FAQ 200건 중 5건이 e5-small의
    512토큰을 초과했다(최대 887토큰). 예를 들어 '폰뱅킹 신청/해지 방법'
    FAQ는 신청이 앞, 해지가 뒤에 있어 잘리면 해지 관련 질의에 답할 수 없다.

    따라서 한계를 넘는 FAQ만 분할하되, 각 조각에 질문을 반복해 붙여
    맥락을 유지한다.
    """

    # 한국어 문자당 토큰비 실측값 0.62를 기준으로 안전 한계를 잡는다.
    # 512토큰 / 0.62 = 약 825자. 여유를 두어 700자로 설정한다.
    SPLIT_THRESHOLD_CHARS = 700

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text.strip()
        if not text:
            return []

        if len(text) <= self.SPLIT_THRESHOLD_CHARS:
            return [self.make_chunk(doc, text, 1, section="FAQ")]

        return self._split_long_faq(doc, text)

    def _split_long_faq(self, doc: Document, text: str) -> list[Chunk]:
        """긴 FAQ를 분할하되 각 조각에 질문을 반복해 붙인다.

        질문 없이 답변 일부만 남으면 그 청크는 무슨 맥락인지 알 수 없어
        검색되어도 쓸모가 없다.
        """
        question, answer = self._split_qa(text)

        # 질문 접두어가 차지하는 몫을 빼고 답변을 자른다
        prefix = f"Q. {question}\nA. " if question else ""
        budget = max(self.target_chars - len(prefix), self.min_chars)

        # target 뿐 아니라 max 도 함께 낮춰야 한다.
        # split_by_length 는 전체 길이가 max_chars 이하면 분할하지 않고
        # 그대로 반환하므로, max 를 낮추지 않으면 700~900자 구간이
        # 분할되지 않는 사각지대가 된다.
        saved = (self.target_chars, self.max_chars)
        self.target_chars = budget
        self.max_chars = budget
        try:
            pieces = self.split_by_length(answer)
        finally:
            self.target_chars, self.max_chars = saved

        if not pieces:
            pieces = [answer]

        chunks: list[Chunk] = []
        for order, piece in enumerate(pieces, 1):
            body = f"{prefix}{piece}" if prefix else piece
            chunks.append(self.make_chunk(doc, body, order, section="FAQ"))
        return chunks

    @staticmethod
    def _split_qa(text: str) -> tuple[str, str]:
        """'Q. ...\\nA. ...' 형태에서 질문과 답변을 분리한다."""
        if text.startswith("Q. ") and "\nA. " in text:
            head, _, tail = text.partition("\nA. ")
            return head[3:].strip(), tail.strip()
        return "", text
