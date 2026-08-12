"""청킹 추상 계층.

문서를 검색 단위(Chunk)로 분할한다. 문서 구조에 따라 자르는 기준이
다르므로, 구조별 구현체를 두고 팩토리가 라우팅한다.

왜 구조별로 나누는가
------------------
약관의 '제7조(이자)'는 하나의 완결된 주제다. 이 경계를 무시하고 400자씩
기계적으로 자르면 제7조 후반과 제8조 전반이 한 청크에 섞인다. 사람이
만들어 둔 의미 경계를 그대로 쓰는 편이 검색 정확도에 유리하다.

다만 모든 문서에 조항 구조가 있는 건 아니다. 전수조사 결과 104건 중
34건은 구조 표지가 없는 서식·안내문이었다. 이런 문서는 길이로 자를
수밖에 없다.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from ..schema import Chunk, Document

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 기본 파라미터
# ------------------------------------------------------------------

# 목표 청크 크기(한글 문자 수).
# 임베딩 모델 선정 시 확정한 값. e5-small은 512토큰이 상한인데
# 한국어는 대략 1자당 1~1.5토큰이므로 500자면 500~750토큰이 된다.
# 초과 여부는 청킹 결과의 token_count 필드로 실측해 조정한다.
DEFAULT_TARGET_CHARS = 500
DEFAULT_MAX_CHARS = 900
DEFAULT_MIN_CHARS = 100
DEFAULT_OVERLAP_CHARS = 80


class BaseChunker(ABC):
    """모든 청커의 공통 인터페이스."""

    def __init__(
        self,
        target_chars: int = DEFAULT_TARGET_CHARS,
        max_chars: int = DEFAULT_MAX_CHARS,
        min_chars: int = DEFAULT_MIN_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    ):
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """문서를 청크 목록으로 분할한다."""

    # --------------------------------------------------------------
    # 공통 유틸
    # --------------------------------------------------------------

    def make_chunk(self, doc: Document, text: str, order: int, section: str = "") -> Chunk:
        """Document의 메타데이터를 상속한 Chunk를 만든다.

        검색 결과로 청크만 반환되어도 출처를 표시할 수 있어야 하므로
        문서명·시행일 등을 복사해 둔다.
        """
        return Chunk(
            chunk_id=f"{doc.doc_id}_c{order:03d}",
            doc_id=doc.doc_id,
            text=text.strip(),
            order=order,
            section=section,
            doc_display_name=doc.display_name,
            doc_type=doc.doc_type.value,
            category=doc.category,
            effective_date=doc.effective_date.isoformat() if doc.effective_date else "",
            source_url=doc.source_url,
            char_count=len(text.strip()),
            metadata={
                "structure": doc.structure.value,
                "compliance_no": doc.compliance_no,
                "is_latest": doc.is_latest,
            },
        )

    def split_by_length(self, text: str) -> list[str]:
        """길이 기준으로 자른다. 문장 경계를 최대한 지킨다.

        문장 중간에서 끊으면 의미가 훼손되므로, 목표 길이 근처의
        문장 끝을 찾아 자른다. 이전 청크의 끝부분을 일부 겹쳐서
        경계에 걸친 내용이 양쪽 청크에서 모두 검색되도록 한다.
        """
        if len(text) <= self.max_chars:
            return [text] if len(text) >= self.min_chars else []

        sentences = self.split_sentences(text)
        chunks: list[str] = []
        buffer = ""

        for sent in sentences:
            # 문장 하나가 최대 길이를 넘으면 그것만 따로 강제 분할한다
            if len(sent) > self.max_chars:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(self.force_split(sent))
                continue

            if len(buffer) + len(sent) > self.target_chars and buffer:
                chunks.append(buffer)
                # 겹침: 직전 청크의 끝부분을 다음 청크 앞에 붙인다
                tail = buffer[-self.overlap_chars:] if self.overlap_chars else ""
                buffer = (tail + " " + sent).strip() if tail else sent
            else:
                buffer = (buffer + " " + sent).strip() if buffer else sent

        if buffer:
            chunks.append(buffer)

        return [c for c in chunks if len(c) >= self.min_chars]

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """한국어 문장 분리.

        마침표만으로 자르면 '제3조.' 같은 표현이나 소수점에서 잘못 끊긴다.
        종결어미(다/요/음/함) 뒤의 구두점을 기준으로 삼는다.
        줄바꿈도 문장 경계로 취급한다(PDF는 항목이 줄 단위로 나뉜다).
        """
        # 줄바꿈 먼저 처리
        parts: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 종결어미 + 구두점 뒤에서 분리
            sents = re.split(r"(?<=[다요음함][.!?])\s+", line)
            parts.extend(s.strip() for s in sents if s.strip())
        return parts

    def force_split(self, text: str) -> list[str]:
        """문장 하나가 지나치게 길 때 강제로 자른다.

        표가 한 줄로 추출되는 경우 등에서 발생한다.
        """
        out: list[str] = []
        step = self.target_chars
        for i in range(0, len(text), step):
            piece = text[i:i + step].strip()
            if len(piece) >= self.min_chars:
                out.append(piece)
        return out
