"""구조 기반 청커.

문서가 스스로 드러내는 의미 경계를 따라 자른다.

왜 구조를 따르는가
----------------
약관의 '제7조(이자)'는 하나의 완결된 주제다. 이 경계를 무시하고 500자씩
기계적으로 자르면 제7조 후반과 제8조 전반이 한 청크에 섞인다. 그러면
"이자는 언제 지급되나요" 질의에 대해 이자 조항과 무관한 내용이 함께
검색되어 정밀도가 떨어진다.

반대로 조항을 그대로 쓰면 청크마다 하나의 주제만 담기므로, 임베딩이
그 주제를 선명하게 표현한다.

베이스라인에서 약관 Recall@5 가 0.565 로 가장 낮았고, 하이브리드 적용
후에도 0.652 로 여전히 최저였다. 약관은 조항 구조가 뚜렷한 문서 유형이
므로 이 전략의 효과가 가장 클 것으로 예상한다.

공통 처리
--------
조항이 최대 길이를 넘으면 하위 단위(항)로 재귀 분할하고, 그래도 넘으면
길이 기반으로 자른다. 반대로 지나치게 짧은 조각은 앞 조각에 병합한다.
'제1조 목적' 같은 한 줄짜리 조항이 단독 청크가 되면 검색 노이즈가 된다.
"""

from __future__ import annotations

import re

from ..schema import Chunk, Document
from .base import BaseChunker

# 조항 표지: 제1조, 제 12 조, 제3조(목적)
RE_ARTICLE = re.compile(r"(?=제\s*\d+\s*조)")
RE_ARTICLE_HEAD = re.compile(r"^제\s*\d+\s*조\s*[(（]?([^)）\n]{0,30})?")

# 항 표지: ①②③ ... ⑮
RE_CLAUSE = re.compile(r"(?=[\u2460-\u246e])")

# 번호 표지: 줄 시작의 "1." "2)" "가." 등
RE_NUMBERED = re.compile(r"(?=^\s*(?:\d{1,2}[.)]|[가-힣][.)])\s)", re.MULTILINE)


class StructuralChunker(BaseChunker):
    """구분자 정규식으로 분할하는 청커의 공통 구현.

    하위 클래스는 splitter(분할 정규식)와 section_label(구간 이름 추출)만
    정의하면 된다.
    """

    splitter: re.Pattern = RE_ARTICLE
    fallback_to_flat: bool = True

    # 이 길이 미만인 조각만 인접 조각과 병합한다.
    # min_chars(100자)를 쓰면 정상 조항까지 묶여 구조 청킹의 이점이 사라진다.
    merge_threshold: int = 60

    # 이 길이 미만인 조각은 아예 버린다.
    # 구조 분할 과정에서 표지만 남거나 공백뿐인 조각이 생길 수 있는데,
    # 임베딩해도 의미가 없고 검색 노이즈가 된다.
    min_fragment: int = 20

    def section_label(self, text: str, order: int) -> str:
        """이 조각의 구간 이름. 검색 결과의 출처 표시에 쓴다."""
        return ""

    # --------------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        pieces = self.split_structurally(doc.text)

        # 구조 표지를 못 찾으면 길이 기반으로 되돌린다.
        # 구조 판정이 잘못된 문서를 통째로 잃지 않기 위한 안전장치다.
        if len(pieces) < 2 and self.fallback_to_flat:
            pieces = [(t, "") for t in self.split_by_length(doc.text)]

        chunks: list[Chunk] = []
        for text, section in pieces:
            body = text.strip()
            # 구조 분할 과정에서 빈 조각이나 표지만 남은 조각이 생길 수 있다.
            # 이런 청크는 임베딩해도 의미가 없고 검색 노이즈가 된다.
            if len(body) < self.min_fragment:
                continue
            chunks.append(
                self.make_chunk(doc, body, len(chunks) + 1, section=section)
            )

        # 모두 걸러졌으면 길이 기반으로 되돌린다
        if not chunks and doc.text.strip():
            for order, text in enumerate(self.split_by_length(doc.text), 1):
                chunks.append(self.make_chunk(doc, text, order))

        return chunks

    # --------------------------------------------------------------

    def split_structurally(self, text: str) -> list[tuple[str, str]]:
        """구분자로 나눈 뒤 길이를 조정한다.

        반환값은 (본문, 구간명) 목록이다.
        """
        raw = [p.strip() for p in self.splitter.split(text) if p.strip()]
        if not raw:
            return []

        # 1) 지나치게 짧은 조각만 병합한다.
        #    '제1조(목적) 이 약관은 ...을 정함을 목적으로 한다' 같은 한 줄
        #    조항이 단독 청크가 되면 문맥이 없어 검색 노이즈가 된다.
        #
        #    다만 병합 기준을 min_chars(100자)로 두면 200~400자짜리 정상
        #    조항까지 묶여 조항 단위 청킹의 이점이 사라진다. 실제로
        #    제1~3조가 한 덩어리가 되는 현상이 확인됐다. 따라서 훨씬 낮은
        #    기준(merge_threshold)을 별도로 둔다.
        merged: list[str] = []
        for piece in raw:
            if merged and len(piece) < self.merge_threshold:
                merged[-1] = merged[-1] + "\n" + piece
            elif merged and len(merged[-1]) < self.merge_threshold:
                # 직전 조각이 너무 짧으면 현재 조각과 합친다
                merged[-1] = merged[-1] + "\n" + piece
            else:
                merged.append(piece)

        # 2) 너무 긴 조각은 다시 자른다
        out: list[tuple[str, str]] = []
        for i, piece in enumerate(merged, 1):
            label = self.section_label(piece, i)
            if len(piece) <= self.max_chars:
                out.append((piece, label))
                continue

            for sub in self.split_oversized(piece):
                out.append((sub, label))

        return out

    def split_oversized(self, text: str) -> list[str]:
        """최대 길이를 넘는 조각을 하위 단위로 다시 자른다.

        먼저 항(①②③)으로 나눠 보고, 그래도 길면 길이 기반으로 자른다.
        조항 하나가 2,000자를 넘는 경우가 실제로 있다.
        """
        subs = [p.strip() for p in RE_CLAUSE.split(text) if p.strip()]

        if len(subs) > 1:
            # 항 단위로 나눈 뒤 목표 길이에 맞춰 다시 묶는다
            packed: list[str] = []
            buf = ""
            for sub in subs:
                if len(buf) + len(sub) > self.target_chars and buf:
                    packed.append(buf)
                    buf = sub
                else:
                    buf = (buf + "\n" + sub).strip() if buf else sub
            if buf:
                packed.append(buf)

            # 그래도 넘치는 것만 길이 기반으로 처리
            out: list[str] = []
            for p in packed:
                if len(p) <= self.max_chars:
                    out.append(p)
                else:
                    out.extend(self.split_by_length(p))
            return out or [text[:self.max_chars]]

        return self.split_by_length(text) or [text[:self.max_chars]]


class ArticleChunker(StructuralChunker):
    """제N조 단위 분할. 약관 표준형(36건)에 적용한다."""

    splitter = RE_ARTICLE

    def section_label(self, text: str, order: int) -> str:
        """'제7조(이자)' 형태의 구간명을 뽑는다.

        검색 결과에 '예금거래 기본약관 제7조(이자)' 처럼 표시되면
        직원이 원문을 바로 찾아갈 수 있다.
        """
        m = re.match(r"제\s*(\d+)\s*조\s*[(（]?\s*([^)）\n]{0,24})?", text)
        if not m:
            return ""
        num, title = m.group(1), (m.group(2) or "").strip()
        # 제목이 본문으로 이어진 경우를 걸러낸다
        if len(title) > 20 or title.endswith(("다", "다.", "함", "음")):
            title = ""
        return f"제{num}조({title})" if title else f"제{num}조"


class ClauseChunker(StructuralChunker):
    """항(①②③) 단위 분할. 조는 없으나 항 구조가 뚜렷한 문서(18건)."""

    splitter = RE_CLAUSE

    # 항은 조보다 짧으므로 여러 개를 묶어야 적정 길이가 된다.
    # split_structurally 의 병합 로직이 이를 처리한다.

    def section_label(self, text: str, order: int) -> str:
        m = re.match(r"([\u2460-\u246e])", text)
        return m.group(1) if m else ""

    def split_structurally(self, text: str) -> list[tuple[str, str]]:
        """항을 목표 길이에 맞춰 묶는다.

        항 하나는 보통 100~300자라 그대로 두면 청크가 지나치게 잘게
        쪼개진다. 검색 단위로는 너무 작아 문맥이 부족하다.
        """
        raw = [p.strip() for p in self.splitter.split(text) if p.strip()]
        if len(raw) < 2:
            return [(t, "") for t in self.split_by_length(text)]

        packed: list[tuple[str, str]] = []
        buf = ""
        first_label = ""

        for piece in raw:
            label = self.section_label(piece, 0)
            if not buf:
                first_label = label
            if len(buf) + len(piece) > self.target_chars and buf:
                packed.append((buf, first_label))
                buf, first_label = piece, label
            else:
                buf = (buf + "\n" + piece).strip() if buf else piece

        if buf:
            packed.append((buf, first_label))

        # 최대 길이 초과분 처리
        out: list[tuple[str, str]] = []
        for body, label in packed:
            if len(body) <= self.max_chars:
                out.append((body, label))
            else:
                out.extend((s, label) for s in self.split_by_length(body))
        return out


class NumberedChunker(StructuralChunker):
    """번호 목록 단위 분할. 항목 나열형 문서(16건)."""

    splitter = RE_NUMBERED

    def section_label(self, text: str, order: int) -> str:
        m = re.match(r"\s*(\d{1,2}[.)]|[가-힣][.)])", text)
        return m.group(1).rstrip(".)") if m else ""

    def split_structurally(self, text: str) -> list[tuple[str, str]]:
        """번호 항목을 목표 길이에 맞춰 묶는다.

        항목 하나가 한 줄인 경우가 많아 그대로 두면 지나치게 잘게 나뉜다.
        """
        raw = [p.strip() for p in self.splitter.split(text) if p.strip()]
        if len(raw) < 2:
            return [(t, "") for t in self.split_by_length(text)]

        packed: list[tuple[str, str]] = []
        buf = ""
        first_label = ""

        for piece in raw:
            label = self.section_label(piece, 0)
            if not buf:
                first_label = label
            if len(buf) + len(piece) > self.target_chars and buf:
                packed.append((buf, first_label))
                buf, first_label = piece, label
            else:
                buf = (buf + "\n" + piece).strip() if buf else piece

        if buf:
            packed.append((buf, first_label))

        out: list[tuple[str, str]] = []
        for body, label in packed:
            if len(body) <= self.max_chars:
                out.append((body, label))
            else:
                out.extend((s, label) for s in self.split_by_length(body))
        return out
