"""BM25 sparse 검색기.

Dense 검색이 놓치는 어휘 매칭을 보완한다.

왜 필요한가
---------
베이스라인 실패 분석에서, 실패한 23건 중 20건(87%)이 정답 청크에 질의
어휘를 25% 이상 포함하고 있었다. 예컨대 "기업 인터넷뱅킹에서 급여 이체는
언제까지 할 수 있나요?" 는 정답 청크에 '기업', '인터넷뱅킹', '급여',
'이체' 가 모두 등장하는데도 20위 밖으로 밀렸다. 어휘 기반 검색이라면
쉽게 찾을 질의다.

특히 약관 실패 10건 중 9건이 20위 밖이었다. 순위가 낮은 것이 아니라
후보에 아예 들지 못한 것이므로, 재순위화가 아니라 후보 확보 단계에서
손봐야 한다.

한국어 처리
---------
공백으로만 자르면 '대출을'과 '대출은'이 다른 토큰이 되어 매칭이 실패한다.
형태소 분석기로 어간을 추출해야 한다. Kiwi를 쓰는 이유는 Java 의존성이
없어 Windows 환경에서 재현이 쉽기 때문이다(KoNLPy 계열은 JVM 필요).
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 색인 대상 품사.
# 명사·동사·형용사·외국어·숫자만 남기고 조사·어미·기호는 버린다.
# 조사를 남기면 '대출을'과 '대출은'이 다른 토큰이 되어 매칭이 깨진다.
CONTENT_POS = {
    "NNG",   # 일반명사
    "NNP",   # 고유명사
    "NNB",   # 의존명사
    "NR",    # 수사
    "VV",    # 동사
    "VA",    # 형용사
    "MAG",   # 일반부사
    "SL",    # 외국어 (ISA, ATM 등)
    "SH",    # 한자
    "SN",    # 숫자
}

# 색인에서 제외할 어휘.
# 거의 모든 문서에 등장해 변별력이 없다. BM25의 IDF가 가중치를 낮추긴
# 하지만, 아예 빼는 편이 노이즈가 줄어든다.
STOPWORDS = {
    "하다", "되다", "있다", "없다", "같다", "이다", "아니다",
    "것", "수", "등", "및", "제", "호", "항", "때", "경우",
    "위", "아래", "다음", "각", "본", "해당", "관련",
}

MIN_TOKEN_LEN = 2


@dataclass
class SparseHit:
    """BM25 검색 결과 1건."""

    rank: int
    score: float
    chunk_id: str
    index: int   # 원본 리스트에서의 위치. dense 결과와 대조할 때 쓴다


class KiwiTokenizer:
    """Kiwi 형태소 분석기 래퍼.

    Kiwi 로딩에 시간이 걸리므로 인스턴스를 재사용한다.
    """

    def __init__(self):
        self._kiwi = None

    @property
    def kiwi(self):
        if self._kiwi is None:
            from kiwipiepy import Kiwi

            logger.info("Kiwi 형태소 분석기 로딩")
            self._kiwi = Kiwi()
        return self._kiwi

    def tokenize(self, text: str) -> list[str]:
        """텍스트를 색인용 토큰 목록으로 바꾼다.

        형태소 분석 결과에서 내용어만 남긴다. 동사·형용사는 어간에
        '다'를 붙여 사전형으로 통일한다('갚으면' -> '갚다').
        """
        if not text.strip():
            return []

        tokens: list[str] = []
        for token in self.kiwi.tokenize(text):
            if token.tag not in CONTENT_POS:
                continue

            form = token.form
            # 용언은 사전형으로 정규화한다
            if token.tag in ("VV", "VA"):
                form = form + "다"

            if len(form) < MIN_TOKEN_LEN and token.tag not in ("SL", "SN"):
                continue
            if form in STOPWORDS:
                continue

            tokens.append(form)

        return tokens

    def tokenize_batch(self, texts: list[str], show_progress: bool = False) -> list[list[str]]:
        """여러 문서를 한 번에 처리한다."""
        out: list[list[str]] = []
        total = len(texts)
        for i, text in enumerate(texts, 1):
            out.append(self.tokenize(text))
            if show_progress and i % 500 == 0:
                print(f"    {i:,}/{total:,} 토큰화")
        return out


class BM25Store:
    """BM25 인덱스와 청크 매핑."""

    INDEX_FILE = "bm25.pkl"

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        # BM25 하이퍼파라미터.
        # k1: 용어 빈도 포화 지점. 클수록 반복 등장에 민감하다.
        # b : 문서 길이 정규화 강도. 우리 청크는 길이가 고른 편(중앙값
        #     479자)이라 기본값으로 충분하다.
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self.chunk_ids: list[str] = []
        self.tokenizer = KiwiTokenizer()

    # --------------------------------------------------------------

    def build(self, chunk_ids: list[str], texts: list[str], show_progress: bool = True) -> None:
        from rank_bm25 import BM25Okapi

        if len(chunk_ids) != len(texts):
            raise ValueError(f"ID {len(chunk_ids)}개와 텍스트 {len(texts)}개의 수가 다릅니다.")

        if show_progress:
            print(f"  형태소 분석 중... ({len(texts):,}개)")
        tokenized = self.tokenizer.tokenize_batch(texts, show_progress)

        empty = sum(1 for t in tokenized if not t)
        if empty:
            logger.warning("토큰이 비어 있는 청크 %d개", empty)

        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self.chunk_ids = list(chunk_ids)

        avg_tokens = sum(len(t) for t in tokenized) / len(tokenized)
        if show_progress:
            print(f"  BM25 인덱스 구축 완료 (청크당 평균 {avg_tokens:.0f}토큰)")

    # --------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[SparseHit]:
        """질의와 어휘가 겹치는 청크를 점수순으로 반환한다."""
        if self._bm25 is None:
            raise RuntimeError("인덱스가 없습니다. build() 또는 load()를 먼저 호출하세요.")

        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # 점수 상위 k개의 인덱스를 뽑는다
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        return [
            SparseHit(rank=r, score=float(scores[i]), chunk_id=self.chunk_ids[i], index=i)
            for r, i in enumerate(ranked, 1)
            if scores[i] > 0   # 점수 0은 겹치는 어휘가 없다는 뜻이다
        ]

    def score_all(self, query: str) -> list[float]:
        """전체 청크에 대한 점수를 반환한다. 하이브리드 결합에 쓴다."""
        if self._bm25 is None:
            raise RuntimeError("인덱스가 없습니다.")
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return [0.0] * len(self.chunk_ids)
        return [float(s) for s in self._bm25.get_scores(tokens)]

    # --------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / self.INDEX_FILE).open("wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunk_ids": self.chunk_ids,
                "k1": self.k1,
                "b": self.b,
            }, f)
        logger.info("BM25 인덱스 저장: %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "BM25Store":
        path = directory / cls.INDEX_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} 없음. 먼저 BM25 인덱스를 구축하세요.")

        with path.open("rb") as f:
            data = pickle.load(f)

        store = cls(k1=data.get("k1", 1.2), b=data.get("b", 0.75))
        store._bm25 = data["bm25"]
        store.chunk_ids = data["chunk_ids"]
        return store

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def __repr__(self) -> str:
        return f"BM25Store(n={self.size}, k1={self.k1}, b={self.b})"


# ------------------------------------------------------------------
# 진단
# ------------------------------------------------------------------


def explain_tokens(tokenizer: KiwiTokenizer, text: str) -> str:
    """토큰화 결과를 사람이 읽을 수 있게 보여준다.

    검색이 왜 실패하는지 진단할 때 쓴다. 질의와 문서의 토큰이
    실제로 겹치는지 눈으로 확인할 수 있다.
    """
    return " · ".join(tokenizer.tokenize(text))
