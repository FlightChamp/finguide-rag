"""임베딩 계층.

문장 임베딩 모델을 감싸 모델별 차이를 흡수한다. 상위 코드는 어떤 모델을
쓰는지 몰라도 encode_passages / encode_queries 만 호출하면 된다.

모델별 주의사항
-------------
E5 계열(multilingual-e5-*)은 학습 시 입력에 접두어를 붙였다.
문서에는 'passage: ', 질의에는 'query: ' 를 붙여야 학습 조건과 일치한다.
이를 생략하면 검색 성능이 눈에 띄게 떨어지는데, 오류가 나지 않고 조용히
나빠지기 때문에 놓치기 쉽다.

BGE-M3는 접두어가 필요 없다. 다국어 학습 과정에서 접두어 없이도 질의와
문서를 구분하도록 설계되었다.

정규화
-----
모든 벡터를 L2 정규화한다. 정규화된 벡터의 내적은 코사인 유사도와 같으므로,
FAISS의 IndexFlatIP(내적 검색)를 그대로 코사인 유사도 검색으로 쓸 수 있다.
정규화를 빼먹으면 긴 텍스트일수록 높은 점수를 받는 편향이 생긴다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """임베딩 모델의 사양과 사용 규약."""

    key: str                  # 내부 식별자 (인덱스 디렉토리명으로도 쓴다)
    model_id: str             # HuggingFace 모델 ID
    dim: int                  # 임베딩 차원
    max_tokens: int           # 최대 입력 토큰
    passage_prefix: str = ""  # 문서 인코딩 시 붙일 접두어
    query_prefix: str = ""    # 질의 인코딩 시 붙일 접두어


# 프로젝트에서 사용하는 모델 목록.
# 새 모델을 추가할 때는 여기에 등록만 하면 된다.
MODELS: dict[str, ModelSpec] = {
    "e5-small": ModelSpec(
        key="e5-small",
        model_id="intfloat/multilingual-e5-small",
        dim=384,
        max_tokens=512,
        passage_prefix="passage: ",
        query_prefix="query: ",
    ),
    "bge-m3": ModelSpec(
        key="bge-m3",
        model_id="BAAI/bge-m3",
        dim=1024,
        max_tokens=8192,
        # BGE-M3는 접두어를 쓰지 않는다
    ),
}

DEFAULT_MODEL = "e5-small"


class Embedder:
    """문장 임베딩 모델 래퍼."""

    def __init__(self, model_key: str = DEFAULT_MODEL, batch_size: int = 16):
        if model_key not in MODELS:
            raise ValueError(
                f"알 수 없는 모델: {model_key}. 사용 가능: {list(MODELS)}"
            )
        self.spec = MODELS[model_key]
        self.batch_size = batch_size
        self._model = None  # 실제 로딩은 첫 사용 시점까지 미룬다

    # --------------------------------------------------------------

    @property
    def model(self):
        """모델을 지연 로딩한다.

        Embedder 를 만들기만 하고 쓰지 않는 경우(예: 인덱스 메타데이터만
        읽을 때) 불필요한 2GB 로딩을 피한다.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("모델 로딩: %s", self.spec.model_id)
            self._model = SentenceTransformer(self.spec.model_id)
        return self._model

    # --------------------------------------------------------------

    def encode_passages(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """문서(청크)를 인코딩한다. 인덱스 구축에 사용한다."""
        return self._encode(texts, self.spec.passage_prefix, show_progress)

    def encode_queries(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """질의를 인코딩한다. 검색 시 사용한다.

        문서와 다른 접두어를 붙여야 하므로 메서드를 분리했다.
        같은 메서드를 쓰면 접두어를 혼동하기 쉽다.
        """
        return self._encode(texts, self.spec.query_prefix, show_progress)

    def _encode(self, texts: list[str], prefix: str, show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.spec.dim), dtype=np.float32)

        prepared = [prefix + t for t in texts] if prefix else texts

        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 정규화 -> 내적 = 코사인 유사도
            convert_to_numpy=True,
        )

        # FAISS는 float32만 받는다
        vectors = vectors.astype(np.float32)

        # 차원이 사양과 다르면 인덱스와 어긋나므로 즉시 중단한다
        if vectors.shape[1] != self.spec.dim:
            raise RuntimeError(
                f"차원 불일치: 사양 {self.spec.dim}, 실제 {vectors.shape[1]}. "
                f"MODELS 정의를 확인하세요."
            )

        return vectors

    def __repr__(self) -> str:
        return f"Embedder({self.spec.key}, dim={self.spec.dim}, max={self.spec.max_tokens})"
