"""하이브리드 검색기.

Dense(의미 유사도)와 Sparse(어휘 매칭)를 결합한다.

왜 결합하는가
-----------
두 방식은 서로 다른 실패 양상을 보인다.

Dense는 "중도해지"와 "만기 전에 찾으면"을 연결하지만, 상품명 같은
고유명사에 약하다. 실제로 "기업 인터넷뱅킹 급여 이체" 질의에서 정답
청크가 해당 어휘를 모두 포함하는데도 20위 밖으로 밀렸다.

Sparse는 고유명사를 정확히 잡지만, 표현이 다르면 못 찾는다.
"돈 빌린 사람"과 "채무자"를 연결하지 못한다.

결합 방식
--------
두 가지를 지원하고 실험으로 고른다.

1) 가중합 (weighted)
   score = alpha * dense_norm + (1 - alpha) * sparse_norm

   문제는 점수 스케일이다. dense는 코사인 유사도로 0.86~0.92의 좁은
   구간에 몰려 있고, BM25는 0~30 범위로 질의마다 다르다. 그대로 더하면
   BM25가 지배한다. 따라서 질의별로 min-max 정규화한 뒤 결합한다.

2) RRF (Reciprocal Rank Fusion)
   score = sum(1 / (k + rank_i))

   점수 대신 순위만 쓰므로 스케일 문제가 없다. 구현이 단순하고 실무에서
   안정적으로 작동한다. 다만 점수의 크기 정보를 버리므로, 1위와 2위의
   실질적 격차가 클 때도 동일하게 취급한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FusionMethod(str, Enum):
    WEIGHTED = "weighted"   # 정규화 후 가중합
    RRF = "rrf"             # 순위 기반 융합
    DENSE_ONLY = "dense"    # 비교용
    SPARSE_ONLY = "sparse"  # 비교용


@dataclass
class HybridHit:
    """하이브리드 검색 결과 1건.

    최종 점수뿐 아니라 각 검색기의 기여도를 함께 담는다.
    어느 쪽이 이 문서를 끌어올렸는지 알아야 개선 방향이 보인다.
    """

    rank: int
    chunk_id: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int = 0     # 0이면 dense 후보에 없었음
    sparse_rank: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        """이 결과를 주로 끌어올린 검색기."""
        if self.dense_rank and not self.sparse_rank:
            return "dense"
        if self.sparse_rank and not self.dense_rank:
            return "sparse"
        if not self.dense_rank and not self.sparse_rank:
            return "none"
        return "both"


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """질의별 min-max 정규화.

    전역 정규화가 아니라 질의별로 하는 이유는, BM25 점수의 절대 크기가
    질의의 어휘 구성에 따라 크게 달라지기 때문이다. 희귀어가 포함된
    질의는 점수가 높게 나오고 흔한 어휘만 있으면 낮게 나온다. 질의 간
    비교가 목적이 아니라 한 질의 안에서의 순위가 목적이므로 질의별
    정규화가 맞다.
    """
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridRetriever:
    """Dense + Sparse 결합 검색기."""

    def __init__(
        self,
        embedder,
        faiss_store,
        bm25_store,
        method: FusionMethod = FusionMethod.WEIGHTED,
        alpha: float = 0.5,
        rrf_k: int = 60,
        candidate_k: int = 100,
    ):
        """
        alpha       : dense 가중치. 1.0이면 dense 단독, 0.0이면 sparse 단독.
        rrf_k       : RRF 상수. 클수록 상위 순위의 영향이 완만해진다.
                      원 논문(Cormack et al.)의 권장값 60을 기본으로 한다.
        candidate_k : 각 검색기에서 가져올 후보 수. 최종 top_k보다 넉넉해야
                      한쪽에서만 잡힌 문서가 결합 과정에서 살아남는다.
        """
        self.embedder = embedder
        self.faiss = faiss_store
        self.bm25 = bm25_store
        self.method = method
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

        # 청크 ID -> 인덱스 매핑. 두 검색기의 결과를 대조할 때 쓴다.
        self._id_to_meta = {m["chunk_id"]: m for m in faiss_store.metas}

    # --------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[HybridHit]:
        if self.method == FusionMethod.DENSE_ONLY:
            return self._dense_only(query, top_k)
        if self.method == FusionMethod.SPARSE_ONLY:
            return self._sparse_only(query, top_k)

        dense = self._dense_candidates(query)
        sparse = self._sparse_candidates(query)

        if self.method == FusionMethod.RRF:
            return self._fuse_rrf(dense, sparse, top_k)
        return self._fuse_weighted(dense, sparse, top_k)

    # --------------------------------------------------------------
    # 후보 수집
    # --------------------------------------------------------------

    def _dense_candidates(self, query: str) -> dict[str, tuple[int, float]]:
        """chunk_id -> (순위, 점수)"""
        qv = self.embedder.encode_queries([query])
        hits = self.faiss.search(qv, top_k=self.candidate_k)
        return {h.chunk_id: (h.rank, h.score) for h in hits}

    def _sparse_candidates(self, query: str) -> dict[str, tuple[int, float]]:
        hits = self.bm25.search(query, top_k=self.candidate_k)
        return {h.chunk_id: (h.rank, h.score) for h in hits}

    # --------------------------------------------------------------
    # 결합
    # --------------------------------------------------------------

    def _fuse_weighted(self, dense: dict, sparse: dict, top_k: int) -> list[HybridHit]:
        """정규화 후 가중합.

        한쪽에만 있는 문서는 없는 쪽 점수를 0으로 둔다. 후보에 들지
        못했다는 것 자체가 낮은 관련도의 신호이므로 타당하다.
        """
        all_ids = set(dense) | set(sparse)
        if not all_ids:
            return []

        ids = sorted(all_ids)
        d_raw = np.array([dense.get(i, (0, 0.0))[1] for i in ids])
        s_raw = np.array([sparse.get(i, (0, 0.0))[1] for i in ids])

        d_norm = minmax_normalize(d_raw)
        s_norm = minmax_normalize(s_raw)
        combined = self.alpha * d_norm + (1 - self.alpha) * s_norm

        order = np.argsort(-combined)[:top_k]

        return [
            HybridHit(
                rank=r,
                chunk_id=ids[i],
                score=float(combined[i]),
                dense_score=float(d_raw[i]),
                sparse_score=float(s_raw[i]),
                dense_rank=dense.get(ids[i], (0, 0))[0],
                sparse_rank=sparse.get(ids[i], (0, 0))[0],
                meta=self._id_to_meta.get(ids[i], {}),
            )
            for r, i in enumerate(order, 1)
        ]

    def _fuse_rrf(self, dense: dict, sparse: dict, top_k: int) -> list[HybridHit]:
        """순위 역수 합산.

        후보에 없는 검색기는 기여도 0으로 둔다.
        """
        all_ids = set(dense) | set(sparse)
        scored: list[tuple[float, str]] = []

        for cid in all_ids:
            score = 0.0
            d_rank = dense.get(cid, (0, 0.0))[0]
            s_rank = sparse.get(cid, (0, 0.0))[0]
            if d_rank:
                score += self.alpha / (self.rrf_k + d_rank)
            if s_rank:
                score += (1 - self.alpha) / (self.rrf_k + s_rank)
            scored.append((score, cid))

        scored.sort(reverse=True)

        return [
            HybridHit(
                rank=r,
                chunk_id=cid,
                score=score,
                dense_score=dense.get(cid, (0, 0.0))[1],
                sparse_score=sparse.get(cid, (0, 0.0))[1],
                dense_rank=dense.get(cid, (0, 0))[0],
                sparse_rank=sparse.get(cid, (0, 0))[0],
                meta=self._id_to_meta.get(cid, {}),
            )
            for r, (score, cid) in enumerate(scored[:top_k], 1)
        ]

    # --------------------------------------------------------------
    # 단독 모드 (비교 실험용)
    # --------------------------------------------------------------

    def _dense_only(self, query: str, top_k: int) -> list[HybridHit]:
        qv = self.embedder.encode_queries([query])
        hits = self.faiss.search(qv, top_k=top_k)
        return [
            HybridHit(rank=h.rank, chunk_id=h.chunk_id, score=h.score,
                      dense_score=h.score, dense_rank=h.rank,
                      meta=self._id_to_meta.get(h.chunk_id, {}))
            for h in hits
        ]

    def _sparse_only(self, query: str, top_k: int) -> list[HybridHit]:
        hits = self.bm25.search(query, top_k=top_k)
        return [
            HybridHit(rank=h.rank, chunk_id=h.chunk_id, score=h.score,
                      sparse_score=h.score, sparse_rank=h.rank,
                      meta=self._id_to_meta.get(h.chunk_id, {}))
            for h in hits
        ]

    def __repr__(self) -> str:
        return (f"HybridRetriever(method={self.method.value}, alpha={self.alpha}, "
                f"candidate_k={self.candidate_k})")
