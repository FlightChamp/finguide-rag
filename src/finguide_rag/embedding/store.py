"""FAISS 벡터 스토어.

벡터 인덱스와 청크 메타데이터를 함께 관리한다.

왜 매핑을 따로 저장하는가
---------------------
FAISS는 "3번째 벡터가 가장 가깝다"는 정보만 반환한다. 3번째가 어느
청크인지는 우리가 기록해야 한다. 이 매핑이 어긋나면 검색은 정상적으로
동작하는데 엉뚱한 문서가 나오는, 가장 찾기 어려운 종류의 버그가 된다.

그래서 인덱스와 메타데이터를 같은 디렉토리에 함께 저장하고, 로드할 때
개수가 일치하는지 검증한다.

왜 IndexFlatIP인가
----------------
현재 청크는 약 2,700개다. 이 규모에서는 전수 비교(brute force)가
밀리초 안에 끝나므로 근사 검색(IVF, HNSW)을 쓸 이유가 없다. 근사 검색은
정확도를 일부 포기하는 대신 속도를 얻는 기법인데, 지금은 포기할 이유가
없다. 수십만 벡터 규모가 되면 그때 교체하면 된다.

IndexFlatIP는 내적 검색이며, 벡터가 L2 정규화되어 있으면 코사인 유사도와
같다. Embedder가 항상 정규화하므로 이 전제가 성립한다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """검색 결과 1건."""

    rank: int
    score: float
    chunk_id: str
    doc_id: str
    text: str
    citation: str
    doc_display_name: str
    doc_type: str
    category: str
    section: str
    effective_date: str
    source_url: str

    @classmethod
    def from_meta(cls, rank: int, score: float, meta: dict) -> "SearchHit":
        return cls(
            rank=rank,
            score=score,
            chunk_id=meta.get("chunk_id", ""),
            doc_id=meta.get("doc_id", ""),
            text=meta.get("text", ""),
            citation=meta.get("citation", ""),
            doc_display_name=meta.get("doc_display_name", ""),
            doc_type=meta.get("doc_type", ""),
            category=meta.get("category", ""),
            section=meta.get("section", ""),
            effective_date=meta.get("effective_date", ""),
            source_url=meta.get("source_url", ""),
        )


class FaissStore:
    """FAISS 인덱스 + 청크 메타데이터."""

    INDEX_FILE = "index.faiss"
    META_FILE = "chunks_meta.jsonl"
    CONFIG_FILE = "config.json"

    def __init__(self, dim: int, model_key: str = ""):
        self.dim = dim
        self.model_key = model_key
        self._index = None
        self.metas: list[dict] = []

    # --------------------------------------------------------------
    # 구축
    # --------------------------------------------------------------

    def build(self, vectors: np.ndarray, metas: list[dict]) -> None:
        """벡터와 메타데이터로 인덱스를 만든다.

        벡터의 i번째와 metas의 i번째가 같은 청크여야 한다.
        이 순서가 어긋나면 검색 결과 전체가 무의미해지므로 개수를 검증한다.
        """
        import faiss

        if len(vectors) != len(metas):
            raise ValueError(
                f"벡터 {len(vectors)}개와 메타데이터 {len(metas)}개의 수가 다릅니다."
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"차원 불일치: 인덱스 {self.dim}, 벡터 {vectors.shape[1]}"
            )

        index = faiss.IndexFlatIP(self.dim)
        index.add(vectors.astype(np.float32))

        self._index = index
        self.metas = metas
        logger.info("인덱스 구축 완료: %d개 벡터", index.ntotal)

    # --------------------------------------------------------------
    # 저장 / 로드
    # --------------------------------------------------------------

    def save(self, directory: Path) -> None:
        import faiss

        if self._index is None:
            raise RuntimeError("인덱스가 없습니다. build()를 먼저 호출하세요.")

        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / self.INDEX_FILE))

        with (directory / self.META_FILE).open("w", encoding="utf-8") as f:
            for meta in self.metas:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        config = {
            "model_key": self.model_key,
            "dim": self.dim,
            "n_vectors": self._index.ntotal,
            "index_type": "IndexFlatIP",
            "normalized": True,
        }
        (directory / self.CONFIG_FILE).write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info("저장 완료: %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "FaissStore":
        """저장된 인덱스를 읽는다. 정합성을 검증한다."""
        import faiss

        config_path = directory / cls.CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(f"{directory} 에 인덱스가 없습니다.")

        config = json.loads(config_path.read_text(encoding="utf-8"))

        store = cls(dim=config["dim"], model_key=config.get("model_key", ""))
        store._index = faiss.read_index(str(directory / cls.INDEX_FILE))

        with (directory / cls.META_FILE).open(encoding="utf-8") as f:
            store.metas = [json.loads(line) for line in f if line.strip()]

        # 매핑이 깨졌는지 여기서 잡는다.
        # 이 검증이 없으면 잘못된 검색 결과를 정상으로 착각하게 된다.
        if store._index.ntotal != len(store.metas):
            raise RuntimeError(
                f"인덱스({store._index.ntotal})와 메타데이터({len(store.metas)})의 "
                f"수가 다릅니다. 인덱스를 다시 구축하세요."
            )

        return store

    # --------------------------------------------------------------
    # 검색
    # --------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        """질의 벡터와 가장 가까운 청크 top_k개를 반환한다."""
        if self._index is None:
            raise RuntimeError("인덱스가 없습니다.")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        scores, indices = self._index.search(
            query_vector.astype(np.float32), min(top_k, self._index.ntotal)
        )

        hits: list[SearchHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            # FAISS는 결과가 부족하면 -1을 채워 반환한다
            if idx < 0:
                continue
            hits.append(SearchHit.from_meta(rank, float(score), self.metas[idx]))
        return hits

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def __repr__(self) -> str:
        return f"FaissStore(model={self.model_key}, dim={self.dim}, n={self.size})"
