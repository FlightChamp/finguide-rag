"""검색 → 거절 판정 → 답변 생성 통합 진입점.

왜 파이프라인을 따로 두는가
------------------------
지금까지 평가 스크립트들이 각자 Embedder, FaissStore, BM25Store,
HybridRetriever 를 조립했다. 스크립트마다 조립이 조금씩 달라지면
"데모에서는 되는데 평가에서는 다른 결과"가 나온다. 실제로 하이브리드
정규화 오류를 찾는 데 시간이 걸렸던 것도 조립 지점이 흩어져 있었기
때문이다.

데모, 평가, CLI 가 모두 이 클래스를 통과하게 해서 조립을 한 곳으로
모은다.

거절과 답변을 하나의 흐름으로
--------------------------
answer() 는 거절이든 답변이든 항상 Answer 를 반환한다. 호출부는
분기를 신경 쓰지 않고 decision 필드만 보면 된다.

    result = pipeline.answer("정기예금 중도해지하면 이자는요?")
    print(result.format_display())

거절 판정이 생성의 게이트다. REFUSE 면 생성 LLM 을 호출하지 않는다.
근거 없는 답변을 막는 가장 확실한 방법은 애초에 만들지 않는 것이다.

기본 설정
--------
검색 설정은 평가에서 확정한 값을 기본값으로 둔다. 임의로 바꾸면
측정된 성능(문서 단위 R@5 0.979, 거절 FAR 0.030)이 재현되지 않는다.

    모델      multilingual-e5-small
    융합      weighted, alpha=0.5, min-max 정규화
    top_k     10
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..embedding import Embedder, FaissStore
from ..retrieval import BM25Store, FusionMethod, HybridRetriever, NormalizeMethod
from .generator import Answer, AnswerGenerator, build_answer
from .refusal import RefusalJudge

DEFAULT_ALPHA = 0.5
DEFAULT_TOP_K = 10
DEFAULT_EVIDENCE_N = 3


class Pipeline:
    """FinGuide-RAG 의 단일 진입점."""

    def __init__(self, retriever, judge: RefusalJudge,
                 generator: AnswerGenerator | None,
                 chunk_texts: dict[str, str],
                 top_k: int = DEFAULT_TOP_K,
                 evidence_n: int = DEFAULT_EVIDENCE_N):
        self.retriever = retriever
        self.judge = judge
        self.generator = generator
        self.chunk_texts = chunk_texts
        self.top_k = top_k
        self.evidence_n = evidence_n

    # ------------------------------------------------------------------
    # 조립
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, project_root: Path, model: str = "e5-small",
              index: str | None = None, bm25: str = "default",
              alpha: float = DEFAULT_ALPHA, client=None,
              llm_model: str = "gpt-4.1-mini",
              use_llm: bool = True,
              top_k: int = DEFAULT_TOP_K,
              evidence_n: int = DEFAULT_EVIDENCE_N) -> "Pipeline":
        """인덱스와 청크를 읽어 파이프라인을 만든다.

        client 가 None 이면 LLM 근거 검증과 답변 생성이 모두 꺼진다.
        검색과 규칙 단계만으로 동작을 확인하고 싶을 때 쓴다.
        """
        faiss_dir = project_root / "data" / "indexes" / "faiss" / (index or model)
        bm25_dir = project_root / "data" / "indexes" / "bm25" / bm25
        chunks_path = project_root / "data" / "interim" / "chunks.jsonl"

        if not (faiss_dir / "config.json").exists():
            raise FileNotFoundError(f"{faiss_dir} 에 인덱스가 없습니다.")
        if not chunks_path.exists():
            raise FileNotFoundError(f"{chunks_path} 없음.")

        retriever = HybridRetriever(
            Embedder(model),
            FaissStore.load(faiss_dir),
            BM25Store.load(bm25_dir),
            method=FusionMethod.WEIGHTED,
            alpha=alpha,
            normalize=NormalizeMethod.MINMAX,
        )

        # 평가와 동일한 신호를 쓰기 위해 chunks.jsonl 을 그대로 읽는다.
        # Hit.meta 에도 text 가 들어 있지만, 거절 판정이 측정된 조건을
        # 그대로 재현하려면 평가 스크립트와 같은 출처를 써야 한다.
        chunk_texts: dict[str, str] = {}
        for line in chunks_path.open(encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                chunk_texts[c["chunk_id"]] = c.get("text", "")

        judge = RefusalJudge(client=client, model=llm_model,
                             use_llm=use_llm and client is not None)
        generator = (AnswerGenerator(client=client, model=llm_model,
                                     top_n=evidence_n)
                     if client is not None else None)

        return cls(retriever, judge, generator, chunk_texts, top_k, evidence_n)

    # ------------------------------------------------------------------
    # 실행
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Answer:
        """질문 하나를 끝까지 처리한다.

        흐름
            검색 -> 거절 판정 -> (통과 시) 답변 생성

        거절 판정이 게이트다. REFUSE 면 생성 LLM 을 호출하지 않으므로
        비용과 환각 위험이 동시에 사라진다.
        """
        started = time.perf_counter()

        hits = self.retriever.search(question, top_k=self.top_k)
        refusal = self.judge.judge(question, hits, self.chunk_texts)

        return build_answer(
            question=question,
            hits=hits,
            refusal=refusal,
            generator=self.generator,
            top_n=self.evidence_n,
            started=started,
        )

    def batch(self, questions: list[str]) -> list[Answer]:
        """여러 질문을 순차 처리한다. 평가 스크립트용."""
        return [self.answer(q) for q in questions]
