"""V2Service — 화면(Streamlit)과 검증 계층 사이의 얇은 서비스 층.

왜 따로 두는가
------------
    - 화면 코드는 '보여주기'만 한다. 검증·쉬운 설명·역검증의 순서와 fail-closed 처리는 여기서 정한다.
    - Streamlit 없이 테스트할 수 있다 (scripts/68_test_v2_service.py).
    - 평가 스크립트(60·67번)와 같은 구성(cv-v1 · sp-v2 · bv-v1)을 그대로 쓴다. 화면이 평가와 다른 설정으로
      돌면 공식 수치가 화면의 성능을 말해 주지 못한다.

LLM 호출 상한
-----------
    데모 키는 월 사용 한도가 있다. 세션마다 호출 상한을 두고, 상한에 닿으면 새 API 호출을 하지 않는다.
    이때 CachedLLM 은 None 을 돌려주고, 각 단계는 원래 설계된 fail-closed 경로로 간다.
        claim 추출 실패   → 초안 전체를 claim 1개로 두고 VERIFIER_ERROR (CHECK)
        쉬운 설명 실패    → SIMPLIFIER_ERROR → BLOCK (고객용 문구 없음)
        조건 검사 실패    → BV_ERROR → BLOCK
    즉 호출 상한은 '틀린 안내'가 아니라 '안내 보류'로 이어진다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .back_verify import BackVerifier, SimplifyOutcome, run_simplify
from .claim_verifier import CachedLLM, ClaimVerifier, EvidenceChunk, VerifyResult
from .simplify import Simplifier

MODEL = "gpt-4.1-mini"


class BudgetedLLM:
    """CachedLLM 을 감싸 이번 실행에서 쓸 수 있는 API 호출 수를 제한한다. 캐시 적중은 세지 않는다."""

    def __init__(self, inner: CachedLLM, max_calls: int):
        self.inner, self.max_calls = inner, max(0, max_calls)
        self.used = 0
        self.blocked = 0          # 상한 때문에 하지 못한 호출 수

    def json(self, system: str, user: str, max_tokens: int = 1500, tag: str = ""):
        before = self.inner.calls
        if self.used >= self.max_calls:
            client, self.inner.client = self.inner.client, None   # 캐시만 본다
            try:
                data = self.inner.json(system, user, max_tokens=max_tokens, tag=tag)
            finally:
                self.inner.client = client
            if data is None:
                self.blocked += 1
            return data
        data = self.inner.json(system, user, max_tokens=max_tokens, tag=tag)
        self.used += self.inner.calls - before
        return data

    # CachedLLM 과 같은 이름의 통계 (화면 표시용)
    @property
    def calls(self) -> int:
        return self.used

    def flush(self) -> None:
        self.inner.flush()


@dataclass
class V2Result:
    question: str
    draft: str
    verify: VerifyResult
    simplify: SimplifyOutcome
    api_calls: int = 0
    blocked_calls: int = 0
    notes: list[str] = field(default_factory=list)


def chunks_from_evidences(evidences: list[dict]) -> list[EvidenceChunk]:
    """v1 Answer 의 근거(dict) → EvidenceChunk. 평가 스크립트와 같은 필드를 쓴다 (doc_name = citation)."""
    out = []
    for e in evidences or []:
        if not e.get("chunk_id"):
            continue
        out.append(EvidenceChunk(str(e["chunk_id"]), e.get("text", "") or "", e.get("citation", "") or "",
                                 e.get("doc_type", "") or "", bool(e.get("is_latest", True))))
    return out


class V2Service:
    def __init__(self, root: Path, client=None, cache_path: Path | None = None,
                 matcher=None, known_products: list[str] | None = None):
        self.root = Path(root)
        self.cache = CachedLLM(client, MODEL, cache_path or self.root / "data" / "interim" / "v2_ui_cache.json")
        self.matcher, self.known = matcher, list(known_products or [])

    @classmethod
    def build(cls, root: Path, client=None, cache_path: Path | None = None) -> "V2Service":
        from ..generation.product_catalog import ProductCatalog
        from ..generation.product_match import ProductMatcher
        catalog = Path(root) / "data" / "interim" / "product_catalog.json"
        matcher = ProductMatcher.from_catalog(catalog)
        known = [e.canonical_name for e in ProductCatalog.load(catalog).all_named if e.canonical_name] \
            if catalog.exists() else []
        return cls(root, client, cache_path, matcher, known)

    @property
    def online(self) -> bool:
        return self.cache.client is not None

    def run(self, question: str, draft: str, evidences: list[dict], max_calls: int) -> V2Result:
        """초안 검증(A) → 쉬운 설명(B) → 역검증(B, 1회 수정 포함)."""
        llm = BudgetedLLM(self.cache, max_calls)
        notes: list[str] = []
        chunks = chunks_from_evidences(evidences)
        if not chunks:
            notes.append("근거 문서가 없습니다. 모든 claim 이 '근거 없음(확인 필요)'으로 나옵니다.")
        cv = ClaimVerifier(llm, self.matcher, known_products=self.known or None, prompt_version="cv-v1")
        vr = cv.verify(question, draft, chunks)
        outcome = run_simplify(question, vr.claims, vr.verifications, Simplifier(llm),
                               BackVerifier(llm, self.matcher, self.known, condition_check=True))
        if llm.blocked:
            notes.append(f"LLM 호출 상한에 닿아 {llm.blocked}개 단계를 실행하지 못했습니다. "
                         "해당 단계는 안전한 쪽(확인 필요·안내 보류)으로 처리했습니다.")
        if vr.errors:
            notes.append("검증 단계 경고: " + ", ".join(vr.errors[:4]))
        try:
            llm.flush()
        except OSError:
            notes.append("LLM 캐시를 저장하지 못했습니다 (읽기 전용 환경일 수 있음).")
        return V2Result(question, draft, vr, outcome, llm.used, llm.blocked, notes)


__all__ = ["V2Service", "V2Result", "BudgetedLLM", "chunks_from_evidences"]
