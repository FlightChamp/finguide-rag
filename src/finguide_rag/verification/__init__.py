"""FinGuide-RAG v2 검증 계층.

    schema   노드 사이 JSON 계약 (로직 없음)
    policy   사유 코드 → 판정 도출 규칙 (PolicyGate, LLM 없음)
    numeric  수치 정규화·대조 규칙 (LLM 없음)
"""

from .policy import aggregate_final, build_verification, derive_verdict, next_bv_status
from .schema import (
    REASON_SPECS,
    SCHEMA_VERSION,
    BVStatus,
    FinalLabel,
    ReasonCode,
    ReasonHit,
    Source,
    Verdict,
)

__all__ = [
    "SCHEMA_VERSION", "REASON_SPECS", "BVStatus", "FinalLabel", "ReasonCode",
    "ReasonHit", "Source", "Verdict", "aggregate_final", "build_verification",
    "derive_verdict", "next_bv_status",
]
