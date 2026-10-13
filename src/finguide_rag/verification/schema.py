"""FinGuide-RAG v2 검증 계층의 데이터 계약.

이 파일은 로직을 담지 않는다. 노드(Router → Retriever → ClaimVerifier →
PolicyGate → Simplifier → BackVerifier → PolicyGate) 사이를 오가는 값의
'모양'만 고정한다. 판정 규칙은 policy.py, 수치 대조는 numeric.py 에 있다.

설계 원칙
--------
1. 추출은 LLM, 판정은 코드
   LLM 은 사유 코드(ReasonCode) 중 일부만 낼 수 있고, OK/CHECK/PROHIBITED
   라벨은 어디서도 LLM 이 직접 쓰지 않는다. 라벨은 policy.derive_verdict 가
   사유 코드의 심각도에서 도출한다. v1 에서 LLM 근거 검증이 "다른 상품이면
   거절" 지시를 어기고 통과시킨 실패(주택청약예금 ↔ 정기예금)를 구조로 막는다.

2. 사유 코드마다 '누가 발급할 수 있는가'를 기록한다
   REASON_SPECS[code].sources 에 없는 주체가 낸 코드는 policy 계층에서
   VERIFIER_ERROR 로 바뀐다(fail-closed).

3. 평가 정답은 chunk_id 가 아니라 doc_id + 원문 인용으로 참조한다
   재색인 때 청크 ID 가 바뀌어도 정답이 살아남게 하기 위해서다
   (v1 재매핑 편향 0.708 ↔ 0.781 의 재발 방지). EvidenceRef 가 두 값을
   모두 갖는 이유다.

4. 직렬화는 표준 라이브러리만 쓴다
   런타임 의존성을 늘리지 않는다. to_dict() 는 Enum 을 value 로 바꾼다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "v2.0-draft1"


# ==================================================================
# 열거형
# ==================================================================


class Mode(str, Enum):
    """Router 가 정하는 실행 모드.

    simplify 는 모드가 아니다. 쉬운 설명은 verify 결과 위에서만 실행되는
    후속 단계이므로, 모드로 두면 검증을 건너뛰는 경로가 생긴다.
    """

    ANSWER = "answer"      # v1 경로: 질문 → 근거 기반 답변
    VERIFY = "verify"      # v2 경로: 질문 + 직원 초안 → claim 검증


class QuestionType(str, Enum):
    PRODUCT = "product_specific"
    CONCEPT = "general_concept"
    POLICY = "policy"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


class CatalogStatus(str, Enum):
    """v1 catalog_matcher 의 판정을 그대로 쓴다."""

    COVERED = "covered"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"
    NONE = "none"


class Verdict(str, Enum):
    """claim 단위 판정. 심각도 순서: OK < CHECK < PROHIBITED."""

    OK = "OK"                  # 안내 가능
    CHECK = "CHECK"            # 확인 필요
    PROHIBITED = "PROHIBITED"  # 안내 금지

    @property
    def severity(self) -> int:
        return _SEVERITY[self]

    @property
    def label_ko(self) -> str:
        return {"OK": "안내 가능", "CHECK": "확인 필요",
                "PROHIBITED": "안내 금지"}[self.value]


_SEVERITY = {Verdict.OK: 0, Verdict.CHECK: 1, Verdict.PROHIBITED: 2}


class BVStatus(str, Enum):
    """역검증 상태."""

    PASS = "PASS"
    NEEDS_FIX = "NEEDS_FIX"
    BLOCK = "BLOCK"


class FinalLabel(str, Enum):
    """초안 전체에 대한 최종 판정."""

    OK = "OK"                  # 모든 claim 안내 가능
    PARTIAL = "PARTIAL"        # 일부 안내 가능, 나머지 확인 필요
    CHECK = "CHECK"            # 안내 가능한 claim 이 없음
    PROHIBITED = "PROHIBITED"  # 안내 금지 claim 포함 — 수정 전 안내 불가


class Source(str, Enum):
    """사유 코드 발급 주체."""

    LLM = "llm"
    RULE = "rule"
    SYSTEM = "system"


class Level(str, Enum):
    CLAIM = "claim"
    DRAFT = "draft"


class ReasonCode(str, Enum):
    SUPPORTED = "SUPPORTED"
    NO_EVIDENCE = "NO_EVIDENCE"
    MISSING_CONDITION = "MISSING_CONDITION"
    OVERSTATED = "OVERSTATED"
    NEEDS_CUSTOMER_DATA = "NEEDS_CUSTOMER_DATA"
    TIME_VARIANT = "TIME_VARIANT"
    BLANK_VALUE = "BLANK_VALUE"
    AMBIGUOUS_PRODUCT = "AMBIGUOUS_PRODUCT"
    OUTDATED_SOURCE = "OUTDATED_SOURCE"
    VERIFIER_ERROR = "VERIFIER_ERROR"
    NUMERIC_MISMATCH = "NUMERIC_MISMATCH"
    PRODUCT_MISMATCH = "PRODUCT_MISMATCH"
    INSTITUTION_MISMATCH = "INSTITUTION_MISMATCH"
    CONTRADICTED = "CONTRADICTED"
    OMITTED_REQUIRED_INFO = "OMITTED_REQUIRED_INFO"


# ==================================================================
# 사유 코드 명세 — 이 표가 판정 정책의 단일 출처(single source of truth)
# ==================================================================


@dataclass(frozen=True)
class ReasonSpec:
    severity: Verdict
    sources: frozenset[Source]
    level: Level
    staff_message: str     # 직원 주의사항 문구 템플릿


_L, _R, _S = Source.LLM, Source.RULE, Source.SYSTEM

REASON_SPECS: dict[ReasonCode, ReasonSpec] = {
    ReasonCode.SUPPORTED: ReasonSpec(
        Verdict.OK, frozenset({_L}), Level.CLAIM,
        "원문 근거와 일치합니다."),
    ReasonCode.MISSING_CONDITION: ReasonSpec(
        Verdict.CHECK, frozenset({_L, _R}), Level.CLAIM,
        "원문에 있는 조건·한정어가 빠졌습니다. 조건을 함께 안내하세요."),
    ReasonCode.OVERSTATED: ReasonSpec(
        Verdict.CHECK, frozenset({_R}), Level.CLAIM,
        "원문보다 단정적인 표현입니다. '최고', '~할 수 있습니다' 등 원문 표현을 유지하세요."),
    ReasonCode.NO_EVIDENCE: ReasonSpec(
        Verdict.CHECK, frozenset({_L, _R}), Level.CLAIM,
        "근거 문서에서 이 내용을 찾지 못했습니다. 원문 확인 전 안내하지 마세요."),
    ReasonCode.NEEDS_CUSTOMER_DATA: ReasonSpec(
        Verdict.CHECK, frozenset({_R}), Level.CLAIM,
        "고객별 정보(거래실적·신용·잔액 등) 조회가 필요한 내용입니다."),
    ReasonCode.TIME_VARIANT: ReasonSpec(
        Verdict.CHECK, frozenset({_R}), Level.CLAIM,
        "현재 시점 정보(금리·환율 등)입니다. 고시 기준으로 다시 확인하세요."),
    ReasonCode.BLANK_VALUE: ReasonSpec(
        Verdict.CHECK, frozenset({_R}), Level.CLAIM,
        "문서상 공란·개별 약정값입니다. 계약서 기준으로 확인하세요."),
    ReasonCode.AMBIGUOUS_PRODUCT: ReasonSpec(
        Verdict.CHECK, frozenset({_R}), Level.CLAIM,
        "상품명이 여러 상품과 겹칩니다. 정확한 상품명을 확인하세요."),
    ReasonCode.OUTDATED_SOURCE: ReasonSpec(
        Verdict.CHECK, frozenset({_R}), Level.CLAIM,
        "근거가 최신 시행본이 아닙니다. 최신 문서로 확인하세요."),
    ReasonCode.VERIFIER_ERROR: ReasonSpec(
        Verdict.CHECK, frozenset({_S}), Level.CLAIM,
        "자동 검증이 완료되지 않았습니다. 원문을 직접 확인하세요."),
    ReasonCode.NUMERIC_MISMATCH: ReasonSpec(
        Verdict.PROHIBITED, frozenset({_R}), Level.CLAIM,
        "수치·기간·단위가 원문과 다릅니다."),
    ReasonCode.PRODUCT_MISMATCH: ReasonSpec(
        Verdict.PROHIBITED, frozenset({_R}), Level.CLAIM,
        "다른 상품의 기준으로 안내하고 있습니다."),
    ReasonCode.INSTITUTION_MISMATCH: ReasonSpec(
        Verdict.PROHIBITED, frozenset({_R}), Level.CLAIM,
        "다른 기관의 기준으로 안내하고 있습니다."),
    ReasonCode.CONTRADICTED: ReasonSpec(
        Verdict.PROHIBITED, frozenset({_L}), Level.CLAIM,
        "원문 근거와 반대되는 내용입니다."),
    ReasonCode.OMITTED_REQUIRED_INFO: ReasonSpec(
        Verdict.CHECK, frozenset({_L}), Level.DRAFT,
        "질문에 답하려면 필요한 원문 조건이 초안에 없습니다."),
}

# LLM 이 발급할 수 있는 코드 전체 (claim 수준 + 초안 수준)
LLM_EMITTABLE: frozenset[ReasonCode] = frozenset(
    c for c, s in REASON_SPECS.items() if Source.LLM in s.sources
)
# claim 검증 프롬프트의 출력 enum 은 반드시 이 집합으로 제한한다.
LLM_CLAIM_EMITTABLE: frozenset[ReasonCode] = frozenset(
    c for c in LLM_EMITTABLE if REASON_SPECS[c].level == Level.CLAIM
)


# ==================================================================
# 노드 입출력
# ==================================================================


def _plain(obj: Any) -> Any:
    """dataclass/Enum 을 JSON 직렬화 가능한 값으로."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {(_plain(k)): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_plain(v) for v in obj]
    return obj


class _Serializable:
    def to_dict(self) -> dict:
        return _plain(asdict(self))  # type: ignore[arg-type]


@dataclass
class RouterOut(_Serializable):
    query_id: str
    mode: Mode
    question_type: QuestionType
    institution: str | None
    product_mentions: list[str]
    catalog_status: CatalogStatus
    pre_flags: list[str] = field(default_factory=list)   # personalized / time_variant / blank_value
    corpus_scope: list[str] = field(default_factory=list)


@dataclass
class NumericSlot(_Serializable):
    """수치 하나를 '의미 단위'로 표현한다.

    값만 같다고 같은 수치가 아니다. 금융 문서에서 위험한 오류는
    0.3% ↔ 0.3%p, 1천만원 이상 ↔ 1천만원 초과, 최고 연 4.0% ↔ 연 4.0% 처럼
    값은 같고 의미가 다른 경우다.
    """

    value: float
    unit: str                     # "%", "%p", "원", "개월", "일", "영업일", "회", "세", "date"
    value_hi: float | None = None  # 범위(12~36개월)의 상한
    comparator: str | None = None  # ">=", ">", "<=", "<"
    qualifier: str | None = None   # 최고/최대/최저/최소/기본/우대/한도
    period: str | None = None      # 금리의 기준 기간: "연" / "월"
    raw: str = ""
    span: tuple[int, int] = (0, 0)

    @property
    def is_range(self) -> bool:
        return self.value_hi is not None


@dataclass
class Claim(_Serializable):
    claim_id: str
    text: str
    span: tuple[int, int]          # 초안 내 문자 위치 — 평가 정렬용
    product: str | None = None
    institution: str | None = None
    event: str | None = None       # 중도해지, 만기, 가입 등
    numerics: list[NumericSlot] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    assertiveness: str = "neutral"  # hedged | neutral | strong
    checkable: bool = True
    needs_customer_data: bool = False


@dataclass
class EvidenceRef(_Serializable):
    doc_id: str
    chunk_id: str
    quote: str                     # 원문 그대로. LLM 을 거치지 않는다.
    is_latest: bool = True


@dataclass
class ReasonHit(_Serializable):
    code: ReasonCode
    source: Source
    detail: str = ""

    @property
    def severity(self) -> Verdict:
        return REASON_SPECS[self.code].severity


@dataclass
class Verification(_Serializable):
    claim_id: str
    verdict: Verdict
    reasons: list[ReasonHit]
    evidence: list[EvidenceRef] = field(default_factory=list)
    rule_checks: dict[str, str] = field(default_factory=dict)  # {"numeric": "pass|fail|n/a", ...}
    llm_rule_conflict: bool = False
    fix_suggestion: str | None = None


@dataclass
class BackVerification(_Serializable):
    status: BVStatus
    dropped_conditions: list[str] = field(default_factory=list)
    numeric_diff: list[str] = field(default_factory=list)
    strengthened_phrases: list[str] = field(default_factory=list)
    lost_check_items: list[str] = field(default_factory=list)
    revised_text: str | None = None
    attempts: int = 0


@dataclass
class FinalDecision(_Serializable):
    final: FinalLabel
    claims_summary: dict[str, int]
    draft_level_reasons: list[ReasonHit] = field(default_factory=list)
    staff_warnings: list[str] = field(default_factory=list)
    audit_id: str = ""
