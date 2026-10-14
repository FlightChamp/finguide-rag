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

SCHEMA_VERSION = "v2.0-draft4"   # draft4: 쉬운 설명(Simplifier) 출력과 역검증 문제 코드(BVCode) 추가
#                                  draft3: 초안 수준 범위 코드(MIXED_SCOPE, SCOPE_MISMATCH) 추가


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
    MIXED_SCOPE = "MIXED_SCOPE"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"


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
    # 초안 수준 범위 검사 (scope.py). 규칙이 발급. 개발 셋 근거로 추가 (2026-10-14).
    ReasonCode.MIXED_SCOPE: ReasonSpec(
        Verdict.CHECK, frozenset({_R}), Level.DRAFT,
        "서로 다른 상품·서비스의 규정이 한 답변에 섞였습니다. 상품별로 나눠 안내하세요."),
    ReasonCode.SCOPE_MISMATCH: ReasonSpec(
        Verdict.CHECK, frozenset({_R}), Level.DRAFT,
        "질문과 다른(더 좁은) 상품 기준으로 답했습니다. 적용 범위를 확인하세요."),
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
    doc_name: str = ""             # 상품 대조용 표시명
    doc_type: str = ""             # 설명서 / 약관 / FAQ — 약관·FAQ 는 이름 대조 제외
    institution: str = "하나은행"   # v2 MVP 코퍼스는 단일 기관


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


# ==================================================================
# 쉬운 설명(Simplifier) · 역검증(BackVerifier)  — v2-B
# ==================================================================
#
# 원칙 (2026-10-14 확정)
#   - OK claim 만 쉬운 설명으로 바꾼다.
#   - CHECK claim 은 '직원 확인 필요' 영역에 원문 그대로 둔다 (LLM 을 거치지 않음).
#   - PROHIBITED claim 은 고객 안내문 생성 대상에서 뺀다 (직원 화면에만 표시).
#   - 역검증 PASS 전에는 고객용 쉬운 설명을 노출하지 않는다.
#   - 자동 수정은 최대 1회. 수정 후 재검증에서도 문제가 있으면 BLOCK.
#   - B 는 틀린 답을 고치는 장치가 아니다. 검증된 내용을 쉽게 말하고, 뜻이 그대로인지 다시 본다.


class BVCode(str, Enum):
    """역검증 문제 코드. ReasonCode(초안 검증)와 섞지 않는다 — 비교 대상이 다르기 때문이다.

    ReasonCode 는 '초안 claim ↔ 근거 원문', BVCode 는 '쉬운 설명 ↔ OK claim 원문'을 비교한다.
    """

    NUMERIC_CHANGED = "NUMERIC_CHANGED"        # 값·단위·기간·비교어가 달라짐
    NUMERIC_ADDED = "NUMERIC_ADDED"            # 원문에 없는 수치
    NUMERIC_DROPPED = "NUMERIC_DROPPED"        # 원문 수치가 사라짐
    PRODUCT_ADDED = "PRODUCT_ADDED"            # 원문에 없는 상품명
    PRODUCT_DROPPED = "PRODUCT_DROPPED"        # 원문 상품명이 사라짐
    INSTITUTION_ADDED = "INSTITUTION_ADDED"    # 원문·근거에 없는 기관명
    STRENGTHENED = "STRENGTHENED"              # 강조어 추가 · 상한(최고/최대) 누락
    HEDGE_LOST = "HEDGE_LOST"                  # '~될 수 있습니다' 등 가능성·한정 표현 소실
    CONDITION_DROPPED = "CONDITION_DROPPED"    # 조건·예외가 빠짐 (LLM 판단 → 코드 판정)
    CONDITION_ALTERED = "CONDITION_ALTERED"    # 조건·예외의 뜻이 바뀜 (LLM 판단 → 코드 판정)
    CONDITION_UNVERIFIED = "CONDITION_UNVERIFIED"  # LLM 이 '보존'이라 했지만 위치를 대지 못함
    CLAIM_UNCOVERED = "CLAIM_UNCOVERED"        # OK claim 이 쉬운 설명 어디에도 반영되지 않음
    CHECK_LEAK = "CHECK_LEAK"                  # CHECK claim 내용이 고객 문구에 들어감
    PROHIBITED_LEAK = "PROHIBITED_LEAK"        # PROHIBITED claim 내용이 고객 문구에 들어감
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"          # 문구가 존재하지 않는 claim id 를 근거로 댐
    LOST_CHECK = "LOST_CHECK"                  # CHECK claim 이 직원 확인 영역에서 사라짐 (코드 결함)
    SIMPLIFIER_ERROR = "SIMPLIFIER_ERROR"      # 쉬운 설명 생성 실패 · 형식 오류
    BV_ERROR = "BV_ERROR"                      # 역검증 LLM 응답 실패 · 형식 오류


@dataclass(frozen=True)
class BVSpec:
    fixable: bool          # True → NEEDS_FIX(1회 수정 기회), False → 즉시 BLOCK
    source: Source
    staff_message: str


BV_SPECS: dict[BVCode, BVSpec] = {
    BVCode.NUMERIC_CHANGED: BVSpec(True, _R, "수치·단위·기간·비교어가 원문과 다릅니다."),
    BVCode.NUMERIC_ADDED: BVSpec(True, _R, "원문에 없는 수치가 들어갔습니다."),
    BVCode.NUMERIC_DROPPED: BVSpec(True, _R, "원문 수치가 빠졌습니다."),
    BVCode.PRODUCT_ADDED: BVSpec(True, _R, "원문에 없는 상품명이 들어갔습니다."),
    BVCode.PRODUCT_DROPPED: BVSpec(True, _R, "원문 상품명이 빠졌습니다."),
    BVCode.INSTITUTION_ADDED: BVSpec(True, _R, "다른 기관명이 들어갔습니다."),
    BVCode.STRENGTHENED: BVSpec(True, _R, "원문보다 단정적인 표현입니다."),
    BVCode.HEDGE_LOST: BVSpec(True, _R, "원문의 가능성·한정 표현이 빠졌습니다."),
    BVCode.CONDITION_DROPPED: BVSpec(True, _L, "원문의 조건·예외가 빠졌습니다."),
    BVCode.CONDITION_ALTERED: BVSpec(True, _L, "원문의 조건·예외의 뜻이 바뀌었습니다."),
    BVCode.CONDITION_UNVERIFIED: BVSpec(True, _S, "조건 보존 여부를 확인하지 못했습니다."),
    BVCode.CLAIM_UNCOVERED: BVSpec(True, _R, "안내 가능 내용 일부가 쉬운 설명에서 빠졌습니다."),
    BVCode.CHECK_LEAK: BVSpec(True, _R, "확인 필요 내용이 고객 문구에 들어갔습니다."),
    BVCode.PROHIBITED_LEAK: BVSpec(True, _R, "안내 금지 내용이 고객 문구에 들어갔습니다."),
    BVCode.UNKNOWN_SOURCE: BVSpec(True, _R, "근거 claim 을 알 수 없는 문구가 있습니다."),
    BVCode.LOST_CHECK: BVSpec(False, _S, "확인 필요 항목이 누락되었습니다 (시스템 오류)."),
    BVCode.SIMPLIFIER_ERROR: BVSpec(False, _S, "쉬운 설명을 만들지 못했습니다."),
    BVCode.BV_ERROR: BVSpec(False, _S, "역검증을 완료하지 못했습니다."),
}


@dataclass
class BVProblem(_Serializable):
    code: BVCode
    detail: str = ""
    item_id: str = ""        # 문제가 난 쉬운 설명 문구 id (k1, s2 …). 전체 대상이면 ""

    @property
    def fixable(self) -> bool:
        return BV_SPECS[self.code].fixable


@dataclass
class SimpleItem(_Serializable):
    """고객용 쉬운 설명의 한 문구."""

    item_id: str                     # core: s1, s2 … / conditions: k1 … 와 겹치지 않게 n1, n2 …
    text: str
    claim_ids: list[str]             # 이 문구가 옮긴 OK claim id. 비면 출처 불명 → 역검증 문제
    condition_ids: list[str] = field(default_factory=list)   # 조건 블록 문구가 담은 조건 id


@dataclass
class RequiredCondition(_Serializable):
    """쉬운 설명이 반드시 보존해야 하는 조건 (OK claim 추출 슬롯 Claim.conditions 에서 옴).

    Simplifier 가 정하지 않는다. 쉬운 설명을 만드는 쪽이 '지켜야 할 목록'을 정하면
    빠뜨린 것도 목록에서 같이 빠지기 때문이다.
    """

    cond_id: str                     # k1, k2 …
    text: str                        # 초안 표기 그대로
    claim_id: str


@dataclass
class StaffCheckItem(_Serializable):
    """'직원 확인 필요' 영역. CHECK claim 원문 + 사유. LLM 을 거치지 않는다."""

    claim_id: str
    text: str
    reasons: list[str]


@dataclass
class SimplifiedExplanation(_Serializable):
    """쉬운 설명 3블록 + 제외 목록.

    고객에게 보일 수 있는 것은 core + conditions 뿐이며, 그것도 역검증 PASS 일 때만이다.
    staff_check · excluded 는 직원 화면 전용이다.
    """

    core: list[SimpleItem] = field(default_factory=list)          # 핵심 설명 (OK claim 만)
    conditions: list[SimpleItem] = field(default_factory=list)    # 꼭 확인할 조건·예외
    staff_check: list[StaffCheckItem] = field(default_factory=list)   # 직원 확인 필요 (CHECK)
    excluded: list[str] = field(default_factory=list)             # PROHIBITED claim id
    required_conditions: list[RequiredCondition] = field(default_factory=list)
    prompt_hash: str = ""

    def customer_items(self) -> list[SimpleItem]:
        return list(self.core) + list(self.conditions)

    def customer_text(self) -> str:
        lines = [i.text for i in self.core]
        if self.conditions:
            lines.append("꼭 확인하세요")
            lines += [f"- {i.text}" for i in self.conditions]
        return "\n".join(lines)


@dataclass
class BackVerification(_Serializable):
    status: BVStatus
    # 아래 4개는 draft1 부터 있던 요약 필드. problems 에서 채운다 (화면·감사 로그 호환)
    dropped_conditions: list[str] = field(default_factory=list)
    numeric_diff: list[str] = field(default_factory=list)
    strengthened_phrases: list[str] = field(default_factory=list)
    lost_check_items: list[str] = field(default_factory=list)
    revised_text: str | None = None
    attempts: int = 0
    # draft4
    problems: list[BVProblem] = field(default_factory=list)
    condition_results: dict[str, str] = field(default_factory=dict)   # cond_id → PRESERVED|DROPPED|ALTERED|NO_RESULT
    customer_visible: bool = False   # status == PASS 일 때만 True


@dataclass
class FinalDecision(_Serializable):
    final: FinalLabel
    claims_summary: dict[str, int]
    draft_level_reasons: list[ReasonHit] = field(default_factory=list)
    staff_warnings: list[str] = field(default_factory=list)
    audit_id: str = ""
