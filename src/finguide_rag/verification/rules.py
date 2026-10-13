"""claim 규칙 검사 — 단정·헤지, 상품, 기관, 고객정보, 시점, 공란, 구버전.

numeric.py 와 같은 원칙: LLM 없음, 같은 입력이면 같은 결과.
LLM 이 '근거와 맞다(SUPPORTED)'고 해도 여기서 문제가 나오면 규칙이 이긴다
(policy.detect_conflict 가 그 충돌을 기록한다).

검사 항목과 발급 코드
-----------------
    check_assertiveness   강조어 추가 / 헤지 소실          → OVERSTATED
    check_institution     근거와 다른 기관명                → INSTITUTION_MISMATCH
    check_product         근거 설명서와 다른 상품           → PRODUCT_MISMATCH
                          카탈로그상 모호한 상품명           → AMBIGUOUS_PRODUCT
    check_customer_data   이 고객에게 적용된다고 확정       → NEEDS_CUSTOMER_DATA
    check_time_variant    현재·오늘 등 시점 의존 표현        → TIME_VARIANT   (v1 패턴 재사용)
    check_blank_value     근거가 공란인데 수치를 확정       → BLANK_VALUE    (v1 패턴 재사용)
    check_outdated        근거가 최신 시행본이 아님          → OUTDATED_SOURCE

입력 근거는 '정렬된 원문 span'이어야 한다
-------------------------------------
단정·헤지 비교는 근거 문장에 있는 표현과 claim 을 비교한다. 청크 전체를
넣으면 claim 과 무관한 문장의 '수 있', '에 한해' 때문에 헤지 소실이
대량으로 오탐된다. 51번 사전 점검에서 정상 답변 78문장 중 68%가 걸렸다
(근거=청크 전체). 반대로 청크에 강조어가 우연히 있으면 강조어 추가는 놓친다.

헤지 소실은 '가능성'·'한정' 묶음만 본다
-----------------------------------
'다만 …은 제외', '…조건 충족 시' 같은 단서·조건은 보통 별도 문장·별도 절이다.
claim 하나가 그것을 담지 않는 것은 정상 요약일 수 있고, 무엇이 빠졌는지는
의미 판단이므로 LLM 의 MISSING_CONDITION(claim 내부)과
OMITTED_REQUIRED_INFO(초안 전체)가 맡는다. 규칙은 같은 명제의 양태가
바뀐 경우('발생할 수 있습니다' → '발생합니다', '…에 한해 적용' → '적용')만 잡는다.

상품 대조를 설명서에만 적용하는 이유
-------------------------------
v1 상품불일치 20건 수동 재분류(data/eval/mismatch_review.md)에서 16건이
평가기 한계였고, 그중 4건이 '약관이라 문서명에 상품명이 없음', 12건이
'FAQ·서비스명 추출 오류'였다. 약관·FAQ 문서명으로 상품을 대조하면 같은
오탐을 반복한다. 그래서 이름 대조는 상품설명서 근거에만 하고, 약관·FAQ
근거만 있으면 'n/a(이름 대조 불가)'로 남긴다. 이 공백은 LLM 정렬 단계의
CONTRADICTED/NO_EVIDENCE 가 메운다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..generation.product_match import ProductMatcher
from ..generation.refusal import RE_BLANK, RE_TIME_VARIANT
from .numeric import extract_numerics
from .schema import CatalogStatus, EvidenceRef, ReasonCode, ReasonHit, Source

# ==================================================================
# 1. 단정 · 헤지
# ==================================================================

# 원문보다 강하게 말하는 표현. 근거에 같은 표현이 있으면 문제 아님.
INTENSIFIERS: tuple[str, ...] = (
    "무조건", "항상", "반드시", "절대", "누구나", "누구든지", "100%", "100퍼센트",
    "전액", "언제든지", "언제든", "확실히", "틀림없이", "보장", "그대로",
)
# 원문이 같은 강도를 다른 말로 표현한 경우. 근거에 이 중 하나라도 있으면 강조어 추가가 아니다.
#   근거 '신분증을 지참하여야 합니다' → claim '반드시 지참' 은 정상
INTENSIFIER_EQUIV: dict[str, tuple[str, ...]] = {
    "반드시": ("반드시", "꼭", "필수", "하여야", "해야", "의무"),
    "항상": ("항상", "언제나", "매번"),
    "누구나": ("누구나", "누구든지", "모든 고객", "제한 없이"),
    "언제든지": ("언제든지", "언제든", "24시간", "365일"),
    "언제든": ("언제든지", "언제든", "24시간", "365일"),
    "전액": ("전액", "전부", "모두"),
}

# 헤지 표현을 '무엇을 한정하는가'로 묶는다. 묶음 단위로 소실을 본다.
HEDGE_GROUPS: dict[str, re.Pattern] = {
    # '할 수 있습니다'(능력·방법 안내)는 헤지가 아니다. FAQ 는 '등록할 수 있습니다'를
    # 흔히 쓰고 직원은 '등록하시면 됩니다'라고 말한다. 51번 사전 점검에서 이 오탐이
    # 경고의 대부분이었다. 그래서 '발생·변동·제한' 같은 사건 동사의 가능성만 본다.
    "가능성": re.compile(
        r"(될|생길|발생할|있을|달라질|변동될|변경될|부과될|제한될|않을|감소할|지연될"
        r"|하락할|상승할|거절될|취소될|해지될|손실이\s*날)\s*수\s*(도\s*)?있"
        r"|수도\s*있|경우에\s*따라|원칙적으로|일반적으로"),
    "한정": re.compile(r"에\s*한(하|해|함|하여)|경우에만|에\s*한정|만\s*해당"),
}
# 아래 두 묶음은 assertiveness_of 에서 'hedged' 판단에만 쓰고 소실 판정에는 쓰지 않는다.
_SOFT_HEDGES = re.compile(r"다만|단,|제외|예외|충족\s*(시|하는|한)|조건|요건")
# claim 쪽은 너그럽게 본다: '영향을 줄 수 있습니다'처럼 어떤 '수 있'이든 헤지로 인정.
# 근거 쪽은 엄격하게(사건 동사만), claim 쪽은 너그럽게 — 오탐을 줄이는 방향의 비대칭이다.
_CLAIM_ANY_POSSIBILITY = re.compile(r"수\s*(도\s*)?있")


def _intensifiers(text: str) -> set[str]:
    return {w for w in INTENSIFIERS if w in text}


def _added_intensifiers(claim_text: str, evidence_text: str) -> list[str]:
    out = []
    for w in _intensifiers(claim_text):
        equiv = INTENSIFIER_EQUIV.get(w, (w,))
        if not any(e in evidence_text for e in equiv):
            out.append(w)
    return sorted(out)


def _hedges(text: str) -> set[str]:
    return {g for g, pat in HEDGE_GROUPS.items() if pat.search(text)}


def assertiveness_of(text: str) -> str:
    """Claim.assertiveness 를 규칙으로 채운다 (LLM 에 맡기지 않는다)."""
    if _intensifiers(text):
        return "strong"
    if _hedges(text) or _SOFT_HEDGES.search(text) or _CLAIM_ANY_POSSIBILITY.search(text):
        return "hedged"
    return "neutral"


def check_assertiveness(claim_text: str, evidence_text: str) -> tuple[list[ReasonHit], str]:
    """원문 대비 강조어 추가, 헤지 소실을 본다.

    헤지 소실은 '근거에는 한정 표현 묶음이 있는데 claim 에는 어떤 헤지도 없을 때'만
    잡는다. claim 이 다른 방식으로라도 한정하고 있으면 넘어간다 — 표현을
    바꿔 말한 정상 요약을 막지 않기 위해서다. 조건이 '무엇'인지의 보존은
    LLM 의 MISSING_CONDITION 이 본다.
    """
    hits: list[ReasonHit] = []
    added = _added_intensifiers(claim_text, evidence_text)
    if added:
        hits.append(ReasonHit(ReasonCode.OVERSTATED, Source.RULE,
                              f"원문에 없는 강조어: {added}"))
    ev_h = _hedges(evidence_text)
    cl_hedged = bool(_hedges(claim_text) or _SOFT_HEDGES.search(claim_text)
                     or _CLAIM_ANY_POSSIBILITY.search(claim_text))
    if ev_h and not cl_hedged:
        hits.append(ReasonHit(ReasonCode.OVERSTATED, Source.RULE,
                              f"원문의 한정 표현 소실: {sorted(ev_h)}"))
    return hits, ("fail" if hits else "pass")


# ==================================================================
# 2. 기관
# ==================================================================

# 표기 변형 → 정규 기관명. 긴 표기를 먼저 본다('KB국민은행' 이 '국민' 보다 먼저).
_INSTITUTION_ALIASES: dict[str, tuple[str, ...]] = {
    "하나은행": ("KEB하나은행", "하나은행"),
    "국민은행": ("KB국민은행", "국민은행", "KB은행"),
    "신한은행": ("신한은행",),
    "우리은행": ("우리은행",),
    "농협은행": ("NH농협은행", "농협은행", "NH농협", "지역농협", "단위농협"),
    "기업은행": ("IBK기업은행", "기업은행", "IBK"),
    "SC제일은행": ("SC제일은행", "제일은행"),
    "씨티은행": ("한국씨티은행", "씨티은행"),
    "카카오뱅크": ("카카오뱅크", "카뱅"),
    "토스뱅크": ("토스뱅크",),
    "케이뱅크": ("케이뱅크", "K뱅크"),
    "iM뱅크": ("iM뱅크", "대구은행"),
    "부산은행": ("BNK부산은행", "부산은행"),
    "경남은행": ("BNK경남은행", "경남은행"),
    "광주은행": ("광주은행",),
    "전북은행": ("전북은행",),
    "제주은행": ("제주은행",),
    "수협은행": ("Sh수협은행", "수협은행"),
    "산업은행": ("KDB산업은행", "산업은행"),
    "우체국": ("우체국",),
    "새마을금고": ("새마을금고", "MG새마을금고"),
    "신협": ("신협", "신용협동조합"),
}
_ALIAS_TABLE = sorted(((a, c) for c, al in _INSTITUTION_ALIASES.items() for a in al),
                      key=lambda x: len(x[0]), reverse=True)


def detect_institutions(text: str) -> set[str]:
    """텍스트에 언급된 금융기관(정규명) 집합."""
    found: set[str] = set()
    masked = text
    for alias, canon in _ALIAS_TABLE:
        if alias in masked:
            found.add(canon)
            masked = masked.replace(alias, " " * len(alias))   # 짧은 별칭 중복 매칭 방지
    return found


def check_institution(claim_text: str, evidence: Iterable[EvidenceRef]) -> tuple[list[ReasonHit], str]:
    claimed = detect_institutions(claim_text)
    if not claimed:
        return [], "n/a"
    ev_inst = {e.institution for e in evidence if e.institution}
    foreign = sorted(claimed - ev_inst)
    if foreign:
        return [ReasonHit(ReasonCode.INSTITUTION_MISMATCH, Source.RULE,
                          f"claim 기관 {foreign} ↔ 근거 기관 {sorted(ev_inst)}")], "fail"
    return [], "pass"


# ==================================================================
# 3. 상품
# ==================================================================

_NAME_COMPARABLE_DOC_TYPES = {"설명서"}


def check_product(claim_product: str | None, evidence: Iterable[EvidenceRef],
                  matcher: ProductMatcher,
                  catalog_status: CatalogStatus | None = None) -> tuple[list[ReasonHit], str]:
    """claim 의 상품명이 근거 설명서의 상품과 같은지.

    claim_product 는 Claim.product 슬롯 값(LLM 추출)이다. 판정은 규칙이 한다.
    """
    if catalog_status == CatalogStatus.AMBIGUOUS:
        return [ReasonHit(ReasonCode.AMBIGUOUS_PRODUCT, Source.RULE,
                          f"상품명 '{claim_product}' 이 여러 상품과 겹침")], "fail"
    if not claim_product or not matcher.informative(claim_product):
        return [], "n/a"
    comparable = [e for e in evidence if e.doc_type in _NAME_COMPARABLE_DOC_TYPES and e.doc_name]
    if not comparable:
        return [], "n/a"   # 약관·FAQ 근거만 있음 → 이름 대조 불가 (모듈 docstring)
    if any(matcher.same_product(claim_product, e.doc_name) for e in comparable):
        return [], "pass"
    names = sorted({matcher.product_of(e.doc_name) for e in comparable})
    return [ReasonHit(ReasonCode.PRODUCT_MISMATCH, Source.RULE,
                      f"claim 상품 '{claim_product}' ↔ 근거 설명서 {names}")], "fail"


# ==================================================================
# 4. 고객정보 · 시점 · 공란 · 구버전
# ==================================================================

# v1 거절 0단계는 '질문' 문면을 봤다. 직원 초안에서는 '이 고객에게 적용된다'고
# 확정하는 말투가 신호다. 일반 규정을 존댓말로 설명하는 문장은 잡지 않는다.
#   잡음:  "고객님은 우대금리를 받으실 수 있습니다", "고객님 신용등급이면 가능합니다"
#   안 잡음: "중도해지 시에는 중도해지이율이 적용됩니다"
RE_CUSTOMER_ATTR = re.compile(
    r"(고객님|손님|회원님)(의|께서|은|는|이|도)?\s*(계좌|잔액|신용|등급|거래\s*실적|실적|소득"
    r"|연체|한도|대출|보유|나이|연령|가입\s*이력|조건)"
)
RE_PERSONAL_OUTCOME = re.compile(
    r"(고객님|손님|회원님)(은|는|께서|도|의\s*경우)?\s*\S*\s*"
    r"(받으실\s*수\s*있|받으실\s*거|대상(이|자)\s*(시|이십|입니다|에\s*해당)"
    r"|해당(되|하)(십|세요|시니)|가능하(십|세요|시니)|적용(되십|받으실))"
)


def check_customer_data(claim_text: str) -> tuple[list[ReasonHit], bool]:
    m = RE_CUSTOMER_ATTR.search(claim_text) or RE_PERSONAL_OUTCOME.search(claim_text)
    if m:
        return [ReasonHit(ReasonCode.NEEDS_CUSTOMER_DATA, Source.RULE,
                          f"고객별 확인이 필요한 확정 표현: '{m.group(0).strip()}'")], True
    return [], False


# 문서상 기간 조건이지 실시간 시점이 아닌 표현: '최근 3개월 이내 발급본'
_DOC_PERIOD = re.compile(r"최근\s*\d+\s*(개월|일|년|주)")


def check_time_variant(claim_text: str, evidence_text: str = "") -> list[ReasonHit]:
    """직원이 '현재·오늘·실시간' 값을 확정해 말하는지.

    v1 패턴(질문용)을 claim 에 쓰면서 두 가지를 뺐다 (51번 dev 점검 오탐 5건 근거).
      - 문서상 기간: '최근 3개월 이내 발급'
      - 근거 원문에 같은 표현이 이미 있는 경우: 원문이 '현재 6개국', '실시간 매입율'
        이라고 쓴 것을 옮긴 것은 직원이 시점 값을 만든 것이 아니다.
        (원문 자체가 낡았을 가능성은 OUTDATED_SOURCE 가 시행일로 본다)
    """
    text = _DOC_PERIOD.sub(" ", claim_text)
    for m in RE_TIME_VARIANT.finditer(text):
        if m.group(0) not in evidence_text:
            return [ReasonHit(ReasonCode.TIME_VARIANT, Source.RULE,
                              f"시점 의존 표현: '{m.group(0)}'")]
    return []


def check_blank_value(claim_text: str, evidence: Iterable[EvidenceRef]) -> list[ReasonHit]:
    """근거 원문이 공란(예: '연 ( )%')인데 claim 이 금리·금액을 확정해서 말함."""
    if not any(RE_BLANK.search(e.quote) for e in evidence):
        return []
    rate_or_money = [s for s in extract_numerics(claim_text) if s.unit in ("%", "%p", "원")]
    if rate_or_money:
        return [ReasonHit(ReasonCode.BLANK_VALUE, Source.RULE,
                          f"근거는 공란인데 수치 확정: {[s.raw for s in rate_or_money]}")]
    return []


def check_outdated(evidence: Iterable[EvidenceRef]) -> list[ReasonHit]:
    old = [e.doc_id for e in evidence if not e.is_latest]
    if old:
        return [ReasonHit(ReasonCode.OUTDATED_SOURCE, Source.RULE, f"구버전 근거: {old}")]
    return []


# ==================================================================
# 5. 묶음 실행 — ClaimVerifier 가 호출하는 진입점
# ==================================================================


@dataclass
class RuleResult:
    hits: list[ReasonHit] = field(default_factory=list)
    rule_checks: dict[str, str] = field(default_factory=dict)
    assertiveness: str = "neutral"
    needs_customer_data: bool = False


def run_claim_rules(claim_text: str, evidence: list[EvidenceRef], *,
                    claim_product: str | None, matcher: ProductMatcher,
                    catalog_status: CatalogStatus | None = None) -> RuleResult:
    """claim 하나에 대한 모든 규칙 검사.

    수치 대조는 numeric.numeric_reason_hits 를 쓴다. 근거 원문은 정렬된 span 들을
    이어 붙여 한 번에 본다.
    """
    from .numeric import numeric_reason_hits   # 순환 import 방지용 지역 import

    ev_text = "\n".join(e.quote for e in evidence)
    r = RuleResult(assertiveness=assertiveness_of(claim_text))

    h, r.rule_checks["numeric"] = numeric_reason_hits(claim_text, ev_text)
    r.hits += h
    h, r.rule_checks["assertiveness"] = check_assertiveness(claim_text, ev_text)
    r.hits += h
    h, r.rule_checks["institution"] = check_institution(claim_text, evidence)
    r.hits += h
    h, r.rule_checks["product"] = check_product(claim_product, evidence, matcher, catalog_status)
    r.hits += h
    h, r.needs_customer_data = check_customer_data(claim_text)
    r.hits += h
    r.rule_checks["customer_data"] = "fail" if h else "pass"
    for name, fn in (("time_variant", lambda: check_time_variant(claim_text, ev_text)),
                     ("blank_value", lambda: check_blank_value(claim_text, evidence)),
                     ("outdated", lambda: check_outdated(evidence))):
        h = fn()
        r.hits += h
        r.rule_checks[name] = "fail" if h else "pass"
    return r
