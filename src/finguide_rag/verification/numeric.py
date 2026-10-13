"""금융 문서 수치 정규화와 대조 — LLM 없음, 결정적 규칙.

왜 값만 비교하면 안 되는가
------------------------
금융 안내에서 위험한 수치 오류는 대부분 '값은 같고 의미가 다른' 경우다.

    0.3%      ↔ 0.3%p        퍼센트와 퍼센트포인트
    1천만원 이상 ↔ 1천만원 초과   경계값 포함 여부
    최고 연 4.0% ↔ 연 4.0%     상한이 확정값으로 바뀜 (과장)
    기본 3.0%   ↔ 최고 3.0%     기본금리와 최고금리 혼동
    연 0.5%     ↔ 월 0.5%      기준 기간

그래서 수치를 NumericSlot(값, 단위, 비교 연산자, 수식어, 기간)으로 뽑고,
대조 결과를 이유가 드러나는 상태(MatchStatus)로 돌려준다.

정규화 규칙
---------
    년 → 개월(×12), 주 → 일(×7), 살 → 세
    %포인트 / 퍼센트포인트 / %P → %p,  퍼센트 → %
    금액: 억·만·천·백·십 조합을 원 단위 정수로 (1억 5천만원 → 150,000,000)
    날짜: YYYY년 M월 D일, YYYY.MM.DD, YYYY-MM-DD → date(YYYYMMDD)
    날짜는 기간으로 오인되지 않도록 먼저 추출하고 가린다 (2026년 ≠ 2026년 기간).

알려진 한계 (테스트에 '한계' 그룹으로 명시)
-----------------------------------------
    - 한글 숫자(삼백만원, 열두 달)는 추출하지 않는다.
    - '매월 25일' 같은 날짜성 일(日)은 기간 25일로 추출된다.
      claim 과 근거가 같은 방식으로 추출되므로 대조에는 영향이 작다.
    - 대조 대상 근거는 '정렬된 원문 span'이어야 한다. 청크 전체를 넣으면
      다른 맥락의 같은 단위 수치 때문에 VALUE_MISMATCH 오탐이 늘어난다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .schema import NumericSlot, ReasonCode, ReasonHit, Source

# ------------------------------------------------------------------
# 상수
# ------------------------------------------------------------------

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_RANGE_SEP = r"\s*[~∼〜～]\s*"

# 단위 표기 → (정규 단위, 배수)
_UNIT_MAP: dict[str, tuple[str, float]] = {
    "%p": ("%p", 1), "%P": ("%p", 1), "%포인트": ("%p", 1), "% 포인트": ("%p", 1),
    "퍼센트포인트": ("%p", 1), "퍼센트 포인트": ("%p", 1),
    "%": ("%", 1), "퍼센트": ("%", 1),
    "개월": ("개월", 1), "년": ("개월", 12), "주일": ("일", 7), "주": ("일", 7),
    "영업일": ("영업일", 1), "일": ("일", 1),
    "회": ("회", 1), "세": ("세", 1), "살": ("세", 1),
}
# 긴 표기부터 매칭해야 '%포인트'가 '%'로 잘리지 않는다
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(_UNIT_MAP, key=len, reverse=True))
# 단위 뒤에 특정 글자가 붙으면 다른 단어다: 1회차, 3주년, 2년제, 3주택, 100주식
_UNIT_STOP = r"(?!차|째|년|제|택|식)"

_COMPARATOR_WORDS = [
    ("이상", ">="), ("부터", ">="), ("초과", ">"), ("넘는", ">"), ("넘게", ">"),
    ("넘으면", ">"), ("이하", "<="), ("이내", "<="), ("까지", "<="), ("미만", "<"),
]
_BOUND_QUALIFIER = {"최고": "<=", "최대": "<=", "한도": "<=", "최저": ">=", "최소": ">="}
_QUALIFIER_RE = re.compile(r"(최고|최대|한도|최저|최소|기본|우대)")
_PERIOD_RE = re.compile(r"(?<![가-힣])(연|월)(?:이율|금리|이자율|간)?(?![가-힣])")

_MONEY_BIG = {"억": 1e8, "만": 1e4}
_MONEY_SMALL = {"천": 1e3, "백": 1e2, "십": 1e1}
# 원 앞의 금액 덩어리. '원금·원리금·원칙·원본'의 원은 제외한다.
_MONEY_RE = re.compile(
    r"((?:(?<![\d,.])\d[\d,]*(?:\.\d+)?|(?<![가-힣\d])[억천백십만])[\d,.\s억천백십만]*?)\s*원(?!금|리|칙|본)"
)
_DATE_PATTERNS = [
    re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
    re.compile(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})"),
]
_YEAR_MONTH = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
_YEAR_ONLY = re.compile(r"(?<!\d)(?:19|20)\d{2}\s*년")   # 2026년부터 → 연도, 기간 아님
_COMPOSITE_YM = re.compile(rf"({_NUM})\s*년\s*({_NUM})\s*개월")


def _f(s: str) -> float:
    return float(s.replace(",", ""))


# ------------------------------------------------------------------
# 추출
# ------------------------------------------------------------------


def _context_before(text: str, start: int, prev_end: int, width: int = 14) -> str:
    """수치 앞 문맥. 이전 수치를 넘어가지 않는다 ('최고 4%, 기본 3%' 분리)."""
    lo = max(prev_end, start - width)
    return text[lo:start]


def _comparator_after(text: str, end: int) -> str | None:
    """수치 바로 뒤의 비교어. '1천만원을 넘는'처럼 조사 하나는 건너뛴다.

    조사를 먼저 지우면 '이상'의 '이'가 조사로 잘리므로, 원문 → 조사 제거
    순서로 두 번 본다.
    """
    tail = text[end:end + 8].lstrip()
    for cand in (tail, re.sub(r"^(을|를|이|가)\s*", "", tail)):
        for word, op in _COMPARATOR_WORDS:
            if cand.startswith(word):
                return op
    return None


def _qualifier_and_period(before: str) -> tuple[str | None, str | None]:
    quals = _QUALIFIER_RE.findall(before)
    # '우대금리 최고 0.3%p' 처럼 종류(우대)와 상한(최고)이 함께 올 수 있어 모두 보존
    qualifier = " ".join(dict.fromkeys(quals)) if quals else None
    periods = _PERIOD_RE.findall(before)
    period = periods[-1] if periods else None
    return qualifier, period


def _parse_money(chunk: str) -> float | None:
    tokens = re.findall(r"\d[\d,]*(?:\.\d+)?|[억천백십만]", chunk)
    if not tokens:
        return None
    total = 0.0
    section = 0.0
    pending: float | None = None
    for tok in tokens:
        if tok[0].isdigit():
            if pending is not None:
                section += pending
            pending = _f(tok)
        elif tok in _MONEY_SMALL:
            section += (pending if pending is not None else 1.0) * _MONEY_SMALL[tok]
            pending = None
        else:  # 억 / 만
            section += pending or 0.0
            pending = None
            total += (section if section else 1.0) * _MONEY_BIG[tok]
            section = 0.0
    section += pending or 0.0
    return total + section


@dataclass
class _Hit:
    start: int
    end: int
    slot: NumericSlot


def extract_numerics(text: str) -> list[NumericSlot]:
    """텍스트에서 NumericSlot 목록을 등장 순서대로 뽑는다."""
    hits: list[_Hit] = []
    masked = list(text)

    def claim_region(s: int, e: int) -> None:
        for i in range(s, e):
            masked[i] = "\x00"

    def free(s: int, e: int) -> bool:
        return all(c != "\x00" for c in masked[s:e])

    # 1) 날짜 — 기간으로 오인되지 않게 먼저
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            if not free(m.start(), m.end()):
                continue
            y, mo, d = (int(g) for g in m.groups())
            hits.append(_Hit(m.start(), m.end(), NumericSlot(
                value=float(y * 10000 + mo * 100 + d), unit="date",
                raw=m.group(0), span=(m.start(), m.end()))))
            claim_region(m.start(), m.end())
    for pat in (_YEAR_MONTH, _YEAR_ONLY):  # 2026년 3월, 2026년 → 가리기만
        for m in pat.finditer(text):
            if free(m.start(), m.end()):
                claim_region(m.start(), m.end())

    view = "".join(masked)

    # 2) 금액
    for m in _MONEY_RE.finditer(view):
        chunk = m.group(1)
        value = _parse_money(chunk)
        if value is None:
            continue
        s = m.start(1) + (len(chunk) - len(chunk.lstrip()))
        hits.append(_Hit(s, m.end(), NumericSlot(
            value=value, unit="원", raw=text[s:m.end()], span=(s, m.end()))))
        claim_region(s, m.end())
    view = "".join(masked)

    # 3) N년 M개월 복합 기간
    for m in _COMPOSITE_YM.finditer(view):
        months = _f(m.group(1)) * 12 + _f(m.group(2))
        hits.append(_Hit(m.start(), m.end(), NumericSlot(
            value=months, unit="개월", raw=m.group(0), span=(m.start(), m.end()))))
        claim_region(m.start(), m.end())
    view = "".join(masked)

    # 4) 범위: 12~36개월, 12개월~36개월, 연 2.0~3.5%
    #    구어체: 4.9%에서 6% 사이, 12개월부터 36개월까지
    #    (50번 구어체 변환에서 '4.9%~6%' → '4.9%에서 6% 사이'로 바뀌어 범위가
    #     단일 수치 2개로 쪼개진 사례가 근거. 직원은 물결표보다 이 표현을 쓴다)
    range_re = re.compile(
        rf"({_NUM})\s*(?:({_UNIT_ALT}){_UNIT_STOP})?{_RANGE_SEP}({_NUM})\s*({_UNIT_ALT}){_UNIT_STOP}")
    spoken_re = re.compile(
        rf"({_NUM})\s*(?:({_UNIT_ALT}){_UNIT_STOP})?\s*(?:에서|부터)\s*({_NUM})\s*({_UNIT_ALT}){_UNIT_STOP}"
        rf"(?=\s*(?:사이|까지))")
    for m in [*range_re.finditer(view), *spoken_re.finditer(view)]:
        if not free(m.start(), m.end()):
            continue
        unit_raw = m.group(4)
        if m.group(2) and _UNIT_MAP[m.group(2)][0] != _UNIT_MAP[unit_raw][0]:
            continue
        unit, mul = _UNIT_MAP[unit_raw]
        mul_lo = _UNIT_MAP[m.group(2)][1] if m.group(2) else mul   # 6개월~1년
        hits.append(_Hit(m.start(), m.end(), NumericSlot(
            value=_f(m.group(1)) * mul_lo, value_hi=_f(m.group(3)) * mul, unit=unit,
            raw=m.group(0), span=(m.start(), m.end()))))
        claim_region(m.start(), m.end())
    view = "".join(masked)

    # 5) 단일 수치 + 단위
    single_re = re.compile(rf"(?<![\d.,])({_NUM})\s*({_UNIT_ALT}){_UNIT_STOP}")
    for m in single_re.finditer(view):
        unit, mul = _UNIT_MAP[m.group(2)]
        hits.append(_Hit(m.start(), m.end(), NumericSlot(
            value=_f(m.group(1)) * mul, unit=unit,
            raw=m.group(0), span=(m.start(), m.end()))))
        claim_region(m.start(), m.end())

    # 6) 문맥 부여: 수식어·기간·비교어
    hits.sort(key=lambda h: h.start)
    prev_end = 0
    for h in hits:
        if h.slot.unit != "date":
            before = _context_before(text, h.start, prev_end)
            h.slot.qualifier, period = _qualifier_and_period(before)
            if h.slot.unit in ("%", "%p"):
                h.slot.period = period
            h.slot.comparator = _comparator_after(text, h.end)
        prev_end = h.end
    return [h.slot for h in hits]


# ------------------------------------------------------------------
# 대조
# ------------------------------------------------------------------


class MatchStatus(str, Enum):
    MATCH = "MATCH"
    WITHIN_RANGE = "WITHIN_RANGE"            # claim 24개월, 근거 12~36개월
    NOT_FOUND = "NOT_FOUND"                  # 근거에 같은 계열 수치 없음
    BOUND_DROPPED = "BOUND_DROPPED"          # 최고 4.0% → 4.0%
    VALUE_MISMATCH = "VALUE_MISMATCH"
    UNIT_CONFUSION = "UNIT_CONFUSION"        # % ↔ %p
    PERIOD_MISMATCH = "PERIOD_MISMATCH"      # 연 ↔ 월
    COMPARATOR_MISMATCH = "COMPARATOR_MISMATCH"  # 이상 ↔ 초과 (같은 방향, 경계 포함 여부만 다름)
    COMPARATOR_COMPLEMENT = "COMPARATOR_COMPLEMENT"  # 이상 ↔ 미만 (여집합). 아래 설명 참고
    QUALIFIER_CONFLICT = "QUALIFIER_CONFLICT"    # 기본 ↔ 최고


_FAMILY = {"%": "rate", "%p": "rate"}
_EPS = 1e-9


def _family(unit: str) -> str:
    return _FAMILY.get(unit, unit)


def _quals(s: NumericSlot) -> list[str]:
    return s.qualifier.split() if s.qualifier else []


def _bound(s: NumericSlot) -> str | None:
    if s.comparator:
        return s.comparator
    for q in _quals(s):
        if q in _BOUND_QUALIFIER:
            return _BOUND_QUALIFIER[q]
    return None


def _kind(s: NumericSlot) -> str | None:
    """금리의 '종류'. 기본금리와 최고(우대 포함)금리를 혼동하면 오안내다."""
    q = _quals(s)
    if "기본" in q:
        return "기본"
    if "우대" in q:
        return "우대"
    if "최고" in q or "최대" in q:
        return "최고"
    return None


# 서로 바꿔 말하면 안 되는 종류 쌍. 우대 ↔ 최고는 같은 맥락에서 흔히 함께 쓰여 허용.
_KIND_CONFLICTS = {frozenset({"기본", "최고"}), frozenset({"기본", "우대"})}


def _eq(a: float, b: float) -> bool:
    return abs(a - b) < _EPS


# 여집합 관계. '만 19세 이상 가능' 과 '만 19세 미만 불가' 는 같은 규정이다.
# 둘이 같은 말인지 반대 말인지는 가능/불가 극성에 달렸고, 그건 의미 판단이라
# LLM(CONTRADICTED)이 본다. 규칙은 사유 코드를 내지 않고 상태만 남긴다.
# 51번 dev 점검에서 정상 문장 2건(만 19세 이상↔미만, 7천만원 초과↔이하)이
# 이 경우로 NUMERIC_MISMATCH 오탐이 났다.
_COMPLEMENTS = {frozenset({">=", "<"}), frozenset({">", "<="})}


def _lower_bound_dropped(cb: str | None, e: NumericSlot) -> bool:
    """근거의 하한 한정어('최소 1개월 전')를 claim 이 생략한 경우는 과장이 아니다.

    과장은 상한(최고·최대·한도·이하)을 확정값으로 말할 때 생긴다
    ('최고 연 4.0%' → '연 4.0%'). 하한 생략은 정보가 덜할 뿐 고객에게
    유리하게 부풀리지 않는다. 51번 dev 점검의 오탐 1건(변경 최소 1개월 전 게시)이 근거.
    """
    return cb is None and _bound(e) in (">=", ">")


def compare_slot(claim: NumericSlot, evidence: list[NumericSlot]) -> tuple[MatchStatus, str]:
    """claim 수치 하나를 근거 수치들과 대조한다. (상태, 설명)"""
    fam = [e for e in evidence if _family(e.unit) == _family(claim.unit)]
    if not fam:
        return MatchStatus.NOT_FOUND, f"근거에 '{claim.unit}' 계열 수치 없음"

    same_unit = [e for e in fam if e.unit == claim.unit]
    found: list[tuple[MatchStatus, str]] = []

    # 범위 claim 의 양 끝이 근거에 단일 값으로 각각 있으면 일치로 본다.
    #   claim '4.9%~6%' ↔ 근거 표 '4.9% … 6%' (중량별 수수료율 표)
    if claim.is_range:
        singles = [e.value for e in same_unit if not e.is_range]
        if any(_eq(claim.value, v) for v in singles) and any(_eq(claim.value_hi, v) for v in singles):
            found.append((MatchStatus.MATCH, "범위 양 끝이 근거에 각각 존재"))

    for e in same_unit:
        # 범위 대 범위
        if claim.is_range and e.is_range:
            if _eq(claim.value, e.value) and _eq(claim.value_hi, e.value_hi):
                found.append((MatchStatus.MATCH, e.raw))
            continue
        # 단일 대 범위
        if not claim.is_range and e.is_range:
            if e.value - _EPS <= claim.value <= e.value_hi + _EPS:
                found.append((MatchStatus.WITHIN_RANGE, e.raw))
            continue
        if claim.is_range or not _eq(claim.value, e.value):
            continue

        # 값과 단위가 같다 — 의미 차이를 본다
        if claim.period and e.period and claim.period != e.period:
            found.append((MatchStatus.PERIOD_MISMATCH, f"{claim.raw} ↔ {e.raw}"))
            continue
        ck, ek = _kind(claim), _kind(e)
        if ck and ek and frozenset({ck, ek}) in _KIND_CONFLICTS:
            found.append((MatchStatus.QUALIFIER_CONFLICT, f"{claim.qualifier} ↔ {e.qualifier}"))
            continue
        cb, eb = _bound(claim), _bound(e)
        if cb == eb or (cb is not None and eb is None) or _lower_bound_dropped(cb, e):
            # 근거가 확정값인데 claim 이 '최대'를 붙인 것은 보수적이므로 허용
            found.append((MatchStatus.MATCH, e.raw))
        elif cb is None:
            found.append((MatchStatus.BOUND_DROPPED, f"근거 '{e.raw}'의 한정어({e.qualifier or eb}) 누락"))
        elif frozenset({cb, eb}) in _COMPLEMENTS:
            found.append((MatchStatus.COMPARATOR_COMPLEMENT, f"{cb} ↔ {eb}"))
        else:
            found.append((MatchStatus.COMPARATOR_MISMATCH, f"{cb} ↔ {eb}"))

    priority = [MatchStatus.MATCH, MatchStatus.WITHIN_RANGE, MatchStatus.BOUND_DROPPED,
                MatchStatus.COMPARATOR_COMPLEMENT,
                MatchStatus.COMPARATOR_MISMATCH, MatchStatus.QUALIFIER_CONFLICT,
                MatchStatus.PERIOD_MISMATCH]
    for st in priority:
        for s, d in found:
            if s == st:
                return s, d

    # 같은 단위에서 값이 맞는 것이 없다 — %/%p 혼동인지 본다
    for e in fam:
        if e.unit != claim.unit and not e.is_range and not claim.is_range and _eq(e.value, claim.value):
            return MatchStatus.UNIT_CONFUSION, f"{claim.raw} ↔ {e.raw}"
    cands = ", ".join(e.raw for e in same_unit) or ", ".join(e.raw for e in fam)
    return MatchStatus.VALUE_MISMATCH, f"{claim.raw} ↔ 근거 [{cands}]"


_STATUS_TO_CODE = {
    MatchStatus.NOT_FOUND: ReasonCode.NO_EVIDENCE,
    MatchStatus.BOUND_DROPPED: ReasonCode.OVERSTATED,
    MatchStatus.VALUE_MISMATCH: ReasonCode.NUMERIC_MISMATCH,
    MatchStatus.UNIT_CONFUSION: ReasonCode.NUMERIC_MISMATCH,
    MatchStatus.PERIOD_MISMATCH: ReasonCode.NUMERIC_MISMATCH,
    MatchStatus.COMPARATOR_MISMATCH: ReasonCode.NUMERIC_MISMATCH,
    MatchStatus.QUALIFIER_CONFLICT: ReasonCode.NUMERIC_MISMATCH,
}


def numeric_reason_hits(claim_text: str, evidence_text: str) -> tuple[list[ReasonHit], str]:
    """claim 문장과 정렬된 근거 원문을 받아 규칙 사유 코드를 낸다.

    반환: (ReasonHit 목록, rule_checks["numeric"] 값 "pass|fail|n/a")
    """
    claim_slots = extract_numerics(claim_text)
    if not claim_slots:
        return [], "n/a"
    ev_slots = extract_numerics(evidence_text)
    hits: list[ReasonHit] = []
    for cs in claim_slots:
        status, detail = compare_slot(cs, ev_slots)
        code = _STATUS_TO_CODE.get(status)
        if code is not None:
            hits.append(ReasonHit(code, Source.RULE, f"{status.value}: {detail}"))
    return hits, ("fail" if hits else "pass")
