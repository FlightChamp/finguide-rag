"""PolicyGate — 사유 코드에서 판정을 도출하는 규칙 계층.

이 모듈에는 LLM 이 없다. 같은 입력이면 항상 같은 판정이 나온다.
금융 판정은 재현 가능하고 감사 가능해야 하기 때문이다.

판정 규칙 요약
------------
claim 판정
    1. 발급된 사유 코드의 심각도 최댓값 (PROHIBITED > CHECK > OK)
    2. SUPPORTED 는 다른 코드가 하나도 없을 때만 OK 가 된다
    3. 사유 코드가 하나도 없으면 CHECK (검증되지 않은 것은 안내 가능이 아니다)
    4. TIME_VARIANT 가 있으면 NUMERIC_MISMATCH 를 무시한다
       — 근거 문서의 수치는 과거 시점 값이라 '현재 금리' 주장과 비교할 수 없다.
         대신 TIME_VARIANT 자체가 CHECK 를 보장한다.

최종 판정
    PROHIBITED claim 이 하나라도 있으면           → PROHIBITED
    전부 OK 이고 초안 수준 사유도 없으면          → OK
    OK 가 1개 이상, 나머지는 CHECK (또는 초안 사유) → PARTIAL
    그 외                                          → CHECK

fail-closed
    LLM 이 허용되지 않은 코드(예: NUMERIC_MISMATCH, 'OK' 같은 라벨,
    오타)를 내면 버리지 않고 VERIFIER_ERROR 로 바꾼다. 버리면 남은 코드만으로
    OK 가 나올 수 있기 때문이다.
"""

from __future__ import annotations

from typing import Iterable

from .schema import (
    LLM_CLAIM_EMITTABLE,
    REASON_SPECS,
    BVStatus,
    FinalDecision,
    FinalLabel,
    Level,
    ReasonCode,
    ReasonHit,
    Source,
    Verdict,
    Verification,
)

MAX_BV_FIX_ATTEMPTS = 1


# ------------------------------------------------------------------
# 1. LLM 출력 정제
# ------------------------------------------------------------------


def sanitize_llm_codes(raw_codes: Iterable[str]) -> list[ReasonHit]:
    """LLM 이 낸 문자열 코드를 ReasonHit 으로 바꾼다.

    LLM_CLAIM_EMITTABLE 밖의 값(라벨, 규칙 전용 코드, 초안 수준 코드, 오타)은 하나라도 있으면 VERIFIER_ERROR 1건으로 대체된다.
    빈 입력도 VERIFIER_ERROR 다 — LLM 이 아무 판단도 내지 않은 것이므로.
    """
    hits: list[ReasonHit] = []
    rejected: list[str] = []
    for raw in raw_codes:
        key = str(raw).strip().upper()
        try:
            code = ReasonCode(key)
        except ValueError:
            rejected.append(str(raw))
            continue
        if code not in LLM_CLAIM_EMITTABLE:
            rejected.append(str(raw))
            continue
        if all(h.code != code for h in hits):
            hits.append(ReasonHit(code, Source.LLM))

    if rejected:
        hits.append(ReasonHit(ReasonCode.VERIFIER_ERROR, Source.SYSTEM,
                              f"허용되지 않은 LLM 코드: {rejected}"))
    if not hits:
        hits.append(ReasonHit(ReasonCode.VERIFIER_ERROR, Source.SYSTEM,
                              "LLM 이 사유 코드를 내지 않음"))
    return hits


def enforce_sources(hits: Iterable[ReasonHit]) -> list[ReasonHit]:
    """명세에 없는 주체가 발급한 코드를 VERIFIER_ERROR 로 바꾼다.

    규칙 함수가 실수로 CONTRADICTED(LLM 전용)를 내는 것 같은 코드 결함도
    여기서 드러난다.
    """
    out: list[ReasonHit] = []
    for h in hits:
        if h.source in REASON_SPECS[h.code].sources or h.code == ReasonCode.VERIFIER_ERROR:
            out.append(h)
        else:
            out.append(ReasonHit(ReasonCode.VERIFIER_ERROR, Source.SYSTEM,
                                 f"{h.code.value} 는 {h.source.value} 가 발급할 수 없음"))
    return out


# ------------------------------------------------------------------
# 2. claim 판정
# ------------------------------------------------------------------


def _apply_suppressions(hits: list[ReasonHit]) -> list[ReasonHit]:
    codes = {h.code for h in hits}
    if ReasonCode.TIME_VARIANT in codes:
        hits = [h for h in hits if h.code != ReasonCode.NUMERIC_MISMATCH]
    return hits


def derive_verdict(hits: list[ReasonHit]) -> Verdict:
    hits = _apply_suppressions(hits)
    if not hits:
        return Verdict.CHECK
    non_supported = [h for h in hits if h.code != ReasonCode.SUPPORTED]
    if not non_supported:
        return Verdict.OK
    return max((h.severity for h in non_supported), key=lambda v: v.severity)


def detect_conflict(hits: list[ReasonHit]) -> bool:
    """LLM 은 SUPPORTED 라 했는데 규칙이 문제를 찾은 경우.

    이 비율(LLM–Rule Conflict Rate)은 평가 지표로 따로 집계한다.
    """
    llm_ok = any(h.code == ReasonCode.SUPPORTED and h.source == Source.LLM for h in hits)
    rule_bad = any(h.source == Source.RULE and h.severity != Verdict.OK for h in hits)
    return llm_ok and rule_bad


def build_verification(claim_id: str, llm_codes: Iterable[str],
                       rule_hits: Iterable[ReasonHit], *,
                       evidence=None, rule_checks=None,
                       fix_suggestion: str | None = None) -> Verification:
    """LLM 코드 + 규칙 결과 → Verification. ClaimVerifier 의 마지막 단계."""
    hits = sanitize_llm_codes(llm_codes) + enforce_sources(rule_hits)
    hits = _apply_suppressions(hits)
    verdict = derive_verdict(hits)
    return Verification(
        claim_id=claim_id,
        verdict=verdict,
        reasons=hits,
        evidence=list(evidence or []),
        rule_checks=dict(rule_checks or {}),
        llm_rule_conflict=detect_conflict(hits),
        fix_suggestion=fix_suggestion if verdict != Verdict.OK else None,
    )


# ------------------------------------------------------------------
# 3. 최종 판정
# ------------------------------------------------------------------


def aggregate_final(verifications: list[Verification],
                    draft_level_reasons: list[ReasonHit] | None = None,
                    audit_id: str = "") -> FinalDecision:
    draft_level_reasons = list(draft_level_reasons or [])
    for h in draft_level_reasons:
        if REASON_SPECS[h.code].level != Level.DRAFT:
            raise ValueError(f"{h.code.value} 는 초안 수준 사유가 아님")

    summary = {v.value: 0 for v in Verdict}
    for v in verifications:
        summary[v.verdict.value] += 1

    if not verifications:
        final = FinalLabel.CHECK              # 검증할 claim 이 없으면 안내 근거도 없다
    elif summary[Verdict.PROHIBITED.value] > 0:
        final = FinalLabel.PROHIBITED
    elif summary[Verdict.OK.value] == len(verifications) and not draft_level_reasons:
        final = FinalLabel.OK
    elif summary[Verdict.OK.value] > 0:
        final = FinalLabel.PARTIAL
    else:
        final = FinalLabel.CHECK

    return FinalDecision(
        final=final,
        claims_summary=summary,
        draft_level_reasons=draft_level_reasons,
        staff_warnings=build_staff_warnings(verifications, draft_level_reasons),
        audit_id=audit_id,
    )


def build_staff_warnings(verifications: list[Verification],
                         draft_level_reasons: list[ReasonHit]) -> list[str]:
    """심각도 높은 순으로, 같은 사유는 한 번만."""
    seen: set[ReasonCode] = set()
    items: list[tuple[int, str]] = []
    all_hits = [h for v in verifications for h in v.reasons] + draft_level_reasons
    for h in all_hits:
        if h.code == ReasonCode.SUPPORTED or h.code in seen:
            continue
        seen.add(h.code)
        spec = REASON_SPECS[h.code]
        tag = spec.severity.label_ko
        items.append((-spec.severity.severity, f"[{tag}] {spec.staff_message}"))
    return [msg for _, msg in sorted(items, key=lambda x: x[0])]


# ------------------------------------------------------------------
# 4. 역검증 상태 전이
# ------------------------------------------------------------------


def next_bv_status(problems_found: bool, attempts_done: int,
                   max_attempts: int = MAX_BV_FIX_ATTEMPTS) -> BVStatus:
    """역검증 결과 → 다음 상태.

    문제 없음                       → PASS
    문제 있음, 수정 시도 여유 있음    → NEEDS_FIX (재생성 1회)
    문제 있음, 수정 시도 소진         → BLOCK (쉬운 설명 숨김, 원문 인용 안내로 대체)

    자동 수정을 여러 번 돌리지 않는 이유: 비용과 재현성. 같은 입력에서
    시도 횟수만 바뀌어도 결과가 달라지는 구조는 감사가 어렵다.
    """
    if not problems_found:
        return BVStatus.PASS
    if attempts_done < max_attempts:
        return BVStatus.NEEDS_FIX
    return BVStatus.BLOCK
