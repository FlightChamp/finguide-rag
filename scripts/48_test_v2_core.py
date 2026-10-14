"""48 — v2 검증 계층 핵심 규칙 테스트.

대상
----
    verification/numeric.py   수치 추출 · 대조
    verification/policy.py    사유 코드 → 판정 도출, 최종 집계, 역검증 상태 전이
    verification/schema.py    사유 코드 명세 무결성

LLM·인덱스·API 키 없이 돈다. 몇 초면 끝난다.
ClaimVerifier 를 구현하기 전에 이 테스트가 전부 통과해야 한다.
판정 규칙을 바꾸면 이 파일의 기대값을 먼저 고치고, 왜 바꿨는지 커밋 메시지에 남긴다.

실행
----
    python scripts/48_test_v2_core.py            # 요약 + 실패 상세
    python scripts/48_test_v2_core.py --verbose  # 전 케이스 출력

산출물
-----
    reports/v2_core_test_<날짜>.md   케이스별 결과표
종료 코드: 실패가 있으면 1 (알려진 한계 그룹은 실패로 세지 않는다)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.verification.numeric import (  # noqa: E402
    MatchStatus,
    compare_slot,
    extract_numerics,
    numeric_reason_hits,
)
from finguide_rag.verification.policy import (  # noqa: E402
    aggregate_final,
    build_verification,
    derive_verdict,
    next_bv_status,
    sanitize_llm_codes,
)
from finguide_rag.verification.schema import (  # noqa: E402
    LLM_CLAIM_EMITTABLE,
    REASON_SPECS,
    BVStatus,
    FinalLabel,
    Level,
    ReasonCode,
    ReasonHit,
    Source,
    Verdict,
)

REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass
class Case:
    group: str
    name: str
    run: Callable[[], tuple[bool, str]]   # (통과 여부, 실제값 설명)
    known_limit: bool = False


CASES: list[Case] = []


def case(group: str, name: str, known_limit: bool = False):
    def deco(fn):
        CASES.append(Case(group, name, fn, known_limit))
        return fn
    return deco


# ==================================================================
# A. 추출 — (텍스트, [(값, 단위, 추가 기대 필드)])
# ==================================================================

EXTRACT = [
    ("퍼센트", "연 3.5%", [(3.5, "%", {"period": "연"})]),
    ("퍼센트 붙여쓰기", "연3.5%", [(3.5, "%", {"period": "연"})]),
    ("월 금리", "월 0.5%", [(0.5, "%", {"period": "월"})]),
    ("%p", "우대금리 0.3%p", [(0.3, "%p", {"qualifier": "우대"})]),
    ("%포인트", "0.3%포인트 우대", [(0.3, "%p", {})]),
    ("퍼센트포인트", "0.3퍼센트포인트", [(0.3, "%p", {})]),
    ("우대+최고 동시", "우대금리 최고 연 0.3%p", [(0.3, "%p", {"qualifier": "우대 최고", "period": "연"})]),
    ("두 금리 문맥 분리", "기본금리 연 3.0%, 최고 연 3.5%",
     [(3.0, "%", {"qualifier": "기본"}), (3.5, "%", {"qualifier": "최고"})]),
    ("금액 만", "100만원", [(1e6, "원", {})]),
    ("금액 쉼표", "1,000만원", [(1e7, "원", {})]),
    ("금액 천만", "1천만원", [(1e7, "원", {})]),
    ("금액 숫자 없는 천만", "천만원", [(1e7, "원", {})]),
    ("금액 억+천만", "1억 5천만원", [(1.5e8, "원", {})]),
    ("금액 백만", "5백만 원", [(5e6, "원", {})]),
    ("금액 소수 억", "3.5억원", [(3.5e8, "원", {})]),
    ("금액 원 단위", "50,000원", [(5e4, "원", {})]),
    ("금액 앞 한글 붙음", "총100만원", [(1e6, "원", {})]),
    ("원금의 원 무시", "원금 100만원", [(1e6, "원", {})]),
    ("이상", "1,000만원 이상", [(1e7, "원", {"comparator": ">="})]),
    ("초과", "1천만원 초과", [(1e7, "원", {"comparator": ">"})]),
    ("조사 뒤 비교어", "1천만원을 넘는 이체", [(1e7, "원", {"comparator": ">"})]),
    ("미만", "만 14세 미만", [(14, "세", {"comparator": "<"})]),
    ("이내", "1주일 이내", [(7, "일", {"comparator": "<="})]),
    ("년→개월", "1년 이상", [(12, "개월", {"comparator": ">="})]),
    ("복합 기간", "1년 6개월", [(18, "개월", {})]),
    ("범위", "12~36개월", [(12, "개월", {"value_hi": 36})]),
    ("범위 단위 혼합", "6개월~1년", [(6, "개월", {"value_hi": 12})]),
    ("범위 금리", "연 2.0~3.5%", [(2.0, "%", {"value_hi": 3.5, "period": "연"})]),
    ("구어체 범위 '에서 … 사이'", "4.9%에서 6% 사이입니다", [(4.9, "%", {"value_hi": 6.0})]),
    ("구어체 범위 '부터 … 까지'", "12개월부터 36개월까지", [(12, "개월", {"value_hi": 36})]),
    ("'부터'만 있으면 범위 아님", "만 19세부터 가입", [(19, "세", {"comparator": ">="})]),
    ("이상·이하 두 수치", "12개월 이상 36개월 이하",
     [(12, "개월", {"comparator": ">="}), (36, "개월", {"comparator": "<="})]),
    ("날짜", "2026년 3월 10일 시행", [(20260310, "date", {})]),
    ("날짜 점 표기", "2026.03.10", [(20260310, "date", {})]),
    ("연도는 기간 아님", "2026년부터 적용", []),
    ("회차는 횟수 아님", "3회차 납입", []),
    ("주택은 주 아님", "3주택 보유", []),
    ("횟수", "24회 이상 납입", [(24, "회", {"comparator": ">="})]),
]


def _check_extract(text, expected):
    def run():
        got = extract_numerics(text)
        desc = [(s.value, s.unit, s.value_hi, s.comparator, s.qualifier, s.period) for s in got]
        if len(got) != len(expected):
            return False, f"{len(got)}개 추출: {desc}"
        for s, (v, u, extra) in zip(got, expected):
            if abs(s.value - v) > 1e-6 or s.unit != u:
                return False, str(desc)
            for k, want in extra.items():
                have = getattr(s, k)
                if isinstance(want, float | int) and have is not None:
                    if abs(have - want) > 1e-6:
                        return False, str(desc)
                elif have != want:
                    return False, str(desc)
        return True, str(desc)
    return run


for name, text, exp in EXTRACT:
    CASES.append(Case("A.추출", name, _check_extract(text, exp)))


# ==================================================================
# B. 대조 — (claim, 근거, 기대 상태)
# ==================================================================

COMPARE = [
    ("동일", "연 3.5%입니다", "연 3.5%", MatchStatus.MATCH),
    ("값 다름", "연 3.5%입니다", "연 3.0%", MatchStatus.VALUE_MISMATCH),
    ("%와 %p 혼동", "0.3% 우대됩니다", "우대금리 0.3%p", MatchStatus.UNIT_CONFUSION),
    ("최고 → 확정 (과장)", "연 4.0%를 받습니다", "최고 연 4.0%", MatchStatus.BOUND_DROPPED),
    ("우대 최고 → 우대 (과장)", "우대금리 0.3%p", "우대금리 최고 0.3%p", MatchStatus.BOUND_DROPPED),
    ("기본 ↔ 최고 혼동", "기본금리 연 3.5%", "기본금리 연 3.0%, 최고 연 3.5%", MatchStatus.QUALIFIER_CONFLICT),
    ("기본 일치", "기본금리 연 3.0%", "기본금리 연 3.0%, 최고 연 3.5%", MatchStatus.MATCH),
    ("연 ↔ 월", "월 0.5%", "연 0.5%", MatchStatus.PERIOD_MISMATCH),
    ("이상 ↔ 초과", "1천만원 초과", "1천만원 이상", MatchStatus.COMPARATOR_MISMATCH),
    ("최대 ↔ 이하 동치", "최대 1천만원", "1천만원 이하", MatchStatus.MATCH),
    ("확정값에 최대 추가 허용", "최대 100만원", "100만원", MatchStatus.MATCH),
    ("범위 안 값", "24개월로 가입", "12~36개월", MatchStatus.WITHIN_RANGE),
    ("범위 밖 값", "48개월로 가입", "12~36개월", MatchStatus.VALUE_MISMATCH),
    ("범위 일치", "가입기간 1~3년", "12~36개월", MatchStatus.MATCH),
    ("범위 양 끝이 근거 표에 각각 존재", "수수료율 4.9%~6%", "10g 6%, 37.5g 5%, 100g 4.9%", MatchStatus.MATCH),
    ("범위 한쪽 끝만 존재", "수수료율 4.0%~6%", "10g 6%, 100g 4.9%", MatchStatus.VALUE_MISMATCH),
    ("년=개월 등가", "1년 이상", "12개월 이상", MatchStatus.MATCH),
    ("금액 표기 등가", "1,000만원", "1천만원", MatchStatus.MATCH),
    ("근거에 해당 단위 없음", "3회 납입", "연 3.5%", MatchStatus.NOT_FOUND),
    ("나이 경계", "만 14세 이하", "만 14세 미만", MatchStatus.COMPARATOR_MISMATCH),
    ("여집합은 규칙이 판정 안 함 (이상 ↔ 미만)", "만 19세 이상 가입 가능", "만 19세 미만은 가입 불가",
     MatchStatus.COMPARATOR_COMPLEMENT),
    ("하한 한정어 생략은 과장 아님", "변경 1개월 전에 게시", "변경 최소 1개월 전에 게시", MatchStatus.MATCH),
    ("상한 표현 바꿔 말하기(최대↔한도)는 일치", "한도 1천만원", "최대 1천만원", MatchStatus.MATCH),
]


def _check_compare(claim, evidence, want):
    def run():
        cs = extract_numerics(claim)
        if not cs:
            return False, "claim 에서 수치 추출 실패"
        status, detail = compare_slot(cs[0], extract_numerics(evidence))
        return status == want, f"{status.value} ({detail})"
    return run


for name, cl, ev, want in COMPARE:
    CASES.append(Case("B.대조", name, _check_compare(cl, ev, want)))


@case("B.대조", "사유 코드 매핑: %/%p → NUMERIC_MISMATCH")
def _():
    hits, chk = numeric_reason_hits("0.3% 우대됩니다", "우대금리 0.3%p")
    codes = [h.code for h in hits]
    return codes == [ReasonCode.NUMERIC_MISMATCH] and chk == "fail", f"{codes}, {chk}"


@case("B.대조", "사유 코드 매핑: 상한 누락 → OVERSTATED")
def _():
    hits, chk = numeric_reason_hits("연 4.0%를 받습니다", "최고 연 4.0%")
    codes = [h.code for h in hits]
    return codes == [ReasonCode.OVERSTATED], f"{codes}, {chk}"


@case("B.대조", "여집합 → 사유 코드 없음 (LLM CONTRADICTED 담당)")
def _():
    hits, chk = numeric_reason_hits("연소득 7천만원 초과면 대출 불가", "연소득 7천만원 이하")
    return hits == [] and chk == "pass", f"{hits}, {chk}"


@case("B.대조", "수치 없는 claim → n/a")
def _():
    hits, chk = numeric_reason_hits("중도해지 시 우대금리는 적용되지 않습니다", "연 3.0%")
    return hits == [] and chk == "n/a", f"{hits}, {chk}"


# ==================================================================
# C. 정책
# ==================================================================

def H(code, src=Source.RULE):
    return ReasonHit(ReasonCode[code], src)


@case("C.정책", "SUPPORTED 단독 → OK")
def _():
    v = derive_verdict([H("SUPPORTED", Source.LLM)])
    return v == Verdict.OK, v.value


@case("C.정책", "SUPPORTED + 규칙 NUMERIC_MISMATCH → PROHIBITED (규칙 우선)")
def _():
    ver = build_verification("c1", ["SUPPORTED"], [H("NUMERIC_MISMATCH")])
    return ver.verdict == Verdict.PROHIBITED and ver.llm_rule_conflict, \
        f"{ver.verdict.value}, conflict={ver.llm_rule_conflict}"


@case("C.정책", "SUPPORTED + OVERSTATED → CHECK")
def _():
    v = derive_verdict([H("SUPPORTED", Source.LLM), H("OVERSTATED")])
    return v == Verdict.CHECK, v.value


@case("C.정책", "코드 없음 → CHECK (검증 안 된 것은 안내 가능 아님)")
def _():
    v = derive_verdict([])
    return v == Verdict.CHECK, v.value


@case("C.정책", "TIME_VARIANT 가 NUMERIC_MISMATCH 를 억제")
def _():
    ver = build_verification("c1", ["SUPPORTED"], [H("TIME_VARIANT"), H("NUMERIC_MISMATCH")])
    codes = [h.code.value for h in ver.reasons]
    return ver.verdict == Verdict.CHECK and "NUMERIC_MISMATCH" not in codes, \
        f"{ver.verdict.value} {codes}"


@case("C.정책", "LLM 이 'OK' 라벨을 직접 냄 → VERIFIER_ERROR, CHECK")
def _():
    ver = build_verification("c1", ["OK"], [])
    codes = [h.code.value for h in ver.reasons]
    return ver.verdict == Verdict.CHECK and "VERIFIER_ERROR" in codes, f"{ver.verdict.value} {codes}"


@case("C.정책", "LLM 이 규칙 전용 코드를 냄 → VERIFIER_ERROR (SUPPORTED 와 섞여도 OK 금지)")
def _():
    ver = build_verification("c1", ["SUPPORTED", "NUMERIC_MISMATCH"], [])
    return ver.verdict == Verdict.CHECK, ver.verdict.value


@case("C.정책", "LLM 빈 출력 → VERIFIER_ERROR")
def _():
    hits = sanitize_llm_codes([])
    return [h.code for h in hits] == [ReasonCode.VERIFIER_ERROR], str([h.code.value for h in hits])


@case("C.정책", "규칙 함수가 LLM 전용 코드 발급 → VERIFIER_ERROR")
def _():
    ver = build_verification("c1", ["SUPPORTED"], [H("CONTRADICTED", Source.RULE)])
    codes = [h.code.value for h in ver.reasons]
    return "VERIFIER_ERROR" in codes and ver.verdict == Verdict.CHECK, f"{ver.verdict.value} {codes}"


@case("C.정책", "OK 판정에는 수정 제안을 붙이지 않음")
def _():
    ver = build_verification("c1", ["SUPPORTED"], [], fix_suggestion="x")
    return ver.fix_suggestion is None, str(ver.fix_suggestion)


def _vers(*verdict_codes):
    out = []
    for i, codes in enumerate(verdict_codes):
        llm = [c for c in codes if ReasonCode[c] in LLM_CLAIM_EMITTABLE and c != "MISSING_CONDITION"]
        rule = [H(c) for c in codes if c not in llm]
        out.append(build_verification(f"c{i}", llm or ["SUPPORTED"], rule))
    return out


@case("C.정책", "최종: 전부 OK → OK")
def _():
    d = aggregate_final(_vers(["SUPPORTED"], ["SUPPORTED"]))
    return d.final == FinalLabel.OK, d.final.value


@case("C.정책", "최종: OK + CHECK → PARTIAL")
def _():
    d = aggregate_final(_vers(["SUPPORTED"], ["NEEDS_CUSTOMER_DATA"]))
    return d.final == FinalLabel.PARTIAL, d.final.value


@case("C.정책", "최종: PROHIBITED 하나라도 → PROHIBITED")
def _():
    d = aggregate_final(_vers(["SUPPORTED"], ["SUPPORTED"], ["PRODUCT_MISMATCH"]))
    return d.final == FinalLabel.PROHIBITED, d.final.value


@case("C.정책", "최종: 전부 OK 이지만 초안 수준 누락 → PARTIAL")
def _():
    d = aggregate_final(_vers(["SUPPORTED"]),
                        [ReasonHit(ReasonCode.OMITTED_REQUIRED_INFO, Source.LLM)])
    return d.final == FinalLabel.PARTIAL, d.final.value


@case("C.정책", "최종: claim 0개 → CHECK")
def _():
    d = aggregate_final([])
    return d.final == FinalLabel.CHECK, d.final.value


@case("C.정책", "직원 주의사항: 심각도 순 정렬, 중복 제거")
def _():
    d = aggregate_final(_vers(["OVERSTATED"], ["NUMERIC_MISMATCH"], ["OVERSTATED"]))
    w = d.staff_warnings
    return len(w) == 2 and w[0].startswith("[안내 금지]"), str(w)


@case("C.정책", "역검증 전이: 문제 없음 → PASS / 1회차 → NEEDS_FIX / 소진 → BLOCK")
def _():
    got = (next_bv_status(False, 0), next_bv_status(True, 0), next_bv_status(True, 1))
    want = (BVStatus.PASS, BVStatus.NEEDS_FIX, BVStatus.BLOCK)
    return got == want, str([g.value for g in got])


@case("C.정책", "명세 무결성: 모든 코드에 명세 존재, 초안 수준 코드는 claim 검증에서 발급 불가")
def _():
    missing = [c for c in ReasonCode if c not in REASON_SPECS]
    draft_bad = [c for c, s in REASON_SPECS.items() if s.level == Level.DRAFT and c in LLM_CLAIM_EMITTABLE]
    return not missing and not draft_bad, f"missing={missing}, draft={draft_bad}"


@case("C.정책", "최종: 전부 OK 이지만 초안 수준 범위 혼합(규칙) → PARTIAL")
def _():
    d = aggregate_final(_vers(["SUPPORTED"], ["SUPPORTED"]),
                        [ReasonHit(ReasonCode.MIXED_SCOPE, Source.RULE)])
    return d.final == FinalLabel.PARTIAL, d.final.value


@case("C.정책", "LLM 이 claim 수준에서 낼 수 있는 코드는 정확히 4개")
def _():
    want = {"SUPPORTED", "CONTRADICTED", "MISSING_CONDITION", "NO_EVIDENCE"}
    have = {c.value for c in LLM_CLAIM_EMITTABLE}
    return have == want, str(sorted(have))


@case("C.정책", "초안 수준 코드를 claim 검증에서 내면 → VERIFIER_ERROR")
def _():
    ver = build_verification("c1", ["SUPPORTED", "OMITTED_REQUIRED_INFO"], [])
    return ver.verdict == Verdict.CHECK, ver.verdict.value


# ==================================================================
# D. 알려진 한계 — 현재 동작을 '기록'한다. 고치면 이 케이스가 실패로 바뀌어 알려준다.
# ==================================================================

@case("D.알려진 한계", "한글 숫자 '삼백만원'은 추출하지 않음", known_limit=True)
def _():
    got = extract_numerics("삼백만원")
    # 한글 숫자 앞 글자 때문에 금액 전체를 건너뛴다(오인보다는 누락). 근거·claim 모두 같은 방식이라 대조 시 NOT_FOUND 로 처리됨
    return True, str([(s.value, s.unit) for s in got])


@case("D.알려진 한계", "'매월 25일'은 기간 25일로 추출됨", known_limit=True)
def _():
    got = extract_numerics("매월 25일 자동이체")
    return True, str([(s.value, s.unit) for s in got])


@case("D.알려진 한계", "금액 범위 '100만원~1,000만원'은 단일 수치 2개로 추출", known_limit=True)
def _():
    got = extract_numerics("100만원~1,000만원")
    return True, str([(s.value, s.unit, s.value_hi) for s in got])


# ==================================================================
# 실행
# ==================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results = []
    for c in CASES:
        try:
            ok, desc = c.run()
        except Exception as e:  # 테스트 자체의 예외도 실패로 센다
            ok, desc = False, f"예외: {type(e).__name__}: {e}"
        results.append((c, ok, desc))

    groups: dict[str, list] = {}
    for c, ok, desc in results:
        groups.setdefault(c.group, []).append((c, ok, desc))

    print("=" * 64)
    print("FinGuide-RAG v2 핵심 규칙 테스트")
    print("=" * 64)
    total_fail = 0
    for g, rows in groups.items():
        if g.startswith("D."):
            print(f"{g:<16} {len(rows):>3}건 기록 (실패로 세지 않음)")
            continue
        n_ok = sum(ok for _, ok, _ in rows)
        total_fail += len(rows) - n_ok
        mark = "OK " if n_ok == len(rows) else "FAIL"
        print(f"{g:<16} {n_ok:>3}/{len(rows):<3} {mark}")

    fails = [(c, d) for c, ok, d in results if not ok and not c.known_limit]
    if fails:
        print("\n[실패 상세]")
        for c, d in fails:
            print(f"  - {c.group} / {c.name}\n      실제: {d}")
    print("\n[알려진 한계 — 현재 동작]")
    for c, ok, d in results:
        if c.known_limit:
            print(f"  - {c.name}\n      현재: {d}")
    if args.verbose:
        print("\n[전체]")
        for c, ok, d in results:
            print(f"  {'✔' if ok else '✘'} {c.group} / {c.name}: {d}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"v2_core_test_{date.today().isoformat()}.md"
    lines = [f"# v2 핵심 규칙 테스트 ({date.today().isoformat()})", "",
             "| 그룹 | 케이스 | 결과 | 실제 |", "|---|---|---|---|"]
    for c, ok, d in results:
        r = "한계" if c.known_limit else ("통과" if ok else "**실패**")
        lines.append(f"| {c.group} | {c.name} | {r} | `{d.replace('|', '/')}` |")
    out.write_text("\n".join(lines), encoding="utf-8")

    n_scored = sum(1 for c in CASES if not c.known_limit)
    print(f"\n총 {n_scored}건 중 실패 {total_fail}건  →  {out.relative_to(PROJECT_ROOT)}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
