"""62 — v2-B 쉬운 설명 · 역검증 테스트 (LLM·API 키 없이, 몇 초).

대상
----
    verification/simplify.py     OK claim 만 LLM 에 보내는지, 직원 확인 영역·제외 목록 구성
    verification/back_verify.py  결정적 규칙 5종 + 조건 보존 판정 + PASS/NEEDS_FIX/BLOCK + 1회 수정
    verification/bv_perturb.py   교란 10종이 실제로 적용되는지, 역검증이 잡는지
    verification/policy.py       decide_bv_status

LLM 자리에는 가짜 클라이언트를 넣는다. 가짜 조건 검사기는 조건의 수치·핵심어가 문구에
남아 있으면 PRESERVED, 없으면 DROPPED 라고 답한다 (실제 LLM 의 성능을 재는 테스트가 아니다).

그룹
----
    A  분리 원칙 (CHECK·PROHIBITED 는 LLM 에 가지 않음, 고객 문구에 없음)
    B  정상 쉬운 설명은 PASS (표현 바꿈·단위 표기 차이·허용 기관 등 — 오탐 방지)
    C  결정적 규칙이 교란을 잡는가 (조건 삭제 제외 9종)
    D  조건 보존 판정 (LLM 응답 → 코드 판정, fail-closed)
    E  상태 전이 (1회 수정, BLOCK, 노출 여부)
    F  교란 생성기 자체

실행
----
    python scripts/62_test_v2_simplify.py
    python scripts/62_test_v2_simplify.py --verbose
종료 코드: 실패가 있으면 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.verification.back_verify import (  # noqa: E402
    COND_SYSTEM,
    BackVerifier,
    BVContext,
    run_simplify,
)
from finguide_rag.verification.bv_perturb import TYPES, key_token, perturb, still_present  # noqa: E402
from finguide_rag.verification.policy import decide_bv_status  # noqa: E402
from finguide_rag.verification.schema import (  # noqa: E402
    BVCode,
    BVProblem,
    BVStatus,
    Claim,
    EvidenceRef,
    ReasonCode,
    ReasonHit,
    SimpleItem,
    Source,
    Verdict,
    Verification,
)
from finguide_rag.verification.simplify import FIX_SYSTEM, SIMPLIFY_SYSTEM, Simplifier  # noqa: E402

# ------------------------------------------------------------------
# 고정 입력 (가상의 예시 — 실제 상품 정보가 아니다)
# ------------------------------------------------------------------
Q = "예금을 만기 전에 해지하면 이율이 어떻게 되나요?"
CLAIMS = [
    Claim("c1", "하나의정기예금은 만기 전에 해지하면 중도해지이율이 적용됩니다.", (0, 0),
          product="하나의정기예금", event="중도해지"),
    Claim("c2", "가입기간 6개월 이상 경과 후 해지 시 기본이율의 최고 50%가 적용될 수 있습니다.", (0, 0),
          conditions=["가입기간 6개월 이상 경과 후 해지 시"]),
    Claim("c3", "우대금리는 급여이체 실적이 있는 경우에 한해 연 0.2%p 적용됩니다.", (0, 0),
          conditions=["급여이체 실적이 있는 경우에 한해"]),
    Claim("c4", "현재 이 상품의 기본금리는 연 3.1%입니다.", (0, 0)),
    Claim("c5", "만기 후에는 국민은행 기준 금리가 적용됩니다.", (0, 0)),
]
EV = [EvidenceRef("d1", "k1", "…", institution="하나은행")]


def _v(cid, verdict, codes):
    return Verification(cid, verdict, [ReasonHit(c, Source.RULE if c != ReasonCode.SUPPORTED else Source.LLM)
                                       for c in codes], evidence=EV)


VERS = [
    _v("c1", Verdict.OK, [ReasonCode.SUPPORTED]),
    _v("c2", Verdict.OK, [ReasonCode.SUPPORTED]),
    _v("c3", Verdict.OK, [ReasonCode.SUPPORTED]),
    _v("c4", Verdict.CHECK, [ReasonCode.TIME_VARIANT]),
    _v("c5", Verdict.PROHIBITED, [ReasonCode.INSTITUTION_MISMATCH]),
]
KNOWN = ["하나의정기예금", "주거래하나월복리적금", "급여하나통장"]

CLEAN = {
    "core": [
        {"text": "하나의정기예금은 만기 전에 해지하면 만기 전에 해지할 때 적용되는 이율(중도해지이율)이 적용됩니다.",
         "claim_ids": ["c1"]},
        {"text": "가입하고 6개월 이상 지난 뒤 해지하면 기본이율의 최고 50%가 적용될 수 있습니다.",
         "claim_ids": ["c2"]},
        {"text": "우대금리로 연 0.2%p가 더해집니다.", "claim_ids": ["c3"]},
    ],
    "conditions": [
        {"text": "가입기간이 6개월 이상 지난 뒤 해지할 때의 기준입니다.", "claim_ids": ["c2"],
         "condition_ids": ["k1"]},
        {"text": "우대금리는 급여이체 실적이 있는 경우에만 받습니다.", "claim_ids": ["c3"],
         "condition_ids": ["k2"]},
    ],
}


class FakeLLM:
    """system 프롬프트로 역할을 구분한다. calls 에 (역할, user) 를 남긴다."""

    def __init__(self, simplify=None, fix=None, cond="auto"):
        self.simplify, self.fix, self.cond = simplify, fix, cond
        self.calls: list[tuple[str, str]] = []

    def json(self, system, user, max_tokens=1500, tag=""):
        if system == SIMPLIFY_SYSTEM:
            self.calls.append(("simplify", user))
            return self.simplify
        if system == FIX_SYSTEM:
            self.calls.append(("fix", user))
            return self.fix
        if system == COND_SYSTEM:
            self.calls.append(("cond", user))
            if self.cond != "auto":
                return self.cond
            return _auto_cond(user)
        raise AssertionError("알 수 없는 프롬프트")


def _auto_cond(user: str) -> dict:
    conds = re.findall(r"^(k\d+) \((c\d+)\): (.+)$", user.split("[지켜야 할 조건]")[1].split("[쉬운 설명]")[0], re.M)
    items = re.findall(r"^([sn]\d+): (.+)$", user.split("[쉬운 설명]")[1], re.M)
    res = []
    for kid, _, text in conds:
        where = [iid for iid, t in items if still_present(text, t)]
        res.append({"cond_id": kid, "status": "PRESERVED" if where else "DROPPED", "where": where})
    return {"results": res, "extra": []}


def clean_expl():
    s = Simplifier(FakeLLM(simplify=CLEAN))
    expl, gen, ok = s.simplify(Q, CLAIMS, VERS)
    assert not gen, gen
    return expl


CTX = BVContext.build(CLAIMS, VERS)


def rules_bv():
    return BackVerifier(None, None, KNOWN, condition_check=False)


# ------------------------------------------------------------------
# 테스트 러너
# ------------------------------------------------------------------
RESULTS: list[tuple[str, str, bool, str]] = []


def case(group: str, name: str):
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((group, name, True, str(detail)))
        except AssertionError as e:
            RESULTS.append((group, name, False, str(e)))
        except Exception as e:   # noqa: BLE001
            RESULTS.append((group, name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


def codes(bv):
    return sorted({p.code.value for p in bv.problems})


# ---------------- A 분리 원칙 ----------------
@case("A", "LLM 입력에 OK claim 만 들어간다")
def _():
    llm = FakeLLM(simplify=CLEAN)
    Simplifier(llm).simplify(Q, CLAIMS, VERS)
    user = llm.calls[0][1]
    assert "c1:" in user and "c3:" in user
    assert CLAIMS[3].text not in user, "CHECK claim 이 LLM 에 감"
    assert CLAIMS[4].text not in user, "PROHIBITED claim 이 LLM 에 감"


@case("A", "CHECK 는 직원 확인 영역에 원문 그대로, PROHIBITED 는 제외 목록")
def _():
    e = clean_expl()
    assert [s.claim_id for s in e.staff_check] == ["c4"]
    assert e.staff_check[0].text == CLAIMS[3].text
    assert e.staff_check[0].reasons == ["TIME_VARIANT"]
    assert e.excluded == ["c5"]


@case("A", "고객 문구에 직원 확인·제외 내용이 없다")
def _():
    t = clean_expl().customer_text()
    assert "3.1%" not in t and "국민은행" not in t


@case("A", "지켜야 할 조건은 OK claim 의 conditions 슬롯에서 온다")
def _():
    e = clean_expl()
    assert [(k.cond_id, k.claim_id) for k in e.required_conditions] == [("k1", "c2"), ("k2", "c3")]


@case("A", "판정이 없는 claim 은 CHECK 로 취급 (fail-closed)")
def _():
    extra = Claim("c9", "판정이 누락된 문장입니다.", (0, 0))
    e, _, ok = Simplifier(FakeLLM(simplify=CLEAN)).simplify(Q, CLAIMS + [extra], VERS)
    assert "c9" in [s.claim_id for s in e.staff_check]


@case("A", "OK claim 이 없으면 쉬운 설명을 만들지 않는다 (LLM 호출 0)")
def _():
    llm = FakeLLM(simplify=CLEAN)
    vers = [_v(c.claim_id, Verdict.CHECK, [ReasonCode.NO_EVIDENCE]) for c in CLAIMS]
    out = run_simplify(Q, CLAIMS, vers, Simplifier(llm), BackVerifier(llm, None, KNOWN))
    assert out.status is None and out.skipped_reason == "no_ok_claims" and not llm.calls
    assert not out.customer_visible


# ---------------- B 정상은 PASS ----------------
@case("B", "정상 쉬운 설명 → 규칙 문제 0")
def _():
    bv = rules_bv().verify(clean_expl(), CTX)
    assert bv.status == BVStatus.PASS, codes(bv) + [p.detail for p in bv.problems]


@case("B", "정상 쉬운 설명 → 조건 검사 포함 PASS")
def _():
    llm = FakeLLM()
    bv = BackVerifier(llm, None, KNOWN).verify(clean_expl(), CTX)
    assert bv.status == BVStatus.PASS and bv.customer_visible, codes(bv)
    assert bv.condition_results == {"k1": "PRESERVED", "k2": "PRESERVED"}


def _with(core_idx, text):
    e = clean_expl()
    e.core[core_idx].text = text
    return rules_bv().verify(e, CTX)


@case("B", "금액 표기 차이(1천만원 ↔ 1,000만원)는 같은 수치")
def _():
    claims = CLAIMS[:1] + [Claim("c2", "1천만원 이상 예치 시 우대됩니다.", (0, 0))]
    vers = VERS[:2]
    ctx = BVContext.build(claims, vers)
    e = clean_expl()
    e.core = [SimpleItem("s1", CLEAN["core"][0]["text"], ["c1"]),
              SimpleItem("s2", "1,000만원 이상 맡기면 우대됩니다.", ["c2"])]
    e.conditions, e.required_conditions = [], []
    bv = rules_bv().verify(e, ctx)
    assert bv.status == BVStatus.PASS, [p.detail for p in bv.problems]


@case("B", "'에 한해' → '경우만' 은 한정 표현 보존")
def _():
    e = clean_expl()
    e.conditions[1].text = "우대금리는 급여이체 실적이 있는 경우만 받습니다."
    bv = rules_bv().verify(e, CTX)
    assert BVCode.HEDGE_LOST.value not in codes(bv), codes(bv)


@case("B", "근거 기관(하나은행) 언급은 추가 기관이 아니다")
def _():
    bv = _with(0, "하나은행 하나의정기예금은 만기 전에 해지하면 중도해지이율이 적용됩니다.")
    assert BVCode.INSTITUTION_ADDED.value not in codes(bv), codes(bv)


@case("B", "'최고 50%' ↔ '50%까지' 는 같은 상한")
def _():
    bv = _with(1, "가입하고 6개월 이상 지난 뒤 해지하면 기본이율의 50%까지 적용될 수 있습니다.")
    assert bv.status == BVStatus.PASS, [p.detail for p in bv.problems]


@case("B", "확정값에 '최대'를 붙이는 것은 약화이므로 허용")
def _():
    bv = _with(2, "우대금리로 최대 연 0.2%p가 더해집니다.")
    assert bv.status == BVStatus.PASS, [p.detail for p in bv.problems]


@case("B", "다른 OK claim 의 수치를 옮긴 문구는 수치 문제 아님 (인용 id 만 어긋남)")
def _():
    e = clean_expl()
    e.core[2].claim_ids = ["c2", "c3"]
    bv = rules_bv().verify(e, CTX)
    assert not [p for p in bv.problems if p.code.value.startswith("NUMERIC")], codes(bv)


@case("B", "원문 '반드시' 동의어(하여야)가 있으면 '반드시'는 강조어 추가 아님")
def _():
    claims = [Claim("c1", "신분증을 지참하여야 합니다.", (0, 0))]
    vers = [_v("c1", Verdict.OK, [ReasonCode.SUPPORTED])]
    e, _, _ = Simplifier(FakeLLM(simplify={"core": [{"text": "신분증을 반드시 가져오셔야 합니다.",
                                                     "claim_ids": ["c1"]}]})).simplify(Q, claims, vers)
    bv = rules_bv().verify(e, BVContext.build(claims, vers))
    assert bv.status == BVStatus.PASS, codes(bv)


# ---------------- C 규칙이 교란을 잡는가 ----------------
RULE_TYPES = [t for t in TYPES if t != "CONDITION_DROP"]
EXPECT = {
    "NUM_VALUE": {"NUMERIC_CHANGED", "NUMERIC_ADDED", "NUMERIC_DROPPED"},
    "NUM_UNIT": {"NUMERIC_CHANGED", "NUMERIC_ADDED", "NUMERIC_DROPPED"},
    "NUM_COMPARATOR": {"NUMERIC_CHANGED"},
    "BOUND_DROP": {"STRENGTHENED"},
    "PRODUCT_SWAP": {"PRODUCT_ADDED", "PRODUCT_DROPPED"},
    "INSTITUTION": {"INSTITUTION_ADDED"},
    "INTENSIFIER": {"STRENGTHENED"},
    "HEDGE_DROP": {"HEDGE_LOST"},
    "CHECK_LEAK": {"CHECK_LEAK", "NUMERIC_CHANGED", "NUMERIC_ADDED"},
}
for _t in RULE_TYPES:
    def _mk(t):
        @case("C", f"{t} 교란 → 기대 코드로 탐지 (시드 0~4)")
        def _():
            base = clean_expl()
            made, out = 0, []
            for seed in range(5):
                r = perturb(base, t, seed, products=["하나의정기예금"], known_products=KNOWN)
                if r is None:
                    continue
                made += 1
                bv = rules_bv().verify(r[0], CTX)
                got = set(codes(bv))
                assert bv.status != BVStatus.PASS, f"놓침: {r[1]}"
                assert got & EXPECT[t], f"{r[1]} → {sorted(got)}"
                out.append(f"{r[1]} → {sorted(got)}")
            assert made, "교란을 만들 곳이 없음"
            return out[0]
    _mk(_t)


@case("C", "CHECK claim 을 출처로 대면 CHECK_LEAK")
def _():
    e = clean_expl()
    e.core[0].claim_ids = ["c1", "c4"]
    assert "CHECK_LEAK" in codes(rules_bv().verify(e, CTX))


@case("C", "PROHIBITED claim 을 출처로 대면 PROHIBITED_LEAK")
def _():
    e = clean_expl()
    e.core[0].claim_ids = ["c5"]
    assert "PROHIBITED_LEAK" in codes(rules_bv().verify(e, CTX))


@case("C", "출처 없는 문구 · 없는 claim id → UNKNOWN_SOURCE")
def _():
    e = clean_expl()
    e.core[0].claim_ids = []
    e.core[1].claim_ids = ["c2", "c99"]
    bv = rules_bv().verify(e, CTX)
    assert [p.code for p in bv.problems].count(BVCode.UNKNOWN_SOURCE) == 2, codes(bv)


@case("C", "OK claim 누락 → CLAIM_UNCOVERED")
def _():
    e = clean_expl()
    e.core = e.core[:2]
    e.conditions = e.conditions[:1]
    assert "CLAIM_UNCOVERED" in codes(rules_bv().verify(e, CTX))


@case("C", "직원 확인 항목 누락 → LOST_CHECK → 즉시 BLOCK")
def _():
    e = clean_expl()
    e.staff_check = []
    bv = rules_bv().verify(e, CTX)
    assert bv.status == BVStatus.BLOCK and "LOST_CHECK" in codes(bv)


@case("C", "원문 수치 누락 → NUMERIC_DROPPED")
def _():
    bv = _with(2, "급여이체를 하시면 우대금리가 더해집니다.")
    assert "NUMERIC_DROPPED" in codes(bv), codes(bv)


# ---------------- D 조건 보존 ----------------
@case("D", "조건 삭제 교란 → DROPPED → NEEDS_FIX")
def _():
    base = clean_expl()
    made = 0
    for seed in range(5):
        r = perturb(base, "CONDITION_DROP", seed)
        if r is None:
            continue
        made += 1
        bv = BackVerifier(FakeLLM(), None, KNOWN).verify(r[0], CTX)
        assert bv.status == BVStatus.NEEDS_FIX and "CONDITION_DROPPED" in codes(bv), (r[1], codes(bv))
    assert made, "교란 없음"


@case("D", "PRESERVED 인데 위치(where) 없음 → CONDITION_UNVERIFIED")
def _():
    llm = FakeLLM(cond={"results": [{"cond_id": "k1", "status": "PRESERVED", "where": []},
                                    {"cond_id": "k2", "status": "PRESERVED", "where": ["n2"]}]})
    bv = BackVerifier(llm, None, KNOWN).verify(clean_expl(), CTX)
    assert codes(bv) == ["CONDITION_UNVERIFIED"] and bv.status == BVStatus.NEEDS_FIX


@case("D", "없는 문구 id 를 where 로 대면 위치 없음과 같다")
def _():
    llm = FakeLLM(cond={"results": [{"cond_id": "k1", "status": "PRESERVED", "where": ["s99"]},
                                    {"cond_id": "k2", "status": "PRESERVED", "where": ["n2"]}]})
    assert codes(BackVerifier(llm, None, KNOWN).verify(clean_expl(), CTX)) == ["CONDITION_UNVERIFIED"]


@case("D", "조건 판단 누락 → CONDITION_UNVERIFIED (NO_RESULT 기록)")
def _():
    llm = FakeLLM(cond={"results": [{"cond_id": "k2", "status": "PRESERVED", "where": ["n2"]}]})
    bv = BackVerifier(llm, None, KNOWN).verify(clean_expl(), CTX)
    assert bv.condition_results.get("k1") == "NO_RESULT" and "CONDITION_UNVERIFIED" in codes(bv)


@case("D", "ALTERED → CONDITION_ALTERED")
def _():
    llm = FakeLLM(cond={"results": [{"cond_id": "k1", "status": "ALTERED", "where": ["n1"]},
                                    {"cond_id": "k2", "status": "PRESERVED", "where": ["n2"]}]})
    assert codes(BackVerifier(llm, None, KNOWN).verify(clean_expl(), CTX)) == ["CONDITION_ALTERED"]


@case("D", "조건 검사 LLM 응답 없음 → BV_ERROR → 즉시 BLOCK")
def _():
    bv = BackVerifier(FakeLLM(cond=None), None, KNOWN).verify(clean_expl(), CTX)
    assert bv.status == BVStatus.BLOCK and codes(bv) == ["BV_ERROR"]


@case("D", "LLM 객체 자체가 없는데 조건 검사 켜짐 → BV_ERROR")
def _():
    bv = BackVerifier(None, None, KNOWN, condition_check=True).verify(clean_expl(), CTX)
    assert bv.status == BVStatus.BLOCK


@case("D", "목록 외 조건(extra): 원문에 있는 인용만 받는다")
def _():
    base = {"results": [{"cond_id": "k1", "status": "PRESERVED", "where": ["n1"]},
                        {"cond_id": "k2", "status": "PRESERVED", "where": ["n2"]}]}
    good = dict(base, extra=[{"source_quote": "만기 전에 해지하면", "claim_id": "c1"}])
    bad = dict(base, extra=[{"source_quote": "법인 고객은 제외", "claim_id": "c1"}])
    bv1 = BackVerifier(FakeLLM(cond=good), None, KNOWN).verify(clean_expl(), CTX)
    bv2 = BackVerifier(FakeLLM(cond=bad), None, KNOWN).verify(clean_expl(), CTX)
    assert "CONDITION_DROPPED" in codes(bv1)
    assert bv2.status == BVStatus.PASS and bv2.condition_results.get("extra1") == "REJECTED"


# ---------------- E 상태 전이 ----------------
@case("E", "decide_bv_status 표")
def _():
    fx, hard = BVProblem(BVCode.NUMERIC_CHANGED), BVProblem(BVCode.BV_ERROR)
    assert decide_bv_status([], 0) == BVStatus.PASS
    assert decide_bv_status([fx], 0) == BVStatus.NEEDS_FIX
    assert decide_bv_status([fx], 1) == BVStatus.BLOCK
    assert decide_bv_status([hard], 0) == BVStatus.BLOCK
    assert decide_bv_status([fx, hard], 0) == BVStatus.BLOCK


BAD = json.loads(json.dumps(CLEAN))
BAD["core"][1]["text"] = "가입하고 6개월 이상 지난 뒤 해지하면 기본이율의 50%가 적용됩니다."


@case("E", "1차 NEEDS_FIX → 수정본 정상 → PASS (시도 1, 생성 호출 2)")
def _():
    llm = FakeLLM(simplify=BAD, fix=CLEAN)
    out = run_simplify(Q, CLAIMS, VERS, Simplifier(llm), BackVerifier(llm, None, KNOWN))
    roles = [r for r, _ in llm.calls]
    assert out.status == BVStatus.PASS and out.customer_visible, out.status
    assert roles.count("simplify") == 1 and roles.count("fix") == 1
    assert [b.status for _, b in out.history] == [BVStatus.NEEDS_FIX, BVStatus.PASS]
    assert out.history[1][1].attempts == 1 and out.history[1][1].revised_text
    assert "직원 확인" not in out.to_dict()["customer_text"]


@case("E", "수정본도 문제 → BLOCK, 더 이상 수정 없음, 원문 대체 안내")
def _():
    llm = FakeLLM(simplify=BAD, fix=BAD)
    out = run_simplify(Q, CLAIMS, VERS, Simplifier(llm), BackVerifier(llm, None, KNOWN))
    assert out.status == BVStatus.BLOCK and not out.customer_visible
    assert [r for r, _ in llm.calls].count("fix") == 1
    assert out.fallback_lines == [c.text for c in CLAIMS[:3]]
    assert out.to_dict()["customer_text"] is None


@case("E", "생성 형식 오류 → SIMPLIFIER_ERROR → 즉시 BLOCK (수정 시도 없음)")
def _():
    llm = FakeLLM(simplify={"core": "문자열"}, fix=CLEAN)
    out = run_simplify(Q, CLAIMS, VERS, Simplifier(llm), BackVerifier(llm, None, KNOWN))
    assert out.status == BVStatus.BLOCK
    assert "fix" not in [r for r, _ in llm.calls]
    assert codes(out.history[0][1]) == ["SIMPLIFIER_ERROR"]


@case("E", "수정 호출 응답 없음 → SIMPLIFIER_ERROR → BLOCK")
def _():
    llm = FakeLLM(simplify=BAD, fix=None)
    out = run_simplify(Q, CLAIMS, VERS, Simplifier(llm), BackVerifier(llm, None, KNOWN))
    assert out.status == BVStatus.BLOCK and codes(out.history[1][1]) == ["SIMPLIFIER_ERROR"]


@case("E", "수정 호출에는 문제 목록이 들어간다")
def _():
    llm = FakeLLM(simplify=BAD, fix=CLEAN)
    run_simplify(Q, CLAIMS, VERS, Simplifier(llm), BackVerifier(llm, None, KNOWN))
    fix_user = [u for r, u in llm.calls if r == "fix"][0]
    assert "[발견된 문제]" in fix_user and "STRENGTHENED" in fix_user, fix_user[-300:]


@case("E", "PASS 가 아니면 고객 문구가 to_dict 에 나오지 않는다")
def _():
    llm = FakeLLM(simplify=BAD, fix=BAD)
    d = run_simplify(Q, CLAIMS, VERS, Simplifier(llm), BackVerifier(llm, None, KNOWN)).to_dict()
    assert d["customer_visible"] is False and d["customer_text"] is None


# ---------------- F 교란 생성기 ----------------
@case("F", "교란 10종 모두 고정 입력에서 생성된다")
def _():
    base = clean_expl()
    missing = [t for t in TYPES
               if all(perturb(base, t, s, products=["하나의정기예금"], known_products=KNOWN) is None
                      for s in range(5))]
    assert not missing, missing


@case("F", "교란은 원본을 바꾸지 않는다 (deepcopy)")
def _():
    base = clean_expl()
    before = base.customer_text()
    for t in TYPES:
        perturb(base, t, 0, products=["하나의정기예금"], known_products=KNOWN)
    assert base.customer_text() == before


@case("F", "같은 시드 → 같은 교란 (재현성)")
def _():
    base = clean_expl()
    a = perturb(base, "NUM_VALUE", 3)
    b = perturb(base, "NUM_VALUE", 3)
    assert a[1] == b[1] and a[0].customer_text() == b[0].customer_text()


@case("F", "조건 잔존 판단: 수치·핵심어")
def _():
    assert key_token("급여이체 실적이 있는 경우에 한해") == "급여이체"
    assert still_present("가입기간 6개월 이상 경과 후", "6개월 이상 지나면")
    assert not still_present("급여이체 실적이 있는 경우에 한해", "우대금리가 더해집니다")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    fails = [r for r in RESULTS if not r[2]]
    groups = sorted({r[0] for r in RESULTS})
    print("=" * 70)
    print(f"v2-B 쉬운 설명·역검증 테스트  {len(RESULTS) - len(fails)}/{len(RESULTS)} 통과")
    print("=" * 70)
    for g in groups:
        rs = [r for r in RESULTS if r[0] == g]
        print(f"  {g}  {sum(r[2] for r in rs)}/{len(rs)}")
    for g, name, ok, detail in RESULTS:
        if args.verbose or not ok:
            print(f"  {'✔' if ok else '✘'} [{g}] {name}" + (f"\n      {detail[:300]}" if detail else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
