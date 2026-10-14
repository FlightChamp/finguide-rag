"""Simplifier — 검증을 통과한 claim(OK)만 고객이 이해하기 쉬운 말로 옮긴다.

B 단계의 위치
------------
    ClaimVerifier → PolicyGate → [Simplifier] → BackVerifier → PolicyGate → 직원 확인

무엇을 LLM 에 맡기고 무엇을 맡기지 않는가
------------------------------------
    LLM   : OK claim 문장을 쉬운 문장으로 바꾸기 (core · conditions 블록)
    코드  : 어떤 claim 을 넣고 뺄지 (판정 기준), 직원 확인 영역 구성, 지켜야 할 조건 목록

    - CHECK claim 은 LLM 에 보내지도 않는다. 직원 확인 영역에 원문 그대로 둔다.
    - PROHIBITED claim 도 보내지 않는다. 직원 화면의 '제외' 목록에만 남는다.
    - 지켜야 할 조건 목록(RequiredCondition)은 claim 추출 단계의 conditions 슬롯에서 온다.
      쉬운 설명을 만드는 쪽이 목록을 정하면, 빠뜨린 조건은 목록에서도 같이 빠진다.

자동 수정은 fix() 1회뿐이다. 반복 루프는 없다 (policy.decide_bv_status).
B 는 틀린 답을 고치는 장치가 아니다. OK 로 판정된 내용의 '표현'만 바꾼다.
"""

from __future__ import annotations

import hashlib
import json
import re

from .schema import (
    BVCode,
    BVProblem,
    Claim,
    RequiredCondition,
    SimpleItem,
    SimplifiedExplanation,
    StaffCheckItem,
    Verdict,
    Verification,
)

# 기본값 sp-v2 (2026-10-15). sp-v1 은 재현용으로 남긴다.
#   sp-v1 개발 셋 51건 진단(65번):
#     - 확정 표현을 '~될 수 있습니다'로 약화한 문장 최소 3건 (b_d016, b_d019, b_q065)
#     - 조건 블록에 '환율이 상승하면', '개인사업자' 같은 조각 문구
#     - 문장당 어절 10.54 로 원문 그대로(Identity) 9.52 보다 오히려 길었다 — 쉬워지지 않았다
#   → 약화 금지, 조건 문구는 완결 문장, 짧은 문장(한 문장 한 사실)을 명시
#   이 수정은 개발 셋을 보고 한 것이다. 효과는 B held-out 으로만 주장한다.
SIMPLIFY_PROMPT_VERSION = "sp-v2"

_COMMON_HEAD = """너는 은행 직원이 확인을 마친 상담 내용을 고객이 이해하기 쉬운 말로 옮기는 작성자다.
[안내 가능 내용]의 문장들은 근거 문서와 대조가 끝난 것이다. 뜻을 바꾸지 말고 쉬운 말로만 옮긴다.

반드시 지킬 것
1. 새 정보를 더하지 마라. 입력에 없는 수치·상품·기관·조건·예시·권유·인사말을 쓰지 마라.
2. 수치는 입력 표기 그대로 쓴다. 값, 단위(%, %p, 원, 개월, 일, 영업일 등), '연/월',
   '이상/초과/이하/미만/이내/까지/부터', '최고/최대/최저/최소/기본/우대'를 바꾸거나 빼지 마라.
3. 상품명·서비스명·기관명은 입력 표기 그대로 쓴다. 다른 기관명을 쓰지 마라."""

_OUTPUT = """출력은 JSON 하나:
{"core": [{"text": "...", "claim_ids": ["c1"]}],
 "conditions": [{"text": "...", "claim_ids": ["c1"], "condition_ids": ["k1"]}]}"""

SIMPLIFY_SYSTEM_V1 = _COMMON_HEAD + """
4. 원문보다 단정적으로 말하지 마라. '~될 수 있습니다', '~에 한해', '원칙적으로' 같은 표현은 유지한다.
   '반드시·항상·무조건·누구나·언제든지' 같은 강조어를 새로 넣지 마라.
5. [지켜야 할 조건]의 조건은 하나도 빠짐없이 conditions 블록에 쉬운 말로 담는다.
   조건의 범위를 넓히거나 좁히지 마라. 조건 목록이 '(없음)'이면 conditions 는 [] 로 둔다.
6. 어려운 용어는 풀어 쓰되 원래 용어를 괄호로 남긴다.
   예: 만기 전에 해지할 때 적용되는 이율(중도해지이율)
7. 존댓말(~습니다/~입니다)로 쓰고, 한 문구는 한 문장으로 쓴다.
8. 각 문구에 그 문구가 옮긴 claim id 를 claim_ids 로 단다.
   conditions 문구에는 담은 조건 id 를 condition_ids 로 단다.
9. 모든 claim 을 core 에 한 번 이상 반영한다.

""" + _OUTPUT

SIMPLIFY_SYSTEM_V2 = _COMMON_HEAD + """
4. 원문과 같은 강도로 말한다.
   - 더 세게 말하지 마라: '반드시·항상·무조건·누구나·언제든지' 같은 강조어를 새로 넣지 마라.
   - 더 약하게 말하지도 마라: 원문이 '~됩니다', '~해야 합니다'처럼 확정이면 '~될 수 있습니다'로 바꾸지 마라.
     원문에 '~될 수 있습니다', '~에 한해', '원칙적으로'가 있으면 그대로 유지한다.
5. [지켜야 할 조건]의 조건은 하나도 빠짐없이 conditions 블록에 담는다. 핵심 설명에 이미 있어도 다시 쓴다.
   조건 문구는 그 조건이 무엇에 적용되는지 알 수 있는 완결된 문장으로 쓴다.
   예: '환율이 오르면 평가손실이 생깁니다.' (조각 '환율이 상승하면'은 안 된다)
   조건의 범위를 넓히거나 좁히지 마라. 조건 목록이 '(없음)'이면 conditions 는 [] 로 둔다.
6. 짧게 쓴다. 한 문장에는 한 가지 사실만 담고, 한 문장이 15어절을 넘지 않게 한다.
   원문 문장이 길면 두세 문장으로 나눠도 된다(나눈 문장은 같은 claim_ids 를 단다).
7. 어려운 용어는 쉬운 말로 풀고 원래 용어를 괄호로 남긴다.
   예: 만기 전에 해지할 때 적용되는 이율(중도해지이율)
8. 존댓말(~습니다/~입니다)로 쓴다. 모든 문구는 '~다.'로 끝나는 완결된 문장이다.
9. 각 문구에 그 문구가 옮긴 claim id 를 claim_ids 로 단다.
   conditions 문구에는 담은 조건 id 를 condition_ids 로 단다.
10. 모든 claim 을 core 에 한 번 이상 반영한다.

""" + _OUTPUT

SIMPLIFY_USER = "고객 질문:\n{question}\n\n[안내 가능 내용]\n{claims}\n\n[지켜야 할 조건]\n{conditions}"

_FIX_TAIL = """

이번에는 이전 결과를 고친다. [발견된 문제]를 모두 해소하고, 문제가 없는 문구는 그대로 둔다.
문제를 고치려고 새 정보를 더하지 마라. 원문 표현으로 되돌리는 것이 가장 안전하다."""

FIX_USER = SIMPLIFY_USER + "\n\n[이전 결과]\n{previous}\n\n[발견된 문제]\n{problems}"

# version → (simplify_system, fix_system)
PROMPTS = {
    "sp-v1": (SIMPLIFY_SYSTEM_V1, SIMPLIFY_SYSTEM_V1 + _FIX_TAIL),
    "sp-v2": (SIMPLIFY_SYSTEM_V2, SIMPLIFY_SYSTEM_V2 + _FIX_TAIL),
}
# 하위 호환 (테스트·다른 스크립트가 이 이름을 쓴다): 기본 버전의 프롬프트
SIMPLIFY_SYSTEM, FIX_SYSTEM = PROMPTS[SIMPLIFY_PROMPT_VERSION]


def simplify_prompt_hash(version: str = SIMPLIFY_PROMPT_VERSION) -> str:
    sys_p, fix_p = PROMPTS[version]
    raw = "|".join([version, sys_p, SIMPLIFY_USER, fix_p, FIX_USER])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def partition(claims: list[Claim], verifications: list[Verification]):
    """판정별로 claim 을 나눈다. 판정이 없는 claim 은 CHECK 로 본다 (fail-closed)."""
    verdict = {v.claim_id: v for v in verifications}
    ok, check, prohibited = [], [], []
    for c in claims:
        v = verdict.get(c.claim_id)
        vd = v.verdict if v is not None else Verdict.CHECK
        (ok if vd == Verdict.OK else prohibited if vd == Verdict.PROHIBITED else check).append(c)
    return ok, check, prohibited, verdict


def required_conditions(ok_claims: list[Claim]) -> list[RequiredCondition]:
    out: list[RequiredCondition] = []
    for c in ok_claims:
        seen: set[str] = set()
        for t in c.conditions:
            t = _norm(t)
            key = re.sub(r"\s+", "", t)
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            out.append(RequiredCondition(f"k{len(out) + 1}", t, c.claim_id))
    return out


def staff_check_items(check_claims: list[Claim], verdict: dict[str, Verification]) -> list[StaffCheckItem]:
    items = []
    for c in check_claims:
        v = verdict.get(c.claim_id)
        reasons = [h.code.value for h in (v.reasons if v else []) if h.code.value != "SUPPORTED"]
        items.append(StaffCheckItem(c.claim_id, c.text, reasons or ["NO_VERDICT"]))
    return items


def parse_items(data, cond_ids: set[str]) -> tuple[list[SimpleItem], list[SimpleItem], str | None]:
    """LLM JSON → (core, conditions, 오류). 형식이 틀리면 오류 문자열."""
    if not isinstance(data, dict):
        return [], [], "응답 없음 또는 JSON 아님"
    core_raw, cond_raw = data.get("core"), data.get("conditions", [])
    if not isinstance(core_raw, list) or not core_raw:
        return [], [], "core 블록 없음"
    if not isinstance(cond_raw, list):
        return [], [], "conditions 블록 형식 오류"

    def conv(raw: list, prefix: str, with_cond: bool) -> list[SimpleItem] | None:
        out = []
        for r in raw:
            if not isinstance(r, dict):
                return None
            text = _norm(r.get("text", ""))
            if not text:
                continue
            ids = r.get("claim_ids", [])
            ids = [str(x) for x in ids] if isinstance(ids, list) else []
            cids = r.get("condition_ids", []) if with_cond else []
            cids = [str(x) for x in cids if str(x) in cond_ids] if isinstance(cids, list) else []
            out.append(SimpleItem(f"{prefix}{len(out) + 1}", text, ids, cids))
        return out

    core = conv(core_raw, "s", False)
    conds = conv(cond_raw, "n", True)
    if core is None or conds is None:
        return [], [], "문구 형식 오류"
    if not core:
        return [], [], "core 문구가 모두 비어 있음"
    return core, conds, None


class Simplifier:
    def __init__(self, llm, version: str = SIMPLIFY_PROMPT_VERSION):
        """llm: claim_verifier.CachedLLM (디스크 캐시 → 같은 입력이면 같은 출력)."""
        if version not in PROMPTS:
            raise ValueError(f"알 수 없는 Simplifier 프롬프트: {version}")
        self.llm = llm
        self.version = version
        self.system, self.fix_system = PROMPTS[version]
        self.tag = f"{version}:{simplify_prompt_hash(version)}"

    # ---------- 입력 구성 ----------
    @staticmethod
    def _claims_block(ok_claims: list[Claim]) -> str:
        return "\n".join(f"{c.claim_id}: {c.text}" for c in ok_claims)

    @staticmethod
    def _conds_block(conds: list[RequiredCondition]) -> str:
        return "\n".join(f"{k.cond_id} ({k.claim_id}): {k.text}" for k in conds) or "(없음)"

    def skeleton(self, claims: list[Claim], verifications: list[Verification]):
        """LLM 없이 정해지는 부분: 직원 확인 영역, 제외 목록, 지켜야 할 조건."""
        ok, check, proh, verdict = partition(claims, verifications)
        expl = SimplifiedExplanation(
            staff_check=staff_check_items(check, verdict),
            excluded=[c.claim_id for c in proh],
            required_conditions=required_conditions(ok),
            prompt_hash=self.tag,
        )
        return expl, ok

    # ---------- 생성 ----------
    def simplify(self, question: str, claims: list[Claim],
                 verifications: list[Verification]) -> tuple[SimplifiedExplanation, list[BVProblem], list[Claim]]:
        expl, ok = self.skeleton(claims, verifications)
        if not ok:
            return expl, [], ok
        user = SIMPLIFY_USER.format(question=question, claims=self._claims_block(ok),
                                    conditions=self._conds_block(expl.required_conditions))
        data = self.llm.json(self.system, user, max_tokens=1500, tag=self.tag)
        return self._fill(expl, data, ok)

    def fix(self, question: str, previous: SimplifiedExplanation, ok_claims: list[Claim],
            problems: list[BVProblem]) -> tuple[SimplifiedExplanation, list[BVProblem]]:
        prev_json = json.dumps({
            "core": [{"text": i.text, "claim_ids": i.claim_ids} for i in previous.core],
            "conditions": [{"text": i.text, "claim_ids": i.claim_ids, "condition_ids": i.condition_ids}
                           for i in previous.conditions],
        }, ensure_ascii=False)
        probs = "\n".join(f"- [{p.code.value}] {p.detail}" + (f" (문구 {p.item_id})" if p.item_id else "")
                          for p in problems)
        user = FIX_USER.format(question=question, claims=self._claims_block(ok_claims),
                               conditions=self._conds_block(previous.required_conditions),
                               previous=prev_json, problems=probs)
        data = self.llm.json(self.fix_system, user, max_tokens=1500, tag=self.tag + ":fix")
        expl = SimplifiedExplanation(staff_check=previous.staff_check, excluded=previous.excluded,
                                     required_conditions=previous.required_conditions,
                                     prompt_hash=self.tag + ":fix")
        expl, gen, _ = self._fill(expl, data, ok_claims)
        return expl, gen

    @staticmethod
    def _fill(expl: SimplifiedExplanation, data, ok: list[Claim]):
        core, conds, err = parse_items(data, {k.cond_id for k in expl.required_conditions})
        if err:
            return expl, [BVProblem(BVCode.SIMPLIFIER_ERROR, err)], ok
        expl.core, expl.conditions = core, conds
        return expl, [], ok


__all__ = ["Simplifier", "SIMPLIFY_PROMPT_VERSION", "PROMPTS", "simplify_prompt_hash", "partition",
           "required_conditions", "parse_items"]
