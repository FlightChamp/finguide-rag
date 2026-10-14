"""BackVerifier — 쉬운 설명이 OK claim 원문의 뜻을 그대로 담았는지 다시 확인한다.

비교 대상
--------
    쉬운 설명(core + conditions)  ↔  OK claim 원문 (근거 문서가 아니다)
    OK claim 은 이미 근거와 대조를 마쳤으므로, 여기서는 '옮기는 동안 뜻이 바뀌었는가'만 본다.

검사 구성
--------
    결정적 규칙 (LLM 없음)
        수치 보존       numeric.extract_numerics / compare_slot 재사용 (값·단위·기간·비교어·상한)
        상품 보존       원문 상품명 누락 / 원문에 없는 카탈로그 상품명 추가
        기관 보존       원문·근거에 없는 기관명 추가 (rules.detect_institutions 재사용)
        단정 강화       강조어 추가 (rules 와 같은 강조어·동의어 표) / 가능성·한정 표현 소실
        구조            출처 claim id 검사, CHECK·PROHIBITED 내용 유입, OK claim 누락, 직원 확인 항목 누락
    LLM 판단 + 코드 판정
        조건·예외 보존  조건 목록(RequiredCondition)마다 PRESERVED / DROPPED / ALTERED 를 LLM 이 고르고
                        '어느 문구에 있는지(where)'를 문구 id 로 대게 한다. 판정은 코드가 한다.
                        PRESERVED 인데 위치를 못 대면 CONDITION_UNVERIFIED (보존으로 치지 않음).

판정 (policy.decide_bv_status)
    문제 없음 → PASS / 고칠 수 없는 문제 → BLOCK / 그 외 → NEEDS_FIX 1회 → 재검증 실패 시 BLOCK
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .numeric import MatchStatus, compare_slot, extract_numerics
from .policy import decide_bv_status
from .rules import (
    _CLAIM_ANY_POSSIBILITY,
    _SOFT_HEDGES,
    HEDGE_GROUPS,
    _added_intensifiers,
    _hedges,
    detect_institutions,
)
from .schema import (
    BackVerification,
    BVCode,
    BVProblem,
    BVStatus,
    Claim,
    SimplifiedExplanation,
    Verdict,
)
from .simplify import Simplifier, partition

BV_PROMPT_VERSION = "bv-v1"

COND_SYSTEM = """너는 쉬운 설명이 원문의 조건·예외를 보존했는지 확인하는 검토자다.
판정 라벨(안내 가능 등)은 쓰지 않는다. [지켜야 할 조건]의 조건마다 다음 중 하나를 고른다.

  PRESERVED  쉬운 설명에 같은 뜻의 조건이 있다. 표현은 달라도 된다. where 에 그 문구 id 를 적는다.
  ALTERED    조건이 있지만 범위·대상·방향이 달라졌다 (넓어짐, 좁아짐, 반대). where 에 그 문구 id 를 적는다.
  DROPPED    쉬운 설명 어디에도 이 조건이 없다. where 는 [].

또, [원문]의 조건·예외·한정 표현 가운데 [지켜야 할 조건]에 없는데 쉬운 설명에서 빠지거나 바뀐 것이 있으면
extra 에 원문 표기 그대로(source_quote) 적는다. 없으면 [].

출력은 JSON 하나:
{"results": [{"cond_id": "k1", "status": "PRESERVED", "where": ["s2"]}],
 "extra": [{"source_quote": "...", "claim_id": "c1"}]}"""

COND_USER = "[원문 — 안내 가능 claim]\n{claims}\n\n[지켜야 할 조건]\n{conds}\n\n[쉬운 설명]\n{items}"

_NUMERIC_OK = {MatchStatus.MATCH}
_NUM_CODE = {
    MatchStatus.NOT_FOUND: BVCode.NUMERIC_ADDED,
    MatchStatus.BOUND_DROPPED: BVCode.STRENGTHENED,
}
# 한정 묶음의 '쉬운 말' 표현 — 근거 쪽 '에 한해'를 '~인 경우만'으로 옮긴 것은 보존으로 본다
_LIMIT_LENIENT = re.compile(r"에\s*한(하|해|함|하여)|경우에만|경우만|에만|만\s*해당|한정|해당하는|대상")
_LEAK_MIN, _LEAK_MARGIN = 0.6, 0.25


def bv_prompt_hash(version: str = BV_PROMPT_VERSION) -> str:
    return hashlib.sha256("|".join([version, COND_SYSTEM, COND_USER]).encode("utf-8")).hexdigest()[:12]


def _ns(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _bigrams(t: str) -> set[str]:
    t = _ns(t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _contain(a: set[str], b: set[str]) -> float:
    """a 의 2-gram 중 b 에 있는 비율."""
    return len(a & b) / len(a) if a else 0.0


@dataclass
class BVContext:
    claims: dict[str, Claim]
    verdicts: dict[str, Verdict]
    evidence_institutions: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, claims: list[Claim], verifications, evidence_institutions=None) -> "BVContext":
        ok, check, proh, verdict = partition(claims, verifications)
        vd = {c.claim_id: Verdict.OK for c in ok}
        vd.update({c.claim_id: Verdict.CHECK for c in check})
        vd.update({c.claim_id: Verdict.PROHIBITED for c in proh})
        inst = set(evidence_institutions or ())
        if not inst:
            inst = {e.institution for v in verifications for e in v.evidence if e.institution}
        return cls({c.claim_id: c for c in claims}, vd, inst)

    def ids(self, verdict: Verdict) -> list[str]:
        return [i for i, v in self.verdicts.items() if v == verdict]

    def ok_claims(self) -> list[Claim]:
        return [self.claims[i] for i in self.ids(Verdict.OK)]


class BackVerifier:
    def __init__(self, llm=None, matcher=None, known_products: list[str] | None = None,
                 condition_check: bool = True):
        """llm: CachedLLM. condition_check=False 는 '규칙만' 평가용 (운영 경로에서는 끄지 않는다).

        condition_check=True 인데 LLM 응답이 없으면 BV_ERROR → BLOCK (fail-closed).
        """
        self.llm, self.matcher = llm, matcher
        self.known_products = list(known_products or [])
        self.condition_check = condition_check
        self.tag = f"{BV_PROMPT_VERSION}:{bv_prompt_hash()}"

    # ==============================================================
    # 결정적 규칙
    # ==============================================================

    def _source_text(self, item, ctx: BVContext) -> str:
        cited = [ctx.claims[i].text for i in item.claim_ids
                 if ctx.verdicts.get(i) == Verdict.OK]
        return "\n".join(cited) if cited else "\n".join(c.text for c in ctx.ok_claims())

    def check_numeric(self, expl: SimplifiedExplanation, ctx: BVContext) -> list[BVProblem]:
        out: list[BVProblem] = []
        src_all = "\n".join(c.text for c in ctx.ok_claims())
        src_all_slots = extract_numerics(src_all)
        cust_slots = []
        for it in expl.customer_items():
            slots = extract_numerics(it.text)
            cust_slots += slots
            item_src_slots = extract_numerics(self._source_text(it, ctx))
            for s in slots:
                st, det = compare_slot(s, item_src_slots)
                if st in _NUMERIC_OK:
                    continue
                if compare_slot(s, src_all_slots)[0] in _NUMERIC_OK:
                    continue   # 다른 OK claim 의 수치를 옮긴 것 — 인용 id 만 어긋남
                out.append(BVProblem(_NUM_CODE.get(st, BVCode.NUMERIC_CHANGED),
                                     f"{st.value}: {det}", it.item_id))
        for s in src_all_slots:
            st, det = compare_slot(s, cust_slots)
            if st in (MatchStatus.NOT_FOUND, MatchStatus.VALUE_MISMATCH):
                out.append(BVProblem(BVCode.NUMERIC_DROPPED, f"원문 '{s.raw}' 이(가) 쉬운 설명에 없음"))
        return out

    def _informative(self, name: str) -> bool:
        return bool(self.matcher.informative(name)) if self.matcher is not None else bool(name.strip())

    def _same(self, a: str, b: str) -> bool:
        if self.matcher is not None and self.matcher.same_product(a, b):
            return True
        return _ns(a) == _ns(b)

    def check_product(self, expl: SimplifiedExplanation, ctx: BVContext) -> list[BVProblem]:
        out: list[BVProblem] = []
        src_all = _ns("\n".join(c.text for c in ctx.ok_claims()))
        cust = _ns(expl.customer_text())
        src_products: list[str] = []
        for c in ctx.ok_claims():
            if c.product and self._informative(c.product) and _ns(c.product) in src_all:
                src_products.append(c.product)
        for k in self.known_products:
            if self._informative(k) and _ns(k) in src_all and k not in src_products:
                src_products.append(k)
        for p in src_products:
            core = re.sub(r"(상품|서비스|통장)$", "", _ns(p))
            if _ns(p) not in cust and not (core and core in cust):
                out.append(BVProblem(BVCode.PRODUCT_DROPPED, f"원문 상품명 '{p}' 없음"))
        for k in self.known_products:
            nk = _ns(k)
            if nk and self._informative(k) and nk in cust and nk not in src_all \
                    and not any(self._same(k, p) for p in src_products):
                out.append(BVProblem(BVCode.PRODUCT_ADDED, f"원문에 없는 상품명 '{k}'"))
        return out

    def check_institution(self, expl: SimplifiedExplanation, ctx: BVContext) -> list[BVProblem]:
        src_all = "\n".join(c.text for c in ctx.ok_claims())
        allowed = detect_institutions(src_all) | set(ctx.evidence_institutions)
        out = []
        for it in expl.customer_items():
            extra = sorted(detect_institutions(it.text) - allowed)
            if extra:
                out.append(BVProblem(BVCode.INSTITUTION_ADDED,
                                     f"{extra} (허용: {sorted(allowed) or '없음'})", it.item_id))
        return out

    def check_assertiveness(self, expl: SimplifiedExplanation, ctx: BVContext) -> list[BVProblem]:
        out: list[BVProblem] = []
        for it in expl.customer_items():
            added = _added_intensifiers(it.text, self._source_text(it, ctx))
            if added:
                out.append(BVProblem(BVCode.STRENGTHENED, f"원문에 없는 강조어 {added}", it.item_id))
        # 헤지는 claim 단위로 본다: 한 claim 을 두 문구로 나눴을 때 한쪽에만 헤지가 있어도 정상이다.
        for c in ctx.ok_claims():
            groups = _hedges(c.text)
            if not groups:
                continue
            joined = " ".join(it.text for it in expl.customer_items() if c.claim_id in it.claim_ids)
            if not joined:
                continue   # 누락은 CLAIM_UNCOVERED 가 잡는다
            lost = []
            if "가능성" in groups and not (HEDGE_GROUPS["가능성"].search(joined)
                                          or _CLAIM_ANY_POSSIBILITY.search(joined)):
                lost.append("가능성")
            if "한정" in groups and not (_LIMIT_LENIENT.search(joined) or _SOFT_HEDGES.search(joined)):
                lost.append("한정")
            if lost:
                out.append(BVProblem(BVCode.HEDGE_LOST, f"{c.claim_id} 의 {lost} 표현 소실"))
        return out

    def check_structure(self, expl: SimplifiedExplanation, ctx: BVContext) -> list[BVProblem]:
        out: list[BVProblem] = []
        ok_ids = set(ctx.ids(Verdict.OK))
        items = expl.customer_items()
        for it in items:
            if not it.claim_ids:
                out.append(BVProblem(BVCode.UNKNOWN_SOURCE, "출처 claim id 없음", it.item_id))
            for cid in it.claim_ids:
                v = ctx.verdicts.get(cid)
                if v is None:
                    out.append(BVProblem(BVCode.UNKNOWN_SOURCE, f"없는 claim id '{cid}'", it.item_id))
                elif v == Verdict.CHECK:
                    out.append(BVProblem(BVCode.CHECK_LEAK, f"확인 필요 claim {cid} 를 출처로 댐", it.item_id))
                elif v == Verdict.PROHIBITED:
                    out.append(BVProblem(BVCode.PROHIBITED_LEAK, f"안내 금지 claim {cid} 를 출처로 댐",
                                         it.item_id))
        # 내용 유입: 문구가 OK claim 보다 CHECK·PROHIBITED claim 원문에 훨씬 가까우면
        ok_grams = _bigrams("\n".join(c.text for c in ctx.ok_claims()))
        for it in items:
            ig = _bigrams(it.text)
            in_ok = _contain(ig, ok_grams)
            for cid, v in ctx.verdicts.items():
                if v == Verdict.OK:
                    continue
                in_x = _contain(ig, _bigrams(ctx.claims[cid].text))
                if in_x >= _LEAK_MIN and in_x >= in_ok + _LEAK_MARGIN:
                    code = BVCode.CHECK_LEAK if v == Verdict.CHECK else BVCode.PROHIBITED_LEAK
                    out.append(BVProblem(code, f"{cid} 원문과 겹침 {in_x:.2f} (OK 원문 {in_ok:.2f})",
                                         it.item_id))
        cited = {cid for it in items for cid in it.claim_ids}
        for cid in sorted(ok_ids - cited):
            out.append(BVProblem(BVCode.CLAIM_UNCOVERED, f"OK claim {cid} 가 반영되지 않음"))
        staff = {s.claim_id for s in expl.staff_check}
        for cid in ctx.ids(Verdict.CHECK):
            if cid not in staff:
                out.append(BVProblem(BVCode.LOST_CHECK, f"확인 필요 claim {cid} 가 직원 확인 영역에 없음"))
        for cid in ctx.ids(Verdict.PROHIBITED):
            if cid not in expl.excluded:
                out.append(BVProblem(BVCode.LOST_CHECK, f"안내 금지 claim {cid} 가 제외 목록에 없음"))
        return out

    def check_rules(self, expl: SimplifiedExplanation, ctx: BVContext) -> list[BVProblem]:
        return (self.check_structure(expl, ctx) + self.check_numeric(expl, ctx)
                + self.check_product(expl, ctx) + self.check_institution(expl, ctx)
                + self.check_assertiveness(expl, ctx))

    # ==============================================================
    # 조건·예외 보존 (LLM 판단 → 코드 판정)
    # ==============================================================

    def check_conditions(self, expl: SimplifiedExplanation, ctx: BVContext
                         ) -> tuple[list[BVProblem], dict[str, str]]:
        conds = expl.required_conditions
        claims_txt = "\n".join(f"{c.claim_id}: {c.text}" for c in ctx.ok_claims())
        conds_txt = "\n".join(f"{k.cond_id} ({k.claim_id}): {k.text}" for k in conds) or "(없음)"
        items_txt = "\n".join(f"{it.item_id}: {it.text}" for it in expl.customer_items())
        data = self.llm.json(COND_SYSTEM, COND_USER.format(claims=claims_txt, conds=conds_txt,
                                                           items=items_txt),
                             max_tokens=1200, tag=self.tag) if self.llm is not None else None
        if not isinstance(data, dict) or not isinstance(data.get("results", []), list):
            return [BVProblem(BVCode.BV_ERROR, "조건 보존 검사 응답 없음·형식 오류")], {}

        item_ids = {it.item_id for it in expl.customer_items()}
        by_id = {str(r.get("cond_id")): r for r in data.get("results", []) if isinstance(r, dict)}
        problems: list[BVProblem] = []
        results: dict[str, str] = {}
        for k in conds:
            r = by_id.get(k.cond_id)
            if r is None:
                results[k.cond_id] = "NO_RESULT"
                problems.append(BVProblem(BVCode.CONDITION_UNVERIFIED, f"{k.cond_id} '{k.text}' 판단 누락"))
                continue
            st = str(r.get("status", "")).strip().upper()
            where = [str(w) for w in (r.get("where") or []) if str(w) in item_ids]
            results[k.cond_id] = st if st in ("PRESERVED", "DROPPED", "ALTERED") else "INVALID"
            if st == "PRESERVED" and where:
                continue
            if st == "PRESERVED":
                problems.append(BVProblem(BVCode.CONDITION_UNVERIFIED,
                                          f"{k.cond_id} '{k.text}' 보존이라 했으나 위치 없음"))
            elif st == "DROPPED":
                problems.append(BVProblem(BVCode.CONDITION_DROPPED, f"{k.cond_id} '{k.text}'"))
            elif st == "ALTERED":
                problems.append(BVProblem(BVCode.CONDITION_ALTERED, f"{k.cond_id} '{k.text}'",
                                          where[0] if where else ""))
            else:
                problems.append(BVProblem(BVCode.CONDITION_UNVERIFIED, f"{k.cond_id} 상태값 '{st}'"))

        # 목록 밖 조건: 원문 claim 에 실제로 있는 표현일 때만 받는다 (LLM 이 지어낸 조건은 버린다)
        src = {c.claim_id: _ns(c.text) for c in ctx.ok_claims()}
        for n, e in enumerate(data.get("extra") or [], 1):
            if not isinstance(e, dict):
                continue
            q = _ns(str(e.get("source_quote", "")))
            if len(q) >= 2 and any(q in s for s in src.values()):
                results[f"extra{n}"] = "DROPPED"
                problems.append(BVProblem(BVCode.CONDITION_DROPPED, f"(목록 외) '{e.get('source_quote')}'"))
            else:
                results[f"extra{n}"] = "REJECTED"   # 원문에 없는 인용 — 판정에 쓰지 않음
        return problems, results

    # ==============================================================
    # 전체
    # ==============================================================

    def verify(self, expl: SimplifiedExplanation, ctx: BVContext, attempts: int = 0,
               generation_problems: list[BVProblem] | None = None) -> BackVerification:
        problems = list(generation_problems or [])
        results: dict[str, str] = {}
        if not any(p.code == BVCode.SIMPLIFIER_ERROR for p in problems):
            problems += self.check_rules(expl, ctx)
            if self.condition_check:
                cp, results = self.check_conditions(expl, ctx)
                problems += cp
        status = decide_bv_status(problems, attempts)
        return summarize(status, problems, results, attempts, expl)


def summarize(status: BVStatus, problems: list[BVProblem], results: dict[str, str],
              attempts: int, expl: SimplifiedExplanation) -> BackVerification:
    def pick(*codes):
        return [p.detail for p in problems if p.code in codes]
    return BackVerification(
        status=status,
        dropped_conditions=pick(BVCode.CONDITION_DROPPED, BVCode.CONDITION_ALTERED,
                                BVCode.CONDITION_UNVERIFIED),
        numeric_diff=pick(BVCode.NUMERIC_CHANGED, BVCode.NUMERIC_ADDED, BVCode.NUMERIC_DROPPED),
        strengthened_phrases=pick(BVCode.STRENGTHENED, BVCode.HEDGE_LOST),
        lost_check_items=pick(BVCode.LOST_CHECK, BVCode.CHECK_LEAK, BVCode.PROHIBITED_LEAK),
        revised_text=expl.customer_text() if attempts > 0 else None,
        attempts=attempts,
        problems=problems,
        condition_results=results,
        customer_visible=(status == BVStatus.PASS),
    )


# ==================================================================
# 파이프라인: 쉬운 설명 → 역검증 → (1회 수정 → 재검증)
# ==================================================================


@dataclass
class SimplifyOutcome:
    explanation: SimplifiedExplanation          # 마지막 버전 (PASS 면 고객에게 보일 수 있는 것)
    status: BVStatus | None                     # None = 쉬운 설명 대상 없음 (OK claim 0개)
    history: list[tuple[SimplifiedExplanation, BackVerification]] = field(default_factory=list)
    skipped_reason: str = ""
    fallback_lines: list[str] = field(default_factory=list)   # BLOCK 시 직원에게 보일 OK claim 원문

    @property
    def customer_visible(self) -> bool:
        return self.status == BVStatus.PASS

    def to_dict(self) -> dict:
        return {"status": self.status.value if self.status else None,
                "skipped_reason": self.skipped_reason,
                "customer_visible": self.customer_visible,
                "customer_text": self.explanation.customer_text() if self.customer_visible else None,
                "explanation": self.explanation.to_dict(),
                "history": [{"explanation": e.to_dict(), "bv": b.to_dict()} for e, b in self.history],
                "fallback_lines": self.fallback_lines}


def run_simplify(question: str, claims: list[Claim], verifications, simplifier: Simplifier,
                 verifier: BackVerifier, evidence_institutions=None) -> SimplifyOutcome:
    """검증 결과 → 쉬운 설명 → 역검증. 자동 수정은 최대 1회."""
    ctx = BVContext.build(claims, verifications, evidence_institutions)
    expl, gen, ok = simplifier.simplify(question, claims, verifications)
    if not ok:
        return SimplifyOutcome(expl, None, skipped_reason="no_ok_claims")
    bv = verifier.verify(expl, ctx, attempts=0, generation_problems=gen)
    history = [(expl, bv)]
    if bv.status == BVStatus.NEEDS_FIX:
        expl2, gen2 = simplifier.fix(question, expl, ok, bv.problems)
        bv2 = verifier.verify(expl2, ctx, attempts=1, generation_problems=gen2)
        history.append((expl2, bv2))
        expl, bv = expl2, bv2
    if bv.status == BVStatus.NEEDS_FIX:   # 방어: 수정 소진 후 NEEDS_FIX 는 있을 수 없다
        bv.status, bv.customer_visible = BVStatus.BLOCK, False
    fallback = [c.text for c in ok] if bv.status == BVStatus.BLOCK else []
    return SimplifyOutcome(expl, bv.status, history, fallback_lines=fallback)


def explanation_from_json(d: dict) -> SimplifiedExplanation:
    """저장된 SimplifiedExplanation.to_dict() → 객체 (평가·교란 스크립트용)."""
    from .schema import RequiredCondition, SimpleItem, StaffCheckItem
    return SimplifiedExplanation(
        core=[SimpleItem(**i) for i in d.get("core", [])],
        conditions=[SimpleItem(**i) for i in d.get("conditions", [])],
        staff_check=[StaffCheckItem(**i) for i in d.get("staff_check", [])],
        excluded=list(d.get("excluded", [])),
        required_conditions=[RequiredCondition(**k) for k in d.get("required_conditions", [])],
        prompt_hash=d.get("prompt_hash", ""),
    )


def claims_from_json(rows: list[dict]) -> list[Claim]:
    from .schema import NumericSlot
    out = []
    for r in rows:
        r = dict(r)
        r["numerics"] = [NumericSlot(**{**n, "span": tuple(n.get("span", (0, 0)))}) for n in r.get("numerics", [])]
        r["span"] = tuple(r.get("span", (0, 0)))
        out.append(Claim(**r))
    return out


__all__ = ["BackVerifier", "BVContext", "SimplifyOutcome", "run_simplify", "bv_prompt_hash",
           "BV_PROMPT_VERSION", "explanation_from_json", "claims_from_json", "summarize"]

