"""역검증(B) 평가용 교란 — 정상 쉬운 설명에 '뜻이 바뀌는' 오류를 하나씩 넣는다.

정답을 구성으로 안다: 교란본은 모두 '고객에게 보이면 안 되는' 설명이다.
역검증이 교란본을 PASS 시키면 False Pass 다.

교란은 문구 텍스트만 바꾸고 claim_ids · condition_ids 메타데이터는 그대로 둔다.
'내용은 틀렸는데 출처 표시는 그럴듯한' 생성 결과를 흉내 낸 것이다.

유형 (기대 탐지 경로)
    NUM_VALUE        수치 값 변경                   규칙 (NUMERIC_*)
    NUM_UNIT         % ↔ %p, 개월 → 년, 일 → 영업일    규칙
    NUM_COMPARATOR   이상→초과, 이하→미만, 이내→미만, 미만→이하   규칙
    BOUND_DROP       '최고/최대/최저/최소' 삭제          규칙 (STRENGTHENED)
    PRODUCT_SWAP     원문 상품명 → 다른 카탈로그 상품명    규칙 (PRODUCT_*)
    INSTITUTION      하나은행 → 다른 은행 / 다른 은행명 삽입  규칙 (INSTITUTION_ADDED)
    INTENSIFIER      서술어 앞에 강조어 삽입              규칙 (STRENGTHENED)
    HEDGE_DROP       '~될 수 있습니다' → '~됩니다'        규칙 (HEDGE_LOST)
    CONDITION_DROP   조건 블록에서 한 조건을 담은 문구 삭제   LLM (CONDITION_DROPPED)
                     ※ 64번 진단(2026-10-15): 놓친 4건 모두 조건이 핵심 설명에 다른 말로 남아 있었다(라벨 잡음).
                       조건 보존 평가는 66번(LLM 생성 + 사람 확인 교란)을 주 지표로 쓰고, 이 유형은 참고로만 본다.
    CHECK_LEAK       CHECK claim 원문을 핵심 설명 문구로 추가  규칙 (CHECK_LEAK)

알려진 한계
    CONDITION_DROP 은 문구를 지워도 같은 조건이 다른 문구에 다른 말로 남아 있을 수 있다.
    조건의 수치나 가장 긴 내용어가 남은 문구에 있으면 교란을 만들지 않지만(still_present),
    완전히 다른 말로 남은 경우는 거르지 못한다.
    이 유형의 '놓침'은 사람이 확인해야 진짜 놓침인지 알 수 있다 (64번 보고서에 목록으로 남긴다).
"""

from __future__ import annotations

import copy
import random
import re

from .numeric import extract_numerics
from .rules import HEDGE_GROUPS, INTENSIFIER_EQUIV, detect_institutions
from .schema import SimpleItem, SimplifiedExplanation

TYPES = ["NUM_VALUE", "NUM_UNIT", "NUM_COMPARATOR", "BOUND_DROP", "PRODUCT_SWAP",
         "INSTITUTION", "INTENSIFIER", "HEDGE_DROP", "CONDITION_DROP", "CHECK_LEAK"]

_OTHER_BANKS = ["국민은행", "신한은행", "우리은행", "농협은행"]
_INTENS = ["반드시", "항상", "무조건", "누구나"]
_HEDGE_VERB = {
    "될": "됩니다", "생길": "생깁니다", "발생할": "발생합니다", "있을": "있습니다", "달라질": "달라집니다",
    "변동될": "변동됩니다", "변경될": "변경됩니다", "부과될": "부과됩니다", "제한될": "제한됩니다",
    "않을": "않습니다", "감소할": "감소합니다", "지연될": "지연됩니다", "하락할": "하락합니다",
    "상승할": "상승합니다", "거절될": "거절됩니다", "취소될": "취소됩니다", "해지될": "해지됩니다",
}
_HEDGE_RE = re.compile(r"(" + "|".join(sorted(_HEDGE_VERB, key=len, reverse=True))
                       + r")\s*수\s*(?:도\s*)?있습니다")
_HEDGE_ADVERB = re.compile(r"(경우에\s*따라|원칙적으로|일반적으로)\s*,?\s*")
_COMP_SWAP = [("이상", "초과"), ("이하", "미만"), ("이내", "미만"), ("미만", "이하"), ("초과", "이상")]
_BOUND_RE = re.compile(r"(최고|최대|최저|최소)\s*")
_ENDING = re.compile(r"(\S+)(니다|세요|어요|아요|해요)([.!]?)\s*$")


def _items(expl: SimplifiedExplanation):
    return expl.core + expl.conditions


def _ns(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _bump(raw_num: str, rng: random.Random) -> str:
    n = raw_num.replace(",", "")
    if "." in n:
        dec = len(n.split(".")[1])
        v = float(n) + (0.5 if rng.random() < 0.5 else -0.1) * (1 if float(n) > 0.5 else 0.2)
        return f"{max(v, 0.01):.{dec}f}"
    v = int(n)
    nv = v * 2 if v < 10 else v + (v // 10 or 1)
    return f"{nv:,}" if "," in raw_num else str(nv)


def _num_value(e, rng, **_):
    cands = [(it, s) for it in _items(e) for s in extract_numerics(it.text) if s.unit != "date"]
    if not cands:
        return None
    it, s = rng.choice(cands)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s.raw)
    if not m:
        return None
    new_raw = s.raw[:m.start()] + _bump(m.group(0), rng) + s.raw[m.end():]
    if s.raw not in it.text:
        return None
    it.text = it.text.replace(s.raw, new_raw, 1)
    return f"{it.item_id}: '{s.raw}' → '{new_raw}'"


def _num_unit(e, rng, **_):
    rules = [(r"(\d)\s*%(?!p)", r"\1%p"), (r"(\d)\s*%p", r"\1%"), (r"(\d)\s*개월", r"\1년"),
             (r"(\d)\s*일(?!시|부|자|반|정)", r"\1영업일")]
    cands = [(it, pat, rep) for it in _items(e) for pat, rep in rules if re.search(pat, it.text)]
    if not cands:
        return None
    it, pat, rep = rng.choice(cands)
    before = it.text
    it.text = re.sub(pat, rep, it.text, count=1)
    return f"{it.item_id}: {pat} → {rep}" if it.text != before else None


def _num_comparator(e, rng, **_):
    cands = []
    for it in _items(e):
        for a, b in _COMP_SWAP:
            if re.search(r"\d[\d,.]*\s*\S{0,3}\s*" + a, it.text):
                cands.append((it, a, b))
    if not cands:
        return None
    it, a, b = rng.choice(cands)
    it.text = it.text.replace(a, b, 1)
    return f"{it.item_id}: '{a}' → '{b}'"


def _is_bound(s) -> bool:
    return s.comparator is None and any(q in (s.qualifier or "") for q in ("최고", "최대", "최저", "최소"))


def _bound_drop(e, rng, **_):
    """'최고 4.0%' → '4.0%'. 단, 수치 뒤에 '까지·이하·이내' 같은 비교어가 남으면 뜻이 같으므로 만들지 않는다
    ('최고 50%까지' → '50%까지' 는 오류가 아니다)."""
    cands = [it for it in _items(e)
             if any(_is_bound(s) for s in extract_numerics(it.text))]
    if not cands:
        return None
    it = rng.choice(cands)
    slot = next(s for s in extract_numerics(it.text) if _is_bound(s))
    # 해당 수치 바로 앞의 한정어만 지운다
    head = it.text[:slot.span[0]] if slot.span[1] else it.text
    m = list(_BOUND_RE.finditer(head))
    if not m:
        return None
    m = m[-1]
    it.text = it.text[:m.start()] + it.text[m.end():]
    return f"{it.item_id}: '{m.group(1)}' 삭제"


def _product_swap(e, rng, products=None, known_products=None, matcher=None, **_):
    if not products or not known_products:
        return None
    for p in products:
        holders = [it for it in _items(e) if p in it.text]
        if not holders:
            continue
        others = [k for k in known_products
                  if _ns(k) != _ns(p) and not (matcher and matcher.same_product(k, p))
                  and (matcher is None or matcher.informative(k))]
        if not others:
            continue
        k = rng.choice(sorted(others))
        it = holders[0]
        it.text = it.text.replace(p, k, 1)
        return f"{it.item_id}: '{p}' → '{k}'"
    return None


def _institution(e, rng, **_):
    bank = rng.choice(_OTHER_BANKS)
    for it in _items(e):
        if "하나은행" in it.text:
            it.text = it.text.replace("하나은행", bank, 1)
            return f"{it.item_id}: 하나은행 → {bank}"
    it = e.core[0]
    if detect_institutions(it.text):
        return None
    it.text = f"{bank}에서는 " + it.text
    return f"{it.item_id}: '{bank}에서는' 삽입"


# 서술어 덩어리 앞 어절이 이 어미로 끝나면 함께 묶는다: '처리해야 합니다', '하게 됩니다', '하고 있습니다'
_BOUND_PREV = re.compile(r"(야|게|고|지|어|아|서|여)$")


def _src(it, claim_texts) -> str | None:
    if not claim_texts:
        return None
    return " ".join(claim_texts.get(c, "") for c in it.claim_ids)


def _intensifier(e, rng, claim_texts=None, **_):
    """서술어 덩어리 앞에 강조어를 넣는다.

    2026-10-15 수정 (64번 진단):
      - '처리해야 반드시 합니다' 처럼 서술어 덩어리를 쪼개던 위치 오류 → 앞 어절이 연결 어미면 그 앞에 넣는다
      - 출처 원문에 같은 강도의 표현(INTENSIFIER_EQUIV: '해야'≈'반드시')이 있으면 강조가 아니므로 만들지 않는다
    """
    words = list(_INTENS)
    rng.shuffle(words)
    for it in e.core:
        m = _ENDING.search(it.text)
        if not m:
            continue
        src = _src(it, claim_texts)
        w = next((x for x in words if x not in it.text
                  and not (src and any(q in src for q in INTENSIFIER_EQUIV.get(x, (x,))))), None)
        if not w:
            continue
        pos = m.start()
        head = it.text[:pos].rstrip()
        prev = head.rsplit(" ", 1)[-1] if head else ""
        if prev and _BOUND_PREV.search(prev):
            pos = len(head) - len(prev)
        it.text = it.text[:pos] + f"{w} " + it.text[pos:]
        return f"{it.item_id}: '{w}' 삽입"
    return None


def _hedge_drop(e, rng, claim_texts=None, **_):
    """'~될 수 있습니다' → '~됩니다'.

    2026-10-15 수정 (64번 진단): 놓친 3건은 모두 Simplifier 가 원문에 없던 헤지를 '덧붙인' 문장이었다.
    그 헤지를 지우면 오히려 원문 뜻으로 돌아가므로 오류 교란이 아니다.
    → 출처 claim 원문에 가능성 헤지가 있을 때만 만든다. (덧붙인 헤지 자체는 back_verify 의 HEDGE_ADDED 가 잡는다)
    """
    for it in _items(e):
        src = _src(it, claim_texts)
        if src is not None and not HEDGE_GROUPS["가능성"].search(src):
            continue
        m = _HEDGE_RE.search(it.text)
        if m and HEDGE_GROUPS["가능성"].search(it.text):
            it.text = it.text[:m.start()] + _HEDGE_VERB[m.group(1)] + it.text[m.end():]
            return f"{it.item_id}: '{m.group(0)}' → '{_HEDGE_VERB[m.group(1)]}'"
    for it in _items(e):
        src = _src(it, claim_texts)
        m = _HEDGE_ADVERB.search(it.text)
        if m and (src is None or _HEDGE_ADVERB.search(src)):
            it.text = it.text[:m.start()] + it.text[m.end():]
            return f"{it.item_id}: '{m.group(1)}' 삭제"
    return None


_JOSA = re.compile(r"(에서|으로|에게|이나|이|가|은|는|을|를|의|에|로|와|과|도|만)$")


def key_token(cond_text: str) -> str:
    """조건에서 가장 긴 내용어 (조사 제거). 교란 가능 여부와 테스트용 가짜 LLM 이 쓴다."""
    toks = [_JOSA.sub("", t) for t in re.split(r"\s+", cond_text) if t]
    toks = [t for t in toks if len(t) >= 2 and not re.search(r"\d", t)]
    return max(toks, key=len) if toks else ""


def still_present(cond_text: str, rest_text: str) -> bool:
    """조건이 나머지 문구에 남아 있을 가능성: 조건의 수치나 가장 긴 내용어가 그대로 있으면 True."""
    rest = _ns(rest_text)
    if _ns(cond_text) in rest:
        return True
    for s in extract_numerics(cond_text):
        m = re.search(r"\d[\d,]*(?:\.\d+)?", s.raw)
        if m and m.group(0) in rest:
            return True
    k = key_token(cond_text)
    return bool(k) and k in rest


def _condition_drop(e, rng, **_):
    conds = list(e.required_conditions)
    rng.shuffle(conds)
    for k in conds:
        holders = [it for it in e.conditions if k.cond_id in it.condition_ids]
        if not holders:
            continue
        rest = [it for it in _items(e) if it not in holders]
        if not rest or still_present(k.text, " ".join(it.text for it in rest)):
            continue   # 조건이 다른 문구에 남아 있을 가능성이 큼 → 교란이 아님
        e.conditions = [it for it in e.conditions if it not in holders]
        return f"{k.cond_id} '{k.text}' 를 담은 문구 {[h.item_id for h in holders]} 삭제"
    return None


def _check_leak(e, rng, **_):
    if not e.staff_check or not e.core:
        return None
    sc = rng.choice(e.staff_check)
    cite = e.core[-1].claim_ids[:1]      # 출처는 OK claim 인 것처럼 단다 (메타데이터로는 못 잡게)
    e.core.append(SimpleItem(f"s{len(e.core) + 1}", sc.text, list(cite)))
    return f"s{len(e.core)} 로 {sc.claim_id} 원문 추가 (출처 표시 {cite})"


_FN = {"NUM_VALUE": _num_value, "NUM_UNIT": _num_unit, "NUM_COMPARATOR": _num_comparator,
       "BOUND_DROP": _bound_drop, "PRODUCT_SWAP": _product_swap, "INSTITUTION": _institution,
       "INTENSIFIER": _intensifier, "HEDGE_DROP": _hedge_drop, "CONDITION_DROP": _condition_drop,
       "CHECK_LEAK": _check_leak}


def perturb(expl: SimplifiedExplanation, kind: str, seed: int, **kw) -> tuple[SimplifiedExplanation, str] | None:
    """교란본과 설명. 적용할 곳이 없으면 None.

    kw: products, known_products, matcher (PRODUCT_SWAP), claim_texts={claim_id: 원문} (INTENSIFIER·HEDGE_DROP 의 원문 확인)
    """
    e = copy.deepcopy(expl)
    detail = _FN[kind](e, random.Random(seed), **kw)
    if detail is None or e.customer_text() == expl.customer_text():
        return None
    return e, detail


__all__ = ["TYPES", "perturb", "key_token", "still_present"]
