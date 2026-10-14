"""초안 수준 범위 검사 — 서로 다른 상품·서비스 규정의 합성, 질문과 다른 범위로의 축소.

왜 claim 단위로는 안 되는가
-----------------------
"A 상품은 영업점에서 가입합니다. B 서비스는 앱에서 신청합니다." 각 문장은 각자의 근거로
맞다. 그래서 claim 단위 검증은 둘 다 SUPPORTED 를 준다. 그런데 고객은 하나의 질문을 했고,
답변은 두 상품의 규정을 '하나의 안내'처럼 섞었다. 이 오류는 claim 들의 근거를 '모아서'
봐야만 보인다. 개발 셋에서 이 유형이 claim 검증의 가장 큰 사각지대였다
(자연 오류 B_01·B_03, 개발 셋 q043·q055 — 두 프롬프트 모두 놓침).

검사 두 가지 (LLM 없음, 결정적)
----------------------------
MIXED_SCOPE      claim 들을 뒷받침한 근거 문서의 '이름 있는 상품·서비스 범위'가 둘 이상이다.
                 · 범위 = 상품설명서·비일반 약관에서 뽑은 상품명 (같은 상품은 matcher 로 묶음)
                 · 일반 약관(예금거래 기본약관 등)은 모든 상품에 걸치므로 범위로 세지 않는다
                 · FAQ 는 그 본문이 다른 근거의 상품명을 언급하지 않으면 별도 범위로 센다
                   (q043: '국가간 ATM 서비스' FAQ + '해외 ATM 출금 서비스' 설명서)
SCOPE_MISMATCH   질문은 특정 상품을 말하지 않았는데, 답변이 특정 상품명을 들어 그 상품의
                 규정으로 답했다 (q055: 일반 '외화 송금' 질문 → 'easy-one 외화송금서비스는 …').
                 · 판정 조건: claim 문장에 상품명이 나오고, 그 claim 의 근거가 그 상품 문서이며,
                   질문에 그 상품의 변별 어절(3자 이상)이 없음
                 · 질문이 '어떤 대출/상품/서비스…'처럼 상품을 묻는 경우는 제외

알려진 한계 (개발 셋 기준)
-----------------------
- 근거가 '어느 상품 문서인가'는 LLM 정렬(evidence_ids)에 의존한다. 정렬이 틀리면 같이 틀린다.
- q068(저축예금 질문에 다른 통장 설명서 규정 혼입)처럼 claim 이 상품명을 말하지 않고 다른 상품
  문서 하나만 빌려 온 경우는 잡지 못한다 (상품 범위가 하나뿐이고 claim 에 이름이 없음).
- 같은 상품군의 두 문서(보금자리론 / 주택연금 사전예약 보금자리론)는 matcher 가 같은 상품으로
  묶는다. 이 경우의 혼합은 놓친다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..generation.product_catalog import extract_product_name, normalize
from ..generation.product_match import ProductMatcher, tokens
from .schema import Claim, ReasonCode, ReasonHit, Source, Verification

# 거래 계약서·기본약관류는 특정 상품이 아니라 여러 상품에 공통 적용된다 → 범위로 세지 않음.
# (개발 셋 오탐 d025·d026: '대출거래약정서'를 상품으로 보고 경고)
_GENERIC_CONTRACT = re.compile(r"(약정서|기본약관|거래약관|추가약정)$")
# '어떤 대출을 못 받나요'처럼 질문 자체가 상품 이름을 묻는 경우, 답변이 상품명을 대는 것은 축소가 아니다.
_ASKS_WHICH = re.compile(r"(어떤|어느|무슨)\s*(대출|상품|서비스|통장|예금|적금|카드|계좌|보험|펀드)")
MIN_Q_TOKEN = 3          # 질문 쪽 어절은 3자 이상만 상품명 언급으로 인정 ('외화' 로 '외화송금서비스' 매칭 방지)
_TAIL = re.compile(r"(상품|서비스|통장|설명서|약관)$")


@dataclass(frozen=True)
class DocInfo:
    doc_id: str
    doc_type: str          # 설명서 / 약관 / FAQ
    display_name: str
    product: str           # 범위 이름. 일반 문서면 ""
    is_general: bool


class DocIndex:
    """doc_id → 범위 정보. documents.jsonl 과 product_catalog.json 에서 만든다."""

    def __init__(self, docs: dict[str, DocInfo]):
        self.docs = docs

    @classmethod
    def build(cls, documents_jsonl: Path, catalog_json: Path, matcher: ProductMatcher) -> "DocIndex":
        general_ids: set[str] = set()
        if catalog_json.exists():
            cat = json.loads(catalog_json.read_text(encoding="utf-8"))
            general_ids = {d["doc_id"] for d in cat.get("general_policy_docs", [])}
        docs: dict[str, DocInfo] = {}
        with documents_jsonl.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                name = d.get("display_name") or d.get("title") or ""
                dtype = d.get("doc_type", "")
                general = d["doc_id"] in general_ids or dtype == "FAQ"
                product = ""
                if dtype in ("설명서", "약관") and not general:
                    p, _ = extract_product_name(name.split(" - ")[0] if dtype == "약관" else name)
                    generic = bool(_GENERIC_CONTRACT.search(re.sub(r"\s+", "", p)))
                    product = p if matcher.informative(p) and not generic else ""
                    general = general or generic
                docs[d["doc_id"]] = DocInfo(d["doc_id"], dtype, name, product,
                                            general and dtype != "FAQ")
        return cls(docs)

    def get(self, doc_id: str) -> DocInfo | None:
        return self.docs.get(doc_id)


def _core(name: str) -> str:
    return _TAIL.sub("", re.sub(r"[\s『』「」\[\]()]", "", name))


def _named_in(text: str, product: str) -> bool:
    core = _core(product)
    return bool(core) and core in re.sub(r"[\s『』「」]", "", text)


def _cluster(products: Iterable[str], matcher: ProductMatcher) -> list[str]:
    """같은 상품을 하나로 묶은 대표 이름 목록."""
    reps: list[str] = []
    for p in products:
        if not any(matcher.same_product(p, r) for r in reps):
            reps.append(p)
    return reps


def _question_mentions(question: str, product: str, matcher: ProductMatcher) -> bool:
    q_tokens = [t for t in tokens(question) if len(t) >= MIN_Q_TOKEN]
    p_tokens = matcher.informative(product)
    if any(a.startswith(b) or b.startswith(a) for a in q_tokens for b in p_tokens):
        return True
    return _named_in(question, product)


def check_draft_scope(question: str, claims: list[Claim], verifications: list[Verification],
                      index: DocIndex, matcher: ProductMatcher) -> list[ReasonHit]:
    """초안 수준 범위 사유 코드. claims 와 verifications 는 같은 순서·같은 claim_id 여야 한다."""
    by_id = {v.claim_id: v for v in verifications}
    named: list[str] = []                       # 근거 문서의 상품 범위
    faq_docs: list[tuple[str, str]] = []        # (doc_id, faq 본문 인용)
    claim_products: list[tuple[Claim, list[str]]] = []

    for c in claims:
        v = by_id.get(c.claim_id)
        if v is None:
            continue
        prods = []
        for e in v.evidence:
            info = index.get(e.doc_id)
            if info is None:
                continue
            if info.product:
                prods.append(info.product)
            elif info.doc_type == "FAQ":
                faq_docs.append((e.doc_id, e.quote))
        named += prods
        claim_products.append((c, prods))

    hits: list[ReasonHit] = []
    reps = _cluster(named, matcher)
    if len(reps) >= 2:
        hits.append(ReasonHit(ReasonCode.MIXED_SCOPE, Source.RULE,
                              f"서로 다른 상품·서비스 근거가 한 답변에 섞임: {reps}"))
    elif len(reps) == 1:
        foreign = sorted({d for d, quote in faq_docs if not _named_in(quote, reps[0])})
        if foreign:
            hits.append(ReasonHit(ReasonCode.MIXED_SCOPE, Source.RULE,
                                  f"'{reps[0]}' 근거와 그 상품을 언급하지 않는 FAQ 근거가 섞임: {foreign}"))

    asks_which = bool(_ASKS_WHICH.search(question))
    for c, prods in ([] if asks_which else claim_products):
        for p in dict.fromkeys(prods):
            if _named_in(c.text, p) and not _question_mentions(question, p, matcher):
                hits.append(ReasonHit(ReasonCode.SCOPE_MISMATCH, Source.RULE,
                                      f"질문에 없는 '{p}' 기준으로 답함 ({c.claim_id})"))
                break
        if any(h.code == ReasonCode.SCOPE_MISMATCH for h in hits):
            break
    return hits
