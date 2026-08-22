"""상품 일치 판정.

LLM 을 쓰지 않는다. 질문에서 뽑은 상품명과 근거 문서의 상품명을
문자열로 대조한다.

왜 별도 모듈인가
-------------
평가기(34, 36번)가 각자 이 로직을 갖고 있으면, 한쪽만 고쳤을 때
25건에서 검증한 결과가 96건에 적용된다는 보장이 깨진다. 판정 로직은
한 곳에만 둔다.

v1 의 실패
---------
첫 구현은 96건 중 7건을 판정해 5건을 fail 로 냈다(0.714). 사람 검수
기준 상품 불일치는 22건 중 3건(약 0.14)이었으므로 명백한 오작동이다.
원인이 둘이었다.

  1. 문서 혼합 조건이 "인용된 문서 2건 이상" 이었다.
     같은 상품의 설명서와 약관을 함께 인용하는 것은 정상인데 fail 이
     됐다. 문제는 문서가 여럿인 것이 아니라 서로 다른 상품이 섞이는
     것이므로, 문서가 아니라 상품 수를 센다.

  2. 문서명 대조가 완전 포함 관계만 봤다.
     "하나은행 환전지갑" 이 "개인뱅킹 환전지갑 서비스 설명서" 에
     통째로 들어가지 않아 unmatched 가 됐다. 실제로는 코퍼스에 있는
     상품이다. 어절 단위로 겹침을 본다.

일반 어절 처리
------------
어절 겹침만 보면 "하나은행" 하나로 모든 문서가 매칭된다. 그래서
카탈로그 전체에서 여러 상품에 걸쳐 나타나는 어절은 변별력이 없다고
보고 제외한다. 기준은 데이터에서 나온다. 불용어를 손으로 나열하면
문서가 늘 때마다 목록을 고쳐야 한다.

    하나은행   20개 상품에 등장 -> 제외
    환전지갑    6개 상품에 등장 -> 유지
    설명서/약관 대부분에 등장   -> 자동 제외

접두 일치만 허용한다
-----------------
"입출금" 과 "입출금이" 는 같은 말이지만 "적금" 과 "자유적금" 은 다른
상품이다. 앞에서부터 겹치는 것만 같은 말로 본다. 뒤에서 겹치는 것은
상위 범주가 같을 뿐이므로 매칭하지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .product_catalog import ProductCatalog, extract_product_name, normalize

# 이 수 이상의 상품에 등장하는 어절은 변별력이 없다고 본다.
GENERIC_DF = 10

# 어절이 이보다 짧으면 비교하지 않는다.
MIN_TOKEN = 2

_RE_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")
_RE_SUBTITLE = re.compile(r"\s+-\s+")


def _clean(name: str) -> str:
    """표시명에서 잘못 붙은 본문 조각을 떼어낸다."""
    n = (name or "").strip()
    if " - " not in n:
        return n
    head, _, tail = n.partition(" - ")
    tail = tail.strip()
    if len(tail) <= 24 and ":" not in tail:
        return n
    return head.strip()


def tokens(name: str) -> list[str]:
    """비교용 어절. 소문자화하고 기호를 제거한다."""
    return [t.lower() for t in _RE_TOKEN.findall(_clean(name))
            if len(t) >= MIN_TOKEN]


def _prefix_match(a: str, b: str) -> bool:
    """한쪽이 다른 쪽의 접두인가."""
    return a.startswith(b) or b.startswith(a)


@dataclass
class ProductMatcher:
    """카탈로그에서 일반 어절을 학습해 두고 대조에 쓴다."""

    generic: frozenset[str]

    @classmethod
    def from_catalog(cls, path: Path) -> "ProductMatcher":
        if not path.exists():
            return cls(generic=frozenset())
        cat = ProductCatalog.load(path)
        df: dict[str, int] = {}
        for e in cat.all_named:
            for t in set(tokens(e.canonical_name)):
                df[t] = df.get(t, 0) + 1
        return cls(generic=frozenset(t for t, c in df.items() if c >= GENERIC_DF))

    # ------------------------------------------------------------------

    def informative(self, name: str) -> list[str]:
        """변별력 있는 어절만 남긴다."""
        return [t for t in tokens(name) if t not in self.generic]

    def same_product(self, question_product: str, doc_name: str) -> bool:
        """질문의 상품과 문서가 같은 상품을 가리키는가."""
        qt = self.informative(question_product)
        dt = self.informative(doc_name)
        if not qt or not dt:
            return False

        # 정규화 후 완전 포함이면 확실히 같다.
        qn, dn = normalize(question_product), normalize(doc_name)
        if qn and dn and (qn in dn or dn in qn):
            return True

        return any(_prefix_match(a, b) for a in qt for b in dt)

    def product_of(self, doc_name: str) -> str:
        """문서명에서 상품명을 뽑는다."""
        return extract_product_name(_clean(doc_name))[0]

    def distinct_products(self, doc_names: list[str]) -> set[str]:
        """서로 다른 상품이 몇 개인가.

        같은 상품의 설명서와 약관은 하나로 센다.
        """
        out: set[str] = set()
        for name in doc_names:
            p = self.product_of(name)
            key = normalize(p)
            if not key:
                continue
            # 이미 담긴 상품과 접두가 겹치면 같은 상품으로 본다.
            if any(_prefix_match(key, k) for k in out):
                continue
            out.add(key)
        return out


def judge_product_match(matcher: ProductMatcher, evidences: list[dict],
                        question_product: str,
                        cited_ranks: set[int]) -> tuple[str, str]:
    """상품 일치 판정.

    혼합이 문제가 되는 기준
    --------------------
    근거가 여러 문서인 것 자체는 문제가 아니다. 질문이 상품군 수준이면
    그 안의 여러 문서를 함께 인용하는 것이 오히려 자연스럽다.

        "환전지갑" 질문에 선물하기 서비스 + 제휴 서비스 인용  -> 정상
        청약통장 질문에 청약통장 + 보금자리론 인용            -> 위험

    차이는 인용된 문서가 질문의 상품에 속하는가다. 따라서 문서 수를
    세지 않고, 질문 상품에 속하지 않는 문서가 섞였는지를 본다.

    질문에 상품 언급이 없으면 기준으로 삼을 상품이 없으므로, 그때만
    서로 다른 상품이 몇 개인지로 판단한다.

    반환: (pass | fail | na, 사유)
    """
    if not evidences:
        return "na", "근거 없음"

    by_rank = {e["rank"]: e for e in evidences}
    cited = [by_rank[i] for i in cited_ranks if i in by_rank] or evidences

    def name_of(e: dict) -> str:
        return e.get("doc_display_name") or e.get("citation", "")

    doc_types = {e.get("doc_type", "") for e in evidences}
    all_terms = bool(doc_types) and doc_types <= {"약관", "FAQ"}
    product = (question_product or "").strip()

    # 질문에 상품이 없으면 상품 수로만 판단한다.
    if not product:
        products = matcher.distinct_products([name_of(e) for e in cited])
        if len(products) >= 2:
            return "fail", f"상품 언급 없음이나 서로 다른 상품 {len(products)}건 혼합"
        return "na", "질문에 상품 언급 없음"

    hits = [matcher.same_product(product, name_of(e)) for e in cited]

    if hits and all(hits):
        return "pass", "인용된 근거가 모두 질문 상품에 속함"
    if any(hits):
        off = [name_of(e) for e, h in zip(cited, hits) if not h]
        return "fail", f"질문 상품 외 문서 혼합: {matcher.product_of(off[0])}"

    # 하나도 맞지 않는 경우
    if any(matcher.same_product(product, name_of(e)) for e in evidences):
        return "fail", "질문 상품 문서가 검색됐으나 답변이 인용하지 않음"
    if all_terms:
        return "na", "근거가 일반 약관·FAQ 이므로 상품 무관"
    return "fail", f"질문 상품 '{product}' 과 일치하는 근거 문서 없음"
