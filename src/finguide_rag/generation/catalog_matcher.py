"""추출된 상품명과 카탈로그를 대조한다.

이 모듈에는 LLM 이 없다. 전부 결정적 규칙이다.
LLM 이 뽑은 상품명을 받아서 "우리 코퍼스에 있는가"만 판단한다.

판정 상태
--------
    covered     정규화 후 정확히 일치하는 문서가 있다.
    ambiguous   비슷하지만 같은 상품이라고 확정할 수 없다.
    unmatched   상품명이 감지됐으나 카탈로그에 없다.
    none        상품 언급이 없다.

왜 ambiguous 를 따로 두는가
------------------------
"청약저축"과 "주택청약종합저축"은 이름이 겹치지만 다른 상품이다.
포함관계만 보고 같다고 처리하면 잘못된 문서를 근거로 답하게 된다.
금융 도메인에서는 "모르는 것"과 "헷갈리는 것"이 서로 다른 실패이고,
사용자에게 줄 안내도 달라야 한다.

    unmatched  -> 그 상품 문서를 코퍼스에 추가해야 함
    ambiguous  -> 상품명을 더 정확히 입력해야 함

유사도 보정을 하지 않는 이유
-------------------------
편집거리나 부분 점수로 임계값을 잡으면, 임계값 근처에서 다른 상품이
같은 상품으로 판정된다. 검색 단계라면 오답이 순위에 섞이는 정도지만
여기서는 잘못된 근거로 확신에 찬 답변이 나간다. 확신이 서지 않으면
ambiguous 로 남기고 사람에게 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .product_catalog import CatalogEntry, ProductCatalog, normalize

# 포함관계 판정 시 이보다 짧은 문자열은 무시한다.
# "예금"이 "주택청약예금"에 포함된다는 이유로 ambiguous 가 남발되는 것을 막는다.
_MIN_OVERLAP_LEN = 4


@dataclass
class CatalogMatch:
    status: str                       # covered | ambiguous | unmatched | none
    matched_product: str | None = None
    doc_id: str | None = None
    reason: str = ""
    candidates: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "matched_product": self.matched_product,
            "doc_id": self.doc_id,
            "reason": self.reason,
            "candidates": self.candidates or [],
        }


def _exact(norm: str, entries: list[CatalogEntry]) -> CatalogEntry | None:
    for e in entries:
        if e.normalized and e.normalized == norm:
            return e
    return None


def _overlapping(norm: str, entries: list[CatalogEntry]) -> list[CatalogEntry]:
    """한쪽이 다른 쪽을 포함하는 항목들."""
    out = []
    for e in entries:
        if not e.normalized or len(e.normalized) < _MIN_OVERLAP_LEN:
            continue
        if len(norm) < _MIN_OVERLAP_LEN:
            continue
        if norm in e.normalized or e.normalized in norm:
            out.append(e)
    return out


def match_product(product: str | None, catalog: ProductCatalog) -> CatalogMatch:
    """상품명을 카탈로그와 대조한다."""
    if not product:
        return CatalogMatch(status="none", reason="상품 언급 없음")

    norm = normalize(product)
    if not norm:
        return CatalogMatch(status="none", reason="정규화 후 빈 문자열")

    entries = catalog.all_named

    hit = _exact(norm, entries)
    if hit:
        return CatalogMatch(
            status="covered",
            matched_product=hit.canonical_name,
            doc_id=hit.doc_id,
            reason="정규화 후 정확히 일치",
        )

    near = _overlapping(norm, entries)
    if near:
        return CatalogMatch(
            status="ambiguous",
            matched_product=near[0].canonical_name,
            doc_id=near[0].doc_id,
            reason="이름이 부분적으로 겹치나 동일 상품으로 확정할 수 없음",
            candidates=[e.canonical_name for e in near[:5]],
        )

    return CatalogMatch(
        status="unmatched",
        reason="카탈로그에 해당 상품 문서가 없음",
    )


# ==================================================================
# 최종 판정 규칙
# ==================================================================

# LLM 분석 결과와 카탈로그 대조 결과를 함께 본다.
# LLM 이 "이 상품 없는 것 같다"고 말한다고 바로 거절하지 않는다.


def should_refuse_product_mismatch(analysis, match: CatalogMatch) -> tuple[bool, str | None]:
    """커버리지 게이트.

    반환: (거절 여부, 사유)

    통과 조건이 거절 조건보다 앞에 온다. 과잉 거절이 더 흔한 실패
    양상이기 때문이다. 특히 2번 조건(일반 약관으로 답변 가능)이
    없으면 "예금 이자는 언제 지급되나요" 같은 정상 질문이 전부
    막힌다.
    """
    # 1. 상품 언급이 없으면 이 게이트의 대상이 아니다.
    if analysis.product_granularity == "none":
        return False, None

    # 2. 일반 약관으로 답할 수 있으면 상품 문서가 없어도 무방하다.
    if analysis.can_answer_with_general_terms:
        return False, None

    # 3. 상품별 문서가 꼭 필요한 질문이 아니면 통과시킨다.
    if not analysis.requires_product_specific_doc:
        return False, None

    # 4. 카탈로그에 정확히 있으면 통과.
    if match.status == "covered":
        return False, None

    # 5. 이름이 겹치지만 확정할 수 없으면 별도 사유로 거절.
    if match.status == "ambiguous":
        return True, "ambiguous_product"

    # 6. 상품별 문서가 필요한데 카탈로그에 없으면 거절.
    if match.status == "unmatched":
        return True, "product_mismatch"

    return False, None


# 사유별 안내 문구.
# 세 유형은 직원이 취해야 할 후속 조치가 다르므로 문구도 달라야 한다.
MESSAGES = {
    "product_mismatch": (
        "질문에 언급된 상품의 상품설명서 또는 약관이 현재 문서 범위에 포함되어 "
        "있지 않아 답변을 생성하지 않았습니다. 해당 상품의 최신 상품설명서 또는 "
        "약관을 확인한 뒤 안내해 주시기 바랍니다."
    ),
    "ambiguous_product": (
        "질문의 상품명이 보유 문서의 상품명과 정확히 일치하지 않습니다. "
        "다른 상품 문서를 근거로 잘못 안내하는 것을 막기 위해 답변을 생성하지 "
        "않았습니다. 상품명을 확인한 뒤 다시 조회해 주시기 바랍니다."
    ),
}
