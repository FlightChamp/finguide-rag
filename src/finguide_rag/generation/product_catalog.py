"""문서 대장 기반 상품 카탈로그.

왜 필요한가
---------
데모 첫 실행에서 "정기예금 중도해지하면 이자는?" 질문에 주택청약예금
설명서를 근거로 답변이 생성됐다. 코퍼스에 일반 정기예금 상품설명서가
없는데, 중도해지금리라는 주제가 겹쳐 검색 점수가 높게 나온 탓이다.

이 실패는 검색 점수로 구별할 수 없다. personalized 가 검색 신호로
구별되지 않았던 것(AUC 0.586)과 같은 구조다. 다만 personalized 는
1인칭이라는 문면 단서가 있었지만, 이 유형은 "우리 코퍼스에 그 상품
문서가 있는가"를 알아야만 판정할 수 있다.

따라서 코퍼스가 무엇을 담고 있는지에 대한 목록이 필요하다.

설계 원칙
--------
1. 상품명을 사람이 손으로 나열하지 않는다. document_registry.csv 에서
   기계적으로 추출한다. 문서가 추가되면 카탈로그가 자동으로 따라간다.
2. 특정 상품과 상품군을 구분한다.
3. 일반 약관(예금거래 기본약관 등)은 별도로 둔다. 상품설명서가 없어도
   이 문서들로 답할 수 있는 질문이 있기 때문이다. 데모 5번
   "예금거래에서 이자는 언제 지급되나요"가 그 경우였고 정상 답변됐다.
   이런 질문까지 막으면 과잉 거절이 폭증한다.

특정 상품과 상품군의 구분
----------------------
이 코퍼스에서는 제목의 겹낫표가 신뢰할 만한 신호다.

    『청년 주택드림 청약통장』 요약 상품설명서   -> 특정 상품
    가계대출 상품설명서                        -> 상품군

겹낫표는 은행이 상품 고유명사를 표기할 때 쓰는 관례다. 다만 이는
관찰에 기반한 휴리스틱이므로, 빗나가는 문서가 있는지 카탈로그 출력을
눈으로 확인해야 한다.

포함 기준
--------
is_parsable=True 이고 is_latest=True 인 문서만 넣는다. 스캔본이나
구버전은 실제로 검색되지 않으므로 카탈로그에 있으면 안 된다. 카탈로그가
인덱스보다 넓으면 "있다고 판단했는데 근거를 못 찾는" 모순이 생긴다.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# 제목 끝에 붙는 문서 종류 표기. 상품명 추출 시 떼어낸다.
_SUFFIXES = [
    "요약 상품설명서", "상품설명서(핵심/요약)", "핵심설명서", "상품설명서",
    "계약권유문서", "서비스 설명서", "서비스설명서", "설명서",
    "기본약관", "이용약관", "약관",
]

# 일반 약관으로 취급할 문서. 특정 상품이 아니라 거래 전반의 규칙을 담는다.
_GENERAL_TERMS_MARKERS = ["기본약관", "이용약관"]

# 카테고리를 업무 계열 이름으로 옮긴다.
# 상품명 alias 가 아니라 분류 체계이므로 설정으로 두어도 무방하다.
_CATEGORY_FAMILY = {
    "deposit": "예금",
    "loan": "대출",
    "digital_banking": "전자금융",
    "fx": "외환",
    "trust": "신탁·투자",
    "retirement_pension": "퇴직연금",
    "derivatives": "장외파생",
    "others": "기타",
}

_RE_BRACKET = re.compile(r"[『「](.+?)[』」]")
_RE_PAREN_TAIL = re.compile(r"\s*[\(（][^)）]*[\)）]\s*$")
_RE_NORM = re.compile(r"[^0-9a-z가-힣]")


def normalize(name: str) -> str:
    """비교용 정규화. 최소한만 한다.

    공백·특수문자 제거, 영문 소문자화. 금융 상품명은 한 글자 차이가
    다른 상품이므로 어간 추출이나 유사도 보정은 하지 않는다.
    """
    return _RE_NORM.sub("", (name or "").lower())


# 제목 뒤에 " - " 로 붙은 꼬리가 본문 조각인지 판단하는 기준.
# 정상 부제는 짧다: "(가계용)", "하나은행". 본문이 딸려 들어간 경우는
# 길고 콜론을 포함한다. 실제 사례:
#   입출금이 자유로운 예금약관 - 2. 저축예금 : 매년 3월, 6월, ...
_SUBTITLE_MAX = 24


def _strip_tails(t: str) -> str:
    """괄호 꼬리와 문서 종류 표기를 반복해서 떼어낸다.

    한 번만 떼면 "ISA 모델포트폴리오 설명서 (AI 최고위험)" 처럼 괄호
    뒤에 종류 표기가 남는 경우를 놓친다.
    """
    prev = None
    while prev != t:
        prev = t
        t = _RE_PAREN_TAIL.sub("", t).strip()
        for suf in _SUFFIXES:
            if t.endswith(suf):
                t = t[: -len(suf)].strip()
                break
    return t.strip(" -–—·")


def extract_product_name(title: str) -> tuple[str, str]:
    """제목에서 상품명과 세분도를 뽑는다.

    반환: (상품명, "specific_product" | "product_family")
    """
    t = (title or "").strip()

    m = _RE_BRACKET.search(t)
    if m:
        return m.group(1).strip(), "specific_product"

    if " - " in t:
        head, _, tail = t.partition(" - ")
        head_name = _strip_tails(head.strip())
        tail_name = _strip_tails(tail.strip()) if _usable_tail(tail.strip()) else ""

        # 어느 쪽이 실체인지는 문서마다 다르다.
        #   장외파생상품설명서 - 이자율 스왑(...)   -> 뒤쪽이 상품
        #   대출거래약정서(가계용) - 하나은행        -> 앞쪽이 상품
        # 정보량이 많은 쪽을 택한다.
        name = tail_name if len(tail_name) > len(head_name) else head_name
        return name, "product_family"

    return _strip_tails(t), "product_family"


def _usable_tail(tail: str) -> bool:
    """꼬리표를 상품명 후보로 볼 수 있는지."""
    if not tail or tail.startswith("("):
        return False
    if len(tail) > _SUBTITLE_MAX or ":" in tail:
        return False          # 본문이 딸려 들어간 경우
    digits = sum(c.isdigit() for c in tail)
    if digits / len(tail) > 0.3:
        return False          # "2020. 12. 10. 개정" 같은 개정 이력
    return True


@dataclass
class CatalogEntry:
    canonical_name: str
    normalized: str
    doc_id: str
    doc_title: str
    doc_type: str
    category: str
    granularity: str        # specific_product | product_family
    effective_date: str = ""


@dataclass
class ProductCatalog:
    covered_products: list[CatalogEntry] = field(default_factory=list)
    covered_families: list[CatalogEntry] = field(default_factory=list)
    general_policy_docs: list[CatalogEntry] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------

    @property
    def all_named(self) -> list[CatalogEntry]:
        """상품명 매칭 대상 전체."""
        return self.covered_products + self.covered_families

    def general_families(self) -> set[str]:
        return {e.canonical_name for e in self.general_policy_docs}

    # ------------------------------------------------------------------

    @classmethod
    def from_registry(cls, registry_csv: Path) -> "ProductCatalog":
        if not registry_csv.exists():
            raise FileNotFoundError(f"{registry_csv} 없음.")

        cat = cls()
        with registry_csv.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                cat._add_row(row)
        return cat

    def _add_row(self, row: dict) -> None:
        doc_id = (row.get("doc_id") or "").strip()
        title = (row.get("title") or "").strip()
        doc_type = (row.get("doc_type") or "").strip()
        category = (row.get("category") or "").strip()

        # 인덱스에 없는 문서는 카탈로그에도 없어야 한다.
        if str(row.get("is_parsable", "")).strip().lower() != "true":
            self.skipped.append({"doc_id": doc_id, "why": "is_parsable=False",
                                 "detail": row.get("exclusion_reason", "")})
            return
        if str(row.get("is_latest", "")).strip().lower() != "true":
            self.skipped.append({"doc_id": doc_id, "why": "is_latest=False",
                                 "detail": row.get("superseded_by", "")})
            return
        if not title:
            self.skipped.append({"doc_id": doc_id, "why": "제목 없음",
                                 "detail": row.get("display_name", "")})
            return

        # FAQ 는 상품 문서가 아니다. 질문-답변 쌍이므로 카탈로그에서 제외한다.
        if doc_type == "FAQ":
            return

        name, granularity = extract_product_name(title)
        if not name:
            self.skipped.append({"doc_id": doc_id, "why": "상품명 추출 실패",
                                 "detail": title})
            return

        entry = CatalogEntry(
            canonical_name=name,
            normalized=normalize(name),
            doc_id=doc_id,
            doc_title=title,
            doc_type=doc_type,
            category=category,
            granularity=granularity,
            effective_date=(row.get("effective_date") or "").strip(),
        )

        # 약관은 특정 상품이 아니라 거래 전반의 규칙을 담는다.
        # 약정서(대출거래약정서, 추가약정서)는 상품에 결합된 계약서이므로
        # 일반 약관으로 보지 않는다.
        is_general = (doc_type == "약관" and "약관" in title) or any(
            mk in title for mk in _GENERAL_TERMS_MARKERS)

        if is_general:
            entry.canonical_name = _CATEGORY_FAMILY.get(category, category)
            entry.normalized = normalize(entry.canonical_name)
            self.general_policy_docs.append(entry)
        elif granularity == "specific_product":
            self.covered_products.append(entry)
        else:
            self.covered_families.append(entry)

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        def dump(items):
            return [{
                "canonical_name": e.canonical_name,
                "doc_id": e.doc_id,
                "doc_title": e.doc_title,
                "doc_type": e.doc_type,
                "category": e.category,
                "granularity": e.granularity,
                "effective_date": e.effective_date,
            } for e in items]

        return {
            "covered_products": dump(self.covered_products),
            "covered_families": dump(self.covered_families),
            "general_policy_docs": dump(self.general_policy_docs),
            "skipped_count": len(self.skipped),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ProductCatalog":
        data = json.loads(path.read_text(encoding="utf-8"))
        cat = cls()
        for key, bucket in [("covered_products", cat.covered_products),
                            ("covered_families", cat.covered_families),
                            ("general_policy_docs", cat.general_policy_docs)]:
            for d in data.get(key, []):
                bucket.append(CatalogEntry(
                    canonical_name=d["canonical_name"],
                    normalized=normalize(d["canonical_name"]),
                    doc_id=d["doc_id"],
                    doc_title=d["doc_title"],
                    doc_type=d["doc_type"],
                    category=d["category"],
                    granularity=d["granularity"],
                    effective_date=d.get("effective_date", ""),
                ))
        return cat
