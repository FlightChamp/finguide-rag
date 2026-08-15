"""질문 분석 LLM 라우터.

역할의 한계
---------
이 모듈은 질문을 구조화하기만 한다. 거절 여부를 결정하지 않는다.
LLM 은 "이 질문이 어떤 상품을 가리키는가"를 뽑는 데만 쓰이고, 최종
판정은 카탈로그 대조와 Python 규칙이 한다.

이 분리가 중요한 이유는 이미 겪었다. 2단계 LLM 근거 검증은 프롬프트에
"근거가 질문과 다른 상품을 다루면 REFUSE"라고 명시했음에도 주택청약예금
근거로 정기예금 질문을 통과시켰다. LLM 판정을 최종 결정으로 삼으면
같은 실패가 반복된다. 사실 추출은 LLM 이 잘하고, 규칙 적용은 코드가
잘한다.

왜 정규식이 아닌가
---------------
상품명 표현 변이를 정규식으로 잡으려면 "정기예금", "정기 예금",
"예금을 정기로", "1년짜리 예금" 을 전부 나열해야 한다. 문서가 추가될
때마다 규칙이 늘고, 어느 시점부터는 아무도 고칠 수 없게 된다.
LLM 은 이 표면 변이 흡수를 값싸게 해낸다.

캐싱
----
분석 결과는 질문에만 의존하고 임계값·규칙과 무관하다. 거절 임계값
탐색에서 LLM 검증을 질문당 1회로 줄였던 것과 같은 성질이다. 따라서
디스크에 캐시하면 규칙을 몇 번 바꿔 재측정하든 호출은 한 번뿐이다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

GRANULARITIES = {"specific_product", "product_family", "none"}
INTENTS = {
    "rate_current", "fee", "limit", "eligibility", "early_termination",
    "required_docs", "procedure", "general_policy", "definition",
}

ANALYZE_SYSTEM = """당신은 은행 문서 검색 시스템의 질문 분석기입니다.
질문을 구조화된 JSON으로 변환합니다. 답변을 생성하거나 거절 여부를 판단하지 마십시오.

extracted_product
- 질문이 가리키는 상품명 또는 상품군을 질문에 쓰인 표현 그대로 뽑으십시오.
- 상품 언급이 없으면 null 입니다.
- 은행 이름(하나은행)은 상품명이 아닙니다.

product_granularity
- specific_product: 고유 상품명을 특정함 (예: 청년 주택드림 청약통장)
- product_family: 상품 종류만 언급 (예: 정기예금, 전세자금대출)
- none: 상품 언급 없음

intent
rate_current(현재 금리), fee(수수료), limit(한도), eligibility(가입/대출 자격),
early_termination(중도해지), required_docs(필요 서류), procedure(절차),
general_policy(거래 전반의 규칙), definition(용어 설명) 중 하나.

requires_product_specific_doc
- 답하려면 그 상품의 상품설명서나 약관이 반드시 필요하면 true.
- 상품마다 값이 다른 것(금리, 한도, 수수료, 우대조건, 중도해지이율)은 true.

can_answer_with_general_terms
- 예금거래 기본약관, 은행여신거래 기본약관, 전자금융거래 기본약관 같은
  일반 약관만으로 답할 수 있으면 true.
- 이자 지급 시기, 계좌 해지 절차, 통지 의무처럼 거래 전반에 적용되는
  규칙이면 true.
- 특정 상품의 수치를 묻고 있으면 false.

confidence
- 위 분석에 대한 확신도 0.0~1.0.

JSON으로만 답하십시오:
{"extracted_product": null 또는 문자열, "product_granularity": "...", "intent": "...", "requires_product_specific_doc": true/false, "can_answer_with_general_terms": true/false, "confidence": 0.0}"""

ANALYZE_TEMPLATE = """[질문]
{question}

위 질문을 분석하십시오."""


@dataclass
class QueryAnalysis:
    extracted_product: str | None = None
    product_granularity: str = "none"
    intent: str = "definition"
    requires_product_specific_doc: bool = False
    can_answer_with_general_terms: bool = False
    confidence: float = 0.0
    error: str = ""
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> "QueryAnalysis":
        gran = str(data.get("product_granularity", "none"))
        if gran not in GRANULARITIES:
            gran = "none"
        intent = str(data.get("intent", "definition"))
        if intent not in INTENTS:
            intent = "definition"

        product = data.get("extracted_product")
        product = str(product).strip() if product else None
        if not product:
            product = None
            gran = "none"

        try:
            conf = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0

        return cls(
            extracted_product=product,
            product_granularity=gran,
            intent=intent,
            requires_product_specific_doc=bool(
                data.get("requires_product_specific_doc", False)),
            can_answer_with_general_terms=bool(
                data.get("can_answer_with_general_terms", False)),
            confidence=max(0.0, min(1.0, conf)),
            raw=data,
        )

    def to_dict(self) -> dict:
        return {
            "extracted_product": self.extracted_product,
            "product_granularity": self.product_granularity,
            "intent": self.intent,
            "requires_product_specific_doc": self.requires_product_specific_doc,
            "can_answer_with_general_terms": self.can_answer_with_general_terms,
            "confidence": round(self.confidence, 3),
            "error": self.error,
        }


def _key(question: str, model: str) -> str:
    h = hashlib.sha256(f"{model}||{question}".encode("utf-8")).hexdigest()
    return h[:24]


class QueryAnalyzer:
    """질문을 구조화한다. 판정은 하지 않는다."""

    def __init__(self, client=None, model: str = "gpt-4.1-mini",
                 cache_path: Path | None = None):
        self.client = client
        self.model = model
        self.cache_path = cache_path
        self._cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    # ------------------------------------------------------------------

    def analyze(self, question: str) -> QueryAnalysis:
        k = _key(question, self.model)
        if k in self._cache:
            return QueryAnalysis.from_json(self._cache[k])

        if self.client is None:
            # 분석기를 쓸 수 없으면 아무 판단도 하지 않는다.
            # 상품 언급 없음으로 두어 커버리지 게이트가 작동하지 않게 한다.
            # 조용히 거절하는 것보다 게이트를 끄는 쪽이 안전하다.
            return QueryAnalysis(error="no_client")

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ANALYZE_SYSTEM},
                    {"role": "user", "content": ANALYZE_TEMPLATE.format(
                        question=question)},
                ],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            data["_tokens"] = resp.usage.prompt_tokens + resp.usage.completion_tokens
        except Exception as exc:
            return QueryAnalysis(error=str(exc)[:80])

        self._cache[k] = data
        return QueryAnalysis.from_json(data)

    def flush(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8")

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    @property
    def total_tokens(self) -> int:
        return sum(int(v.get("_tokens", 0)) for v in self._cache.values())
