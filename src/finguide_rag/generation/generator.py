"""근거 기반 답변 생성.

설계 원칙
--------
이 모듈은 "근거 원문을 그대로 보여주고, 답변은 그 근거에만 기반한다"를
지킨다. 영업점 직원이 사용자이기 때문이다. 직원은 결국 약관·설명서의
원문 조항을 근거로 고객에게 설명해야 하므로, 매끄럽게 요약된 문장보다
"원문 + 정리"가 실무에 맞는다.

환각을 막는 세 가지 장치
---------------------
1. 게이트      근거가 부족하면 애초에 생성을 호출하지 않는다.
               refusal.RefusalJudge 가 REFUSE 를 내면 이 모듈은 실행되지
               않는다. 환각의 최대 원인은 "근거가 없는데 억지로 답하는
               것"이므로 그 경로 자체를 차단한다.

2. 프롬프트 제약  제공된 근거 밖의 내용을 쓰지 못하게 하고, 질문이 물었으나
               근거에 없는 항목은 별도 필드(not_found)로 신고하게 한다.
               모델이 "모른다"를 표현할 자리를 만들어 주면 지어내는 압력이
               줄어든다.

3. 검증 가능성  Evidence.text 는 청크 원문 그대로이며 LLM 을 거치지 않는다.
               따라서 답변 문장을 발췌와 대조하면 환각이 즉시 드러난다.
               2번이 지켜졌는지를 3번으로 측정할 수 있다. 프롬프트 제약만
               있고 검증 수단이 없으면 그것은 선언에 불과하다.

발췌 원문을 가공하지 않는 이유
--------------------------
LLM 이 근거를 다듬으면 그 순간 "원문"이 아니게 되고, 3번 장치가 무너진다.
읽기 어려운 원문(PDF 추출 과정의 띄어쓰기 손실 등)이 있으나, 이는 표시
계층에서 다룰 문제이지 생성 단계에서 건드릴 문제가 아니다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .refusal import Decision, RefusalResult


# 제목 뒤에 " - " 로 붙은 꼬리표가 본문 조각인지 판단하는 기준.
# 정상 부제는 짧다: "(가계용)", "하나은행", "2020. 12. 10. 개정".
# 본문이 딸려 들어간 경우는 길고 콜론이나 문장부호를 포함한다.
_SUBTITLE_MAX = 24


def clean_doc_name(display_name: str) -> str:
    """문서 표시명에서 잘못 붙은 본문 조각을 떼어낸다.

    제목 추출 단계에서 일부 문서의 subtitle 에 본문 첫 줄이 들어갔다.
    데이터를 고치려면 재파싱이 필요하고 그러면 인덱스와 평가셋이 전부
    무효가 되므로, 표시 계층에서만 정리한다.
    """
    name = (display_name or "").strip()
    if " - " not in name:
        return name

    head, _, tail = name.partition(" - ")
    tail = tail.strip()
    # 짧으면 정상 부제로 보고 유지한다.
    if len(tail) <= _SUBTITLE_MAX and ":" not in tail:
        return name
    return head.strip()


@dataclass
class Evidence:
    """답변의 근거가 된 문서 조각.

    text 는 청크 원문 그대로다. 어떤 가공도 하지 않는다.
    """

    rank: int
    chunk_id: str
    text: str
    citation: str            # "『주택청약예금』 요약 상품설명서 (2026-03-10 시행)"
    doc_display_name: str
    doc_type: str            # 약관 / 설명서 / FAQ
    section: str             # "제7조(이자)" 등. 없을 수 있다.
    effective_date: str
    score: float
    is_latest: bool = True

    @classmethod
    def from_hit(cls, hit, rank: int) -> "Evidence":
        meta = getattr(hit, "meta", {}) or {}
        inner = meta.get("metadata", {}) or {}
        return cls(
            rank=rank,
            chunk_id=meta.get("chunk_id", getattr(hit, "chunk_id", "")),
            text=meta.get("text", ""),
            citation=meta.get("citation", meta.get("doc_display_name", "")),
            doc_display_name=meta.get("doc_display_name", ""),
            doc_type=meta.get("doc_type", ""),
            section=meta.get("section", "") or "",
            effective_date=meta.get("effective_date", ""),
            score=float(getattr(hit, "score", 0.0)),
            is_latest=bool(inner.get("is_latest", True)),
        )

    @property
    def header(self) -> str:
        """근거 블록의 제목 줄.

        citation 필드를 그대로 쓰지 않는다. citation 은 display_name 과
        section 을 이미 합쳐 만든 값이라, 여기서 section 을 또 붙이면
        구간명이 두 번 나온다. 또한 일부 display_name 에는 제목 추출
        과정에서 본문 일부가 subtitle 로 딸려 들어가 있다.

            입출금이 자유로운 예금약관 - 2. 저축예금 : 매년 3월, ...

        표시 계층에서 정리해 쓴다. 원문(text)은 건드리지 않는다.
        """
        name = clean_doc_name(self.doc_display_name)
        parts = [name]
        if self.section and self.section not in name:
            parts.append(self.section)
        head = " ".join(parts)
        if self.effective_date:
            head = f"{head} ({self.effective_date} 시행)"
        return head


@dataclass
class Answer:
    """파이프라인의 최종 산출물.

    거절이든 답변이든 같은 타입을 반환한다. 호출부(데모, 평가 스크립트)가
    분기를 신경 쓰지 않도록 하기 위해서다.
    """

    question: str
    decision: Decision
    answer: str                              # 거절이면 거절 안내 문구
    evidences: list[Evidence] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    refusal_reason: str | None = None
    stage: str = ""                          # pattern / retrieval / llm
    tokens: int = 0
    latency_ms: int = 0

    @property
    def refused(self) -> bool:
        return self.decision == Decision.REFUSE

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "decision": self.decision.value,
            "answer": self.answer,
            "not_found": self.not_found,
            "refusal_reason": self.refusal_reason,
            "stage": self.stage,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "evidences": [
                {
                    "rank": e.rank,
                    "chunk_id": e.chunk_id,
                    "citation": e.citation,
                    "section": e.section,
                    "doc_type": e.doc_type,
                    "score": round(e.score, 4),
                    "text": e.text,
                }
                for e in self.evidences
            ],
        }

    def format_display(self, max_evidence_chars: int = 600) -> str:
        """영업점 직원 화면용 텍스트.

        답변은 간결하게, 근거는 원문 그대로. 두 블록을 명확히 분리한다.
        직원이 근거 블록을 고객에게 그대로 읽어줄 수 있어야 한다.
        """
        lines: list[str] = []

        if self.refused:
            lines.append("[답변 불가]")
            lines.append(self.answer)
            if self.refusal_reason:
                lines.append(f"\n(사유: {self.refusal_reason} / 판정: {self.stage})")
        else:
            lines.append("[답변]")
            lines.append(self.answer)
            if self.not_found:
                lines.append("\n[문서에서 확인되지 않은 항목]")
                for item in self.not_found:
                    lines.append(f"  - {item}")

        if self.evidences:
            lines.append("\n[근거]")
            marks = "①②③④⑤⑥⑦⑧⑨⑩"
            for e in self.evidences:
                mark = marks[e.rank - 1] if e.rank <= len(marks) else f"[{e.rank}]"
                stale = "" if e.is_latest else "  ※ 최신본이 아닐 수 있음"
                lines.append(f"\n{mark} {e.header}{stale}")
                body = e.text[:max_evidence_chars]
                if len(e.text) > max_evidence_chars:
                    body += " …"
                lines.append(f"   {body}")

        return "\n".join(lines)


# ==================================================================
# 프롬프트
# ==================================================================

# 설계 의도
# --------
# - "근거에만 기반하라"는 지시는 흔하지만, 모델이 모를 때 빠져나갈 곳이
#   없으면 결국 지어낸다. not_found 필드가 그 출구다.
# - 수치는 금융 도메인에서 가장 위험한 환각 대상이다. 근거에 있는 값만
#   쓰고, 없으면 계산하거나 추정하지 말라고 명시한다.
# - 답변 대상은 고객이 아니라 직원이다. 존댓말 안내문이 아니라 업무용
#   요약이어야 한다.

GENERATE_SYSTEM = """당신은 은행 영업점 직원을 돕는 문서 기반 답변 생성기입니다.
제공된 근거만으로 답변하며, 직원이 고객 응대에 바로 활용할 수 있도록 정리합니다.

반드시 지킬 것:
- 제공된 근거에 있는 내용만 사용하십시오. 일반적인 금융 지식이나 추측을 더하지 마십시오.
- 금리, 기간, 금액, 비율은 근거에 적힌 값을 그대로 쓰십시오. 계산하거나 추정하지 마십시오.
- 질문이 물었으나 근거에서 확인되지 않는 항목은 답변에 쓰지 말고 not_found 에 적으십시오.
- 근거가 특정 상품에 대한 것이면 그 상품명을 밝히십시오. 다른 상품에 일반화하지 마십시오.

답변 형식:
- 3~5문장, 핵심부터. 직원이 읽고 바로 이해할 수 있게.
- 인사말, 사과, "문서에 따르면" 같은 군더더기를 넣지 마십시오.
- 조건이나 예외가 있으면 함께 적으십시오.

JSON으로만 답하십시오:
{"answer": "답변 본문", "not_found": ["근거에서 확인되지 않은 항목", ...]}
확인되지 않은 항목이 없으면 not_found 는 빈 배열입니다."""

GENERATE_TEMPLATE = """[질문]
{question}

[근거]
{evidence}

위 근거만으로 답변하십시오."""


def build_evidence_block(evidences: list[Evidence], max_chars: int = 700) -> str:
    """LLM 에게 넘길 근거 텍스트.

    refusal.build_evidence 와 같은 형식을 유지한다. 검증 단계와 생성
    단계가 같은 근거를 보아야 판정과 답변이 어긋나지 않는다.
    """
    parts = []
    for e in evidences:
        parts.append(f"[{e.rank}] {e.header}\n{e.text[:max_chars]}")
    return "\n\n".join(parts)


class AnswerGenerator:
    """근거 기반 답변 생성기.

    거절 판정을 통과한 질문만 여기에 도달한다. 이 클래스는 거절 여부를
    다시 판단하지 않는다. 책임을 분리해 두어야 각 단계를 따로 평가할 수
    있다.
    """

    def __init__(self, client=None, model: str = "gpt-4.1-mini",
                 top_n: int = 3, max_chars: int = 700,
                 max_tokens: int = 500):
        self.client = client
        self.model = model
        self.top_n = top_n
        self.max_chars = max_chars
        self.max_tokens = max_tokens

    def generate(self, question: str, hits) -> tuple[str, list[str], int]:
        """답변 본문, 미확인 항목, 토큰 수를 반환한다."""
        evidences = [Evidence.from_hit(h, i)
                     for i, h in enumerate(hits[:self.top_n], 1)]
        block = build_evidence_block(evidences, self.max_chars)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GENERATE_SYSTEM},
                    {"role": "user", "content": GENERATE_TEMPLATE.format(
                        question=question, evidence=block
                    )},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            answer = str(data.get("answer", "")).strip()
            not_found = [str(x) for x in (data.get("not_found") or [])]
            tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens
        except Exception as exc:
            # 생성 실패 시 답변을 지어내지 않는다. 근거는 이미 확보돼
            # 있으므로 직원이 원문을 직접 읽을 수 있다.
            return (f"답변 생성 중 오류가 발생했습니다. 아래 근거를 직접 "
                    f"확인해 주시기 바랍니다. ({str(exc)[:60]})"), [], 0

        return answer, not_found, tokens


def build_answer(question: str, hits, refusal: RefusalResult,
                 generator: AnswerGenerator | None,
                 top_n: int = 3, started: float | None = None) -> Answer:
    """거절 결과와 검색 결과를 받아 최종 Answer 를 만든다.

    거절이어도 근거를 함께 담는다. 시스템이 무엇을 찾았는지 직원이 볼 수
    있어야 판단을 이어갈 수 있기 때문이다. 다만 질문 패턴 단계에서 거절된
    경우(개인정보 필요 등)는 검색 결과가 질문과 무관할 수 있으므로 근거를
    싣지 않는다.
    """
    started = started if started is not None else time.perf_counter()

    if refusal.should_refuse:
        evidences = ([] if refusal.stage == "pattern"
                     else [Evidence.from_hit(h, i)
                           for i, h in enumerate(hits[:top_n], 1)])
        return Answer(
            question=question,
            decision=Decision.REFUSE,
            answer=refusal.message,
            evidences=evidences,
            refusal_reason=refusal.reason.value if refusal.reason else None,
            stage=refusal.stage,
            tokens=int((refusal.signals or {}).get("llm_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    evidences = [Evidence.from_hit(h, i)
                 for i, h in enumerate(hits[:top_n], 1)]

    if generator is None or generator.client is None:
        return Answer(
            question=question,
            decision=Decision.ANSWER,
            answer="(생성기가 없어 근거만 표시합니다.)",
            evidences=evidences,
            stage=refusal.stage or "no_generator",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    text, not_found, gen_tokens = generator.generate(question, hits)
    verify_tokens = int((refusal.signals or {}).get("llm_tokens", 0))

    return Answer(
        question=question,
        decision=Decision.ANSWER,
        answer=text,
        evidences=evidences,
        not_found=not_found,
        stage=refusal.stage,
        tokens=verify_tokens + gen_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
