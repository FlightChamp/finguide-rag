"""ClaimVerifier — 직원 초안을 claim 으로 나누고 근거와 대조해 판정한다.

흐름 (LLM 호출은 초안당 정확히 2회, 둘 다 캐시)
--------------------------------------------
    1. extract   [LLM]  초안 → claim 목록 (원문 부분 문자열 + 상품·사건·조건 슬롯)
    2. locate    [규칙] claim 텍스트를 초안에서 찾아 span 확정, 수치·단정도는 규칙으로 채움
    3. align     [LLM]  claim 전부 × 근거 문장 목록 → 근거 문장 id + LLM 코드 4종 중 하나
    4. rules     [규칙] rules.run_claim_rules (정렬된 근거 문장만 사용)
    5. policy    [규칙] policy.build_verification → claim 판정, aggregate_final → 초안 판정

LLM 이 하지 않는 것
-----------------
    - 판정 라벨(OK/CHECK/PROHIBITED): policy 가 사유 코드에서 도출
    - 근거 인용문 작성: LLM 은 근거 문장 'id' 만 고른다. 인용문(quote)은 우리가 가진
      원문 문장을 그대로 붙인다. LLM 이 근거를 '요약'하다 바꿔 쓰는 위험을 없앤다.
    - 수치·상품·기관·단정 대조: 규칙

fail-closed
-----------
    추출 실패           → 초안 전체를 claim 1개로 두고 VERIFIER_ERROR (CHECK)
    정렬 결과 누락       → 해당 claim VERIFIER_ERROR
    SUPPORTED 인데 근거 id 없음 → 자기모순. SUPPORTED 를 버리고 VERIFIER_ERROR
    허용 밖 코드         → policy.sanitize_llm_codes 가 VERIFIER_ERROR 로 변환

근거 범위
--------
MVP 는 호출부가 근거 청크를 넘겨준다(평가에서는 v1 답변이 인용한 청크).
claim 별 보조 검색(v1 하이브리드 검색기 재사용)은 다음 단계에서 Retriever 노드로 붙인다.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..generation.product_match import ProductMatcher
from .numeric import extract_numerics
from .policy import aggregate_final, build_verification
from .rules import assertiveness_of, run_claim_rules
from .schema import (
    LLM_CLAIM_EMITTABLE,
    CatalogStatus,
    Claim,
    EvidenceRef,
    FinalDecision,
    ReasonCode,
    ReasonHit,
    Source,
    Verification,
)

PROMPT_VERSION = "cv-v2"   # 기본값. 이전 버전은 ClaimVerifier(prompt_version="cv-v1") 로 재현
MAX_EVIDENCE_SENTENCES = 60

# ==================================================================
# 프롬프트
# ==================================================================

EXTRACT_SYSTEM = """너는 금융 상담 답변 초안을 '검증 가능한 주장(claim)' 단위로 나누는 분석기다.
참·거짓 판단은 하지 않는다.

규칙
1. 각 claim 의 text 는 초안의 연속된 부분 문자열이어야 한다. 한 글자도 바꾸거나 요약하지 마라.
2. 조건과 결과는 한 claim 으로 둔다. 예: "급여이체 실적이 있으면 연 0.2%p 우대됩니다"는 하나다.
3. 한 문장에 서로 독립된 사실이 둘 이상이면 나눈다. 예: "A는 면제되고, B는 1.5%가 부과됩니다"는 둘이다.
4. 인사, 연결어, 안내 문구("참고해 주세요" 등)만 있는 부분은 claim 으로 만들지 않는다.
5. product: claim 이 가리키는 상품·서비스명을 초안 표기 그대로. 없으면 null.
6. event: 중도해지·만기·가입·연체·이체 같은 상황어를 초안 표기 그대로. 없으면 null.
7. conditions: claim 안의 조건·한정·예외 표현을 초안 표기 그대로 나열. 없으면 [].
8. needs_customer_data: 이 고객 개인의 정보(거래실적·신용·잔액·가입이력 등)를 조회해야만
   참·거짓을 알 수 있으면 true.

출력은 JSON 하나:
{"claims": [{"text": "...", "product": null, "event": null, "conditions": [], "needs_customer_data": false}]}"""

EXTRACT_USER = "고객 질문:\n{question}\n\n직원 답변 초안:\n{draft}"

ALIGN_SYSTEM = """너는 금융 상담 claim 을 근거 문장과 대조하는 검토자다.
각 claim 마다 아래를 수행한다.

1. evidence_ids: 근거 목록에서 claim 의 내용과 직접 관련된 문장 id 를 최대 3개 고른다. 없으면 [].
2. code: 다음 중 정확히 하나.
   SUPPORTED          근거가 claim 을 뒷받침하고, 근거의 조건·제한이 claim 에서 빠지지 않았다.
   MISSING_CONDITION  내용은 근거와 맞지만, 근거에 있는 조건·한정·예외가 claim 에서 빠져
                      claim 이 근거보다 넓게 읽힌다.
   CONTRADICTED       근거와 반대되거나 다르다 (수치·기간·대상·상품이 다른 경우 포함).
   NO_EVIDENCE        claim 을 판단할 근거 문장이 목록에 없다.
3. missing_conditions: MISSING_CONDITION 일 때 빠진 조건을 근거 표현 그대로. 아니면 [].
4. fix: SUPPORTED 가 아니면 근거 표현을 살려 claim 을 고친 한 문장. SUPPORTED 면 null.

주의
- 근거 문장이 claim 과 다른 상품·서비스·기관에 대한 것이면 SUPPORTED 가 아니다.
- 근거에 없는 내용을 상식으로 채우지 마라. 확신이 없으면 NO_EVIDENCE.
- 판정 라벨(안내 가능 등)이나 다른 코드는 쓰지 마라.

출력은 JSON 하나:
{"results": [{"claim_id": "c1", "evidence_ids": ["E3"], "code": "SUPPORTED", "missing_conditions": [], "fix": null}]}"""

ALIGN_USER = "고객 질문:\n{question}\n\n[claim]\n{claims}\n\n[근거 문장]\n{evidence}"

# ------------------------------------------------------------------
# cv-v2 정렬 프롬프트 — 2026-10-14 3차 평가 진단을 반영
#   정상 초안 claim 70개 중 9개가 MISSING_CONDITION 단독으로 CHECK 였고, LLM 이 적은
#   '빠진 조건'을 보니 세 부류였다.
#     (a) 앞 문장에 이미 있는 조건 — claim 분해가 공통 맥락을 잘라냄 ('외화현찰로 입금할 때')
#     (b) claim 이 다른 말로 이미 담은 조건 ('로그인 상태에서')
#     (c) 조건이 아닌 정의·산식·표 세부값 ('환율 민감도(Delta)는 …')
#   반대로 자연 오류 d029('은행 책임 없는 사유로' 생략)는 SUPPORTED 로 놓쳤다.
#   → 초안 전문을 함께 주고, '조건'의 정의와 제외 대상, 면책 사유 한정의 중요성을 명시한다.
#   주의: 이 수정은 평가셋(정상 25건·자연 8건)의 결과를 보고 한 것이므로, 같은 셋의
#   개선 폭은 낙관적이다. 새 자연 오류 셋으로 다시 확인해야 한다.
# ------------------------------------------------------------------
ALIGN_SYSTEM_V2 = """너는 금융 상담 claim 을 근거 문장과 대조하는 검토자다.
각 claim 마다 아래를 수행한다.

1. evidence_ids: 근거 목록에서 claim 의 내용과 직접 관련된 문장 id 를 최대 3개 고른다. 없으면 [].
2. code: 다음 중 정확히 하나.
   SUPPORTED          근거가 claim 을 뒷받침하고, 근거의 조건이 claim(또는 초안의 맥락)에 살아 있다.
   MISSING_CONDITION  내용은 근거와 맞지만, 근거의 '조건'이 빠져 claim 이 근거보다 넓게 적용되는 것처럼 읽힌다.
   CONTRADICTED       근거와 반대되거나 다르다 (수치·기간·대상·상품이 다른 경우 포함).
   NO_EVIDENCE        claim 을 판단할 근거 문장이 목록에 없거나, claim 표현이 모호·불완전해
                      근거와 같은 뜻인지 확인할 수 없다.
3. missing_conditions: MISSING_CONDITION 일 때 빠진 조건을 근거 표현 그대로. 아니면 [].
   MISSING_CONDITION 이면 반드시 1개 이상 적는다.
4. fix: SUPPORTED 가 아니면 근거 표현을 살려 claim 을 고친 한 문장. SUPPORTED 면 null.

'조건'의 정의
- 조건이란 claim 이 적용되는 대상·상황·시점·한도·예외를 좁히는 표현이다.
  예: '~인 경우에 한해', '~로 인한', '~를 제외하고', '단, ~', 'OO 고객에 한함'.
- 다음은 빠진 조건이 아니다.
  (a) [초안 전문]의 다른 문장에 이미 있고 이 claim 에도 이어서 적용되는 조건
  (b) claim 이 다른 표현으로 이미 담고 있는 조건
  (c) 용어 정의, 산식, 표의 세부 값, 배경 설명
- 반드시 확인할 것: 책임·면책·손해부담·제재 조항에서 '사유'를 한정하는 표현
  (예: '은행의 책임 없는 사유로', '고객의 과실로', '정당한 사유 없이')이 빠지면
  claim 이 근거보다 넓어진다 → MISSING_CONDITION.

주의
- 근거 문장이 claim 과 다른 상품·서비스·기관에 대한 것이면 SUPPORTED 가 아니다.
- 근거에 없는 내용을 상식으로 채우지 마라.
- 판정 라벨(안내 가능 등)이나 다른 코드는 쓰지 마라.

출력은 JSON 하나:
{"results": [{"claim_id": "c1", "evidence_ids": ["E3"], "code": "SUPPORTED", "missing_conditions": [], "fix": null}]}"""

ALIGN_USER_V2 = ("고객 질문:\n{question}\n\n[초안 전문]\n{draft}\n\n[claim]\n{claims}"
                 "\n\n[근거 문장]\n{evidence}")

PROMPTS = {
    "cv-v1": (ALIGN_SYSTEM, ALIGN_USER),
    "cv-v2": (ALIGN_SYSTEM_V2, ALIGN_USER_V2),
}


def prompt_hash(version: str = "cv-v1") -> str:
    """프롬프트 버전 지문. 감사 로그와 캐시 키에 쓴다.

    cv-v1 의 값(5ebe0746f3a6)은 기존 캐시 키에 들어 있으므로 계산식을 바꾸지 않는다.
    """
    align_system = PROMPTS[version][0]
    raw = "|".join([version, EXTRACT_SYSTEM, align_system])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


# 추출 프롬프트는 cv-v1 이후 바뀌지 않았다. 추출 호출의 캐시 키는 항상 cv-v1 지문을 써서
# 정렬 프롬프트만 바꿀 때 추출 결과(=claim 분해)를 재사용한다. 비교 실험에서 claim 이
# 같아야 정렬 프롬프트의 효과만 분리해서 볼 수 있다.
EXTRACT_TAG = "cv-v1"


# ==================================================================
# LLM 호출 + 캐시
# ==================================================================


class ChatClient(Protocol):
    """openai.OpenAI 와 같은 모양. 테스트에서는 가짜 클라이언트를 넣는다."""
    chat: Any


class CachedLLM:
    """JSON 응답 LLM 호출을 디스크 캐시와 함께 감싼다.

    캐시 키 = sha256(모델 | 프롬프트 해시 | system | user). 프롬프트를 고치면 키가 바뀐다.
    v1 에서 temperature=0 이어도 판정이 흔들려 과잉거절이 0.100 ↔ 0.200 으로 오갔다.
    같은 입력에 같은 판정을 보장하는 유일한 방법이 캐시다.
    """

    def __init__(self, client: ChatClient | None, model: str = "gpt-4.1-mini",
                 cache_path: Path | None = None):
        self.client, self.model, self.cache_path = client, model, cache_path
        self.cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}
        self.calls = self.hits = self.tokens = 0

    def _key(self, system: str, user: str, tag: str) -> str:
        raw = f"{self.model}|{tag}|{system}|{user}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def json(self, system: str, user: str, max_tokens: int = 1500, tag: str = "") -> dict | None:
        k = self._key(system, user, tag or prompt_hash("cv-v1"))
        if k in self.cache:
            self.hits += 1
            return self.cache[k]["data"]
        if self.client is None:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            data = json.loads(resp.choices[0].message.content)
            used = resp.usage.prompt_tokens + resp.usage.completion_tokens
        except Exception:   # 네트워크·파싱 실패 모두 fail-closed 경로로
            return None
        self.calls += 1
        self.tokens += used
        self.cache[k] = {"data": data, "tokens": used}
        return data

    def flush(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1),
                                       encoding="utf-8")


# ==================================================================
# 입력 · 출력 타입
# ==================================================================


@dataclass
class EvidenceChunk:
    """호출부가 넘기는 근거 청크 (v1 Evidence 와 호환되는 최소 필드)."""
    chunk_id: str
    text: str
    doc_name: str = ""
    doc_type: str = ""
    is_latest: bool = True
    institution: str = "하나은행"

    @property
    def doc_id(self) -> str:
        return self.chunk_id.rsplit("_c", 1)[0] if "_c" in self.chunk_id else self.chunk_id


@dataclass
class VerifyResult:
    claims: list[Claim]
    verifications: list[Verification]
    final: FinalDecision
    llm_codes: dict[str, list[str]] = field(default_factory=dict)    # ablation 용 원본 LLM 코드
    rule_hits: dict[str, list[ReasonHit]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    prompt_hash: str = ""

    def to_dict(self) -> dict:
        return {"claims": [c.to_dict() for c in self.claims],
                "verifications": [v.to_dict() for v in self.verifications],
                "final": self.final.to_dict(), "llm_codes": self.llm_codes,
                "errors": self.errors, "prompt_hash": self.prompt_hash}


# ==================================================================
# 본체
# ==================================================================

_EV_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|\s(?=\d+\.\s)|\s(?=[·•\-]\s)")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def locate(draft: str, text: str) -> tuple[int, int] | None:
    """claim 텍스트의 초안 내 위치. 공백 차이는 허용한다."""
    i = draft.find(text)
    if i >= 0:
        return i, i + len(text)
    pat = r"\s*".join(re.escape(ch) for ch in re.sub(r"\s+", "", text))
    m = re.search(pat, draft) if pat else None
    return (m.start(), m.end()) if m else None


class ClaimVerifier:
    def __init__(self, llm: CachedLLM, matcher: ProductMatcher,
                 promote_customer_slot: bool = False,
                 known_products: list[str] | None = None,
                 prompt_version: str = PROMPT_VERSION):
        """promote_customer_slot: LLM 추출 슬롯 needs_customer_data=true 를 사유 코드로 승격할지.

        54번 첫 실행(2026-10-14)에서 켜 두었더니 주입 샘플 대상 claim 75개 중 26개에
        NEEDS_CUSTOMER_DATA 가 붙었다. 대부분 고객 개인 정보와 무관한 일반 규정 문장이었고,
        자연 오류 셋의 '탐지' 6건 중 3건이 기대 코드가 아닌 이 코드로만 걸렸다 —
        우연히 맞힌 경보다. 그래서 기본값을 끄고, 고객정보 판정은 51번 dev 셋에서
        오탐을 확인한 규칙(rules.check_customer_data)만 쓴다. 슬롯 값은 Claim 에 그대로 남겨
        화면 참고 정보로만 쓴다. 비교 재현이 필요하면 True 로 켠다.
        """
        self.llm, self.matcher = llm, matcher
        self.promote_customer_slot = promote_customer_slot
        # 카탈로그 상품명 목록. 없으면(None) 상품명 존재 여부 확인을 건너뛴다 (rules.check_product)
        self.known_products = known_products
        if prompt_version not in PROMPTS:
            raise ValueError(f"알 수 없는 프롬프트 버전: {prompt_version}")
        self.prompt_version = prompt_version

    # ---------- 1~2. 추출 + 위치 ----------
    def extract(self, question: str, draft: str) -> tuple[list[Claim], list[str]]:
        errors: list[str] = []
        data = self.llm.json(EXTRACT_SYSTEM, EXTRACT_USER.format(question=question, draft=draft),
                             tag=prompt_hash(EXTRACT_TAG))
        raw = (data or {}).get("claims") if isinstance(data, dict) else None
        if not isinstance(raw, list) or not raw:
            errors.append("extract_failed")
            return [self._fallback_claim(draft)], errors

        claims: list[Claim] = []
        for i, c in enumerate(raw, 1):
            text = _norm_ws(str(c.get("text", "")))
            if not text:
                continue
            span = locate(draft, text)
            if span is None:
                errors.append(f"c{i}:not_substring")   # 요약·재작성 → 위치 불명. 판정은 하되 기록
                span = (-1, -1)
            claims.append(Claim(
                claim_id=f"c{i}", text=text, span=span,
                product=(c.get("product") or None),
                event=(c.get("event") or None),
                numerics=extract_numerics(text),                  # 규칙
                conditions=[str(x) for x in (c.get("conditions") or [])],
                assertiveness=assertiveness_of(text),             # 규칙
                needs_customer_data=bool(c.get("needs_customer_data", False)),
            ))
        if not claims:
            errors.append("extract_empty")
            return [self._fallback_claim(draft)], errors
        return claims, errors

    @staticmethod
    def _fallback_claim(draft: str) -> Claim:
        return Claim(claim_id="c1", text=draft, span=(0, len(draft)), checkable=False)

    # ---------- 3. 정렬·함의 ----------
    @staticmethod
    def evidence_sentences(chunks: list[EvidenceChunk]) -> list[tuple[str, EvidenceChunk, str]]:
        out = []
        for ch in chunks:
            for s in _EV_SPLIT.split(ch.text):
                s = _norm_ws(s)
                if len(s) > 4:
                    out.append((f"E{len(out) + 1}", ch, s))
                if len(out) >= MAX_EVIDENCE_SENTENCES:
                    return out
        return out

    def align(self, question: str, claims: list[Claim],
              ev: list[tuple[str, EvidenceChunk, str]], draft: str = "") -> dict[str, dict] | None:
        claims_txt = "\n".join(f"{c.claim_id}: {c.text}" for c in claims)
        ev_txt = "\n".join(f"{eid}: [{ch.doc_name}] {s}" for eid, ch, s in ev) or "(없음)"
        system, user_tmpl = PROMPTS[self.prompt_version]
        user = user_tmpl.format(question=question, claims=claims_txt, evidence=ev_txt, draft=draft)
        data = self.llm.json(system, user, max_tokens=2000, tag=prompt_hash(self.prompt_version))
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            return None
        return {str(r.get("claim_id")): r for r in data["results"] if isinstance(r, dict)}

    # ---------- 전체 ----------
    def verify(self, question: str, draft: str, chunks: list[EvidenceChunk],
               catalog_status: CatalogStatus | None = None) -> VerifyResult:
        claims, errors = self.extract(question, draft)
        ev = self.evidence_sentences(chunks)
        ev_by_id = {eid: (ch, s) for eid, ch, s in ev}
        aligned = self.align(question, claims, ev, draft)   # 추출 실패 시에도 대체 claim 으로 정렬 시도
        if aligned is None:
            errors.append("align_failed")

        vers: list[Verification] = []
        llm_codes: dict[str, list[str]] = {}
        rule_hits: dict[str, list[ReasonHit]] = {}
        for c in claims:
            r = (aligned or {}).get(c.claim_id)
            ids = [i for i in (r or {}).get("evidence_ids", []) if i in ev_by_id][:3]
            refs = [EvidenceRef(doc_id=ev_by_id[i][0].doc_id, chunk_id=ev_by_id[i][0].chunk_id,
                                quote=ev_by_id[i][1], is_latest=ev_by_id[i][0].is_latest,
                                doc_name=ev_by_id[i][0].doc_name, doc_type=ev_by_id[i][0].doc_type,
                                institution=ev_by_id[i][0].institution) for i in ids]

            if r is None:
                codes = ["__MISSING__"]              # sanitize → VERIFIER_ERROR
            else:
                code = str(r.get("code", "")).strip().upper()
                if code == ReasonCode.SUPPORTED.value and not ids:
                    codes = ["__SUPPORTED_WITHOUT_EVIDENCE__"]
                else:
                    codes = [code]
            llm_codes[c.claim_id] = codes

            rr = run_claim_rules(c.text, refs, claim_product=c.product, matcher=self.matcher,
                                 catalog_status=catalog_status, known_products=self.known_products)
            if self.promote_customer_slot and c.needs_customer_data and not rr.needs_customer_data:
                # '추출은 LLM, 판정은 코드': LLM 이 채운 슬롯(needs_customer_data=true)을
                # 규칙이 사유 코드로 바꾼다. 슬롯 출처를 detail 에 남겨 감사 시 구분한다.
                rr.hits.append(ReasonHit(ReasonCode.NEEDS_CUSTOMER_DATA, Source.RULE,
                                         "slot:needs_customer_data (LLM 추출 슬롯 기반)"))
            rule_hits[c.claim_id] = rr.hits
            fix = (r or {}).get("fix") or None
            v = build_verification(c.claim_id, codes, rr.hits, evidence=refs,
                                   rule_checks=rr.rule_checks, fix_suggestion=fix)
            # LLM 이 낸 '빠진 조건'을 사유 상세에 남긴다 (감사·진단용). 판정에는 영향 없음
            missing = [str(x) for x in (r or {}).get("missing_conditions", []) if str(x).strip()]
            for h in v.reasons:
                if h.code == ReasonCode.MISSING_CONDITION and h.source == Source.LLM:
                    h.detail = "빠진 조건: " + " / ".join(missing) if missing else "빠진 조건: (LLM 미기재)"
            vers.append(v)

        final = aggregate_final(vers, audit_id=uuid.uuid4().hex[:12])
        return VerifyResult(claims, vers, final, llm_codes, rule_hits, errors,
                            prompt_hash(self.prompt_version))


__all__ = ["CachedLLM", "ClaimVerifier", "EvidenceChunk", "VerifyResult", "prompt_hash",
           "LLM_CLAIM_EMITTABLE"]
