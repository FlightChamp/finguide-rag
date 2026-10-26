# -*- coding: utf-8 -*-
"""
v2 상담 답변 검증 — 직원 초안 검증(A) → 쉬운 설명(B) → 역검증(B) → 직원 확인

실행
----
    streamlit run app.py        # 사이드바 페이지 목록에 'v2 상담 답변 검증'이 생긴다

설계 원칙
--------
1. v1 화면(app.py)은 한 줄도 바꾸지 않는다. 이 페이지는 app.py 의 표시 유틸(테마·파이프라인 캐시)만 빌려 쓴다.
2. 판정 로직은 화면에 없다. verification.service.V2Service 가 평가 때와 같은 구성(cv-v1 · sp-v2 · bv-v1)으로
   검증·쉬운 설명·역검증을 돌리고, 이 파일은 결과를 보여 주기만 한다.
3. 쉬운 설명은 '직원 확인 보조'다. held-out 평가에서 역검증 PASS 설명의 6/26 이 사람 기준으로 틀렸다
   (docs/v2_b_heldout_eval.md). 그래서 PASS 여도 직원 확인 체크 전에는 복사용 고객 문구를 보여 주지 않는다.
4. LLM 호출은 세션마다 상한(V2_CALL_BUDGET, 기본 30)을 둔다. 상한에 닿으면 각 단계는 fail-closed
   (확인 필요 · 안내 보류)로 간다 — 틀린 안내가 아니라 안내 보류로 끝난다.
5. 결과는 session_state 에 둔다. 체크박스 조작이 재호출을 일으키지 않게 하기 위해서다.
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import app as v1  # noqa: E402  — v1 표시 유틸 (테마 · 파이프라인 캐시 · 결과 변환)

DEFAULT_BUDGET = int(os.environ.get("V2_CALL_BUDGET", "30"))
V1_ANSWER_COST = 2          # v1 답변 생성(근거 검증 + 생성)을 상한에서 차감할 대략의 호출 수

VERDICT_STYLE = {
    "OK": ("안내 가능", "#0F5132", "#E8F5EC"),
    "CHECK": ("확인 필요", "#7A5B00", "#FFF6E0"),
    "PROHIBITED": ("안내 금지", "#7F1D1D", "#FDECEC"),
}
FINAL_STYLE = {
    "OK": ("✅", "모든 내용이 근거와 일치", "#0F5132", "#E8F5EC"),
    "PARTIAL": ("⚠️", "일부 내용은 확인 필요", "#7A5B00", "#FFF6E0"),
    "CHECK": ("⚠️", "안내 가능한 내용이 확인되지 않음", "#7A5B00", "#FFF6E0"),
    "PROHIBITED": ("⛔", "근거와 다른 내용 포함 — 수정 전 안내 불가", "#7F1D1D", "#FDECEC"),
}
BV_STYLE = {
    "PASS": ("역검증 통과", "#0F5132", "#E8F5EC"),
    "BLOCK": ("쉬운 설명 보류", "#7F1D1D", "#FDECEC"),
}
REASON_KO = {
    "SUPPORTED": "근거 일치", "NO_EVIDENCE": "근거 없음", "MISSING_CONDITION": "조건 누락",
    "OVERSTATED": "과장·단정", "NEEDS_CUSTOMER_DATA": "고객정보 필요", "TIME_VARIANT": "시점 의존",
    "BLANK_VALUE": "공란·개별 약정", "AMBIGUOUS_PRODUCT": "상품명 모호", "OUTDATED_SOURCE": "구버전 근거",
    "VERIFIER_ERROR": "자동 검증 미완료", "NUMERIC_MISMATCH": "수치 불일치", "PRODUCT_MISMATCH": "다른 상품",
    "INSTITUTION_MISMATCH": "다른 기관", "CONTRADICTED": "근거와 반대",
}


# ---------------------------------------------------------------------------
# 준비
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_service(online: bool):
    v1.ensure_importable()
    v1.load_env()
    from finguide_rag.verification.service import V2Service
    client = None
    if online and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
    return V2Service.build(ROOT, client=client)


def find_report(pattern: str) -> Path | None:
    files = sorted((ROOT / "reports").glob(pattern))
    return files[-1] if files else None


@st.cache_resource(show_spinner=False)
def load_official_metrics() -> dict:
    """공식 평가 보고서에서 수치를 읽는다 (하드코딩하지 않음). 못 찾으면 빈 값."""
    out: dict[str, str] = {}
    a = find_report("v2_heldout_v1_eval_*.md")
    if a:
        t = v1.read_text(a)
        for key, label in (("오류 탐지 (N)", "A 오류 탐지"), ("정상 오탐 soft (Y)", "A 정상 답변 확인 필요"),
                           ("정상 오탐 hard (Y)", "A 정상 답변 안내 금지 오판")):
            m = re.search(re.escape(key) + r"\s+(\d+/\d+=\d\.\d\d)", t)
            if m:
                out[label] = m.group(1)
    b = find_report("v2_b_heldout_v1_eval_*.md")
    if b:
        t = v1.read_text(b)
        for key, label in (("규칙·일반 교란 False Pass", "B 교란 False Pass"),
                           ("조건 교란 False Pass", "B 조건 교란 False Pass"),
                           ("과차단 (사람 Y 인데 PASS 아님)", "B 과차단"),
                           ("자연 오류 놓침 (사람 N 인데 PASS)", "B 실제 오류 놓침")):
            m = re.search(re.escape(key) + r"[^\n]*?(\d+/\d+=\d\.\d\d)", t)
            if m:
                out[label] = m.group(1)
    return out


def hits_to_evidences(pipeline, question: str) -> list[dict]:
    """LLM 없이 검색만 해서 근거를 만든다 (직접 입력 모드). v1 Answer 와 같은 Evidence 형식."""
    from finguide_rag.generation.generator import Evidence
    hits = pipeline.retriever.search(question, top_k=pipeline.top_k)
    return [asdict(Evidence.from_hit(h, i)) for i, h in enumerate(hits[:pipeline.evidence_n], 1)]


# ---------------------------------------------------------------------------
# 렌더링
# ---------------------------------------------------------------------------

def badge(text: str, fg: str, bg: str) -> str:
    return (f'<span style="background:{bg};color:{fg} !important;padding:2px 8px;border-radius:6px;'
            f'font-size:0.85rem;font-weight:600;white-space:nowrap">{v1.esc(text)}</span>')


def render_header() -> None:
    st.markdown("""
    <div class="fg-banner">
      <h1>FinGuide-RAG v2</h1>
      <div class="rule"></div>
      <div class="sub">상담 답변 검증 · 쉬운 설명 · 역검증 (직원용)</div>
      <p class="desc">
        직원이 작성한 답변 초안을 문장(claim) 단위로 근거 원문과 대조하고, 안내 가능한 내용만 고객용 쉬운 말로
        바꾼 뒤 뜻이 그대로인지 다시 검사합니다. 최종 안내 여부는 직원이 판단합니다.
      </p>
    </div>
    <div class="fg-notice">
      ※ 공개 문서 기반 개인 포트폴리오 프로토타입이며 어떤 금융기관의 공식 서비스가 아닙니다.
      역검증을 통과한 쉬운 설명도 직원 확인 없이 고객에게 그대로 안내하면 안 됩니다.
    </div>
    """, unsafe_allow_html=True)


def render_sidebar() -> bool:
    with st.sidebar:
        st.markdown('<div class="fg-card-title">공식 held-out 평가</div>', unsafe_allow_html=True)
        metrics = load_official_metrics()
        if metrics:
            for label, value in metrics.items():
                st.markdown(f"""
                <div class="fg-metric">
                  <div class="m-label">{v1.esc(label)}</div>
                  <div class="m-value">{v1.esc(value)}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('<div class="fg-hint">reports/ 의 공식 평가(1회) 결과에서 읽습니다. 표본이 작아 '
                        '구간이 넓습니다. B 실제 오류 놓침이 높아 직원 확인이 필수입니다.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="fg-hint">reports/ 에서 공식 평가 결과를 찾지 못했습니다.</div>',
                        unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="fg-card-title">실행 설정</div>', unsafe_allow_html=True)
        offline = st.checkbox("오프라인 모드 (캐시된 결과만)", value=not bool(os.environ.get("OPENAI_API_KEY")),
                              help="새 LLM 호출을 하지 않습니다. 캐시에 없는 단계는 확인 필요·안내 보류로 처리됩니다.")
        used = st.session_state.get("v2_calls_used", 0)
        st.markdown(f'<div class="fg-hint">이 세션 LLM 호출 {used} / {DEFAULT_BUDGET}회<br>'
                    '상한에 닿으면 새 호출 없이 안전한 쪽(확인 필요·안내 보류)으로 처리합니다.</div>',
                    unsafe_allow_html=True)
        return offline


def render_claims(res) -> None:
    final = res.verify.final
    icon, detail, fg, bg = FINAL_STYLE[final.final.value]
    st.markdown(f"""
    <div class="fg-panel fg-panel-verdict">
      <div class="fg-verdict-head" style="background:{bg};color:{fg};">
        <span>{icon}</span><span><span class="v-label">초안 판정: {v1.esc(final.final.value)}</span>
        <span class="v-detail"> — {v1.esc(detail)}</span></span>
      </div>
    </div>""", unsafe_allow_html=True)
    s = final.claims_summary
    st.caption(f"claim {sum(s.values())}개 · 안내 가능 {s.get('OK', 0)} · 확인 필요 {s.get('CHECK', 0)} · "
               f"안내 금지 {s.get('PROHIBITED', 0)}")
    if final.staff_warnings:
        with st.expander("직원 주의사항", expanded=final.final.value != "OK"):
            for w in final.staff_warnings:
                st.markdown(f"- {w}")
    claims = {c.claim_id: c for c in res.verify.claims}
    for v in res.verify.verifications:
        c = claims.get(v.claim_id)
        label, fg, bg = VERDICT_STYLE[v.verdict.value]
        reasons = [h for h in v.reasons if h.code.value != "SUPPORTED"]
        tags = " ".join(badge(REASON_KO.get(h.code.value, h.code.value), "#334155", "#EEF2F6") for h in reasons)
        st.markdown(f'<div style="margin:6px 0">{badge(label, fg, bg)} {tags}<br>'
                    f'<span>{v1.esc(c.text if c else v.claim_id)}</span></div>', unsafe_allow_html=True)
        if v.evidence or reasons:
            with st.expander(f"{v.claim_id} 근거 · 사유", expanded=False):
                for e in v.evidence:
                    st.markdown(f"**{v1.esc(e.doc_name or e.doc_id)}**")
                    st.markdown(f"> {e.quote}")
                for h in reasons:
                    st.caption(f"[{h.code.value}] {h.detail}" if h.detail else f"[{h.code.value}]")
                if v.fix_suggestion:
                    st.caption(f"수정 제안(LLM, 참고용): {v.fix_suggestion}")


def render_simplify(res) -> None:
    out = res.simplify
    if out.status is None:
        st.info("안내 가능(OK)으로 확인된 내용이 없어 쉬운 설명을 만들지 않았습니다. 원문 근거를 직접 확인해 주세요.")
        return
    label, fg, bg = BV_STYLE[out.status.value]
    attempts = out.history[-1][1].attempts if out.history else 0
    st.markdown(f"{badge(label, fg, bg)} " + (badge("1회 자동 수정 후", "#334155", "#EEF2F6") if attempts else ""),
                unsafe_allow_html=True)
    if out.status.value == "PASS":
        e = out.explanation
        st.markdown("**고객 안내 초안** (직원 확인 전 사용 금지)")
        for it in e.core:
            st.markdown(f"- {it.text}")
        if e.conditions:
            st.markdown("**꼭 확인하세요**")
            for it in e.conditions:
                st.markdown(f"- {it.text}")
        st.caption("held-out 평가에서 역검증을 통과한 설명 중에도 사람 기준 오류가 있었습니다 "
                   "(가능→의무 전환, '만' 삽입, 모순 등). 아래 OK claim 원문과 대조해 확인하세요.")
    else:
        st.warning("쉬운 설명이 원문 뜻을 그대로 담았는지 확인하지 못해 보류했습니다. 대신 안내 가능 내용의 원문을 보여 드립니다.")
        for line in out.fallback_lines:
            st.markdown(f"- {line}")
        problems = [p for _, bv in out.history for p in bv.problems]
        if problems:
            with st.expander("보류 사유 (역검증 문제)", expanded=False):
                for p in problems:
                    st.caption(f"[{p.code.value}] {p.detail}")


def render_staff_area(res) -> None:
    e = res.simplify.explanation
    if e.staff_check:
        st.markdown("**직원 확인 필요** — 고객 안내문에서 제외됨")
        for s in e.staff_check:
            tags = ", ".join(REASON_KO.get(r, r) for r in s.reasons)
            st.markdown(f"- {s.text}  \n  <span class='fg-hint'>{v1.esc(tags)}</span>", unsafe_allow_html=True)
    if e.excluded:
        claims = {c.claim_id: c.text for c in res.verify.claims}
        st.markdown("**안내 금지** — 근거와 다른 내용, 고객에게 안내하지 마세요")
        for cid in e.excluded:
            st.markdown(f"- ~~{claims.get(cid, cid)}~~")
    if not e.staff_check and not e.excluded:
        st.caption("직원 확인 필요 · 안내 금지 항목 없음")


def render_confirm(res) -> None:
    """직원 확인 게이트: 체크 전에는 복사용 고객 문구를 보여 주지 않는다."""
    out = res.simplify
    if not out.customer_visible:
        return
    st.markdown("---")
    ok = st.checkbox("위 고객 안내 초안을 근거 원문과 대조해 확인했습니다 (뜻·조건·수치)", key="v2_staff_confirmed")
    if ok:
        st.success("직원 확인 완료 — 아래 문구를 복사해 사용할 수 있습니다.")
        st.code(out.explanation.customer_text(), language=None)
    else:
        st.caption("확인 체크 전에는 복사용 문구를 표시하지 않습니다.")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def fetch_evidence(question: str, offline: bool, use_v1_answer: bool) -> None:
    st.session_state["v2_error"] = ""
    st.session_state["v2_result"] = None
    st.session_state["v2_staff_confirmed"] = False
    try:
        with st.spinner("근거 검색 중…" if not use_v1_answer else "근거 검색 · v1 답변 생성 중…"):
            pipeline = v1.get_pipeline(use_llm=not offline)
            if use_v1_answer:
                answer = pipeline.answer(question)
                result = v1.enrich_evidences(v1.to_result_dict(answer), answer)
                st.session_state["v2_evidences"] = v1.evidence_list(result)
                if v1.derive_status(result) == "refused":
                    st.session_state["v2_draft"] = ""
                    st.session_state["v2_notice"] = ("v1 이 이 질문에 대한 답변을 보류했습니다. "
                                                     "초안을 직접 작성하거나 질문을 바꿔 주세요.")
                else:
                    st.session_state["v2_draft"] = str(result.get("answer", ""))
                    st.session_state["v2_notice"] = "v1 답변을 초안으로 불러왔습니다. 수정한 뒤 검증할 수 있습니다."
                if not offline:
                    st.session_state["v2_calls_used"] = st.session_state.get("v2_calls_used", 0) + V1_ANSWER_COST
            else:
                st.session_state["v2_evidences"] = hits_to_evidences(pipeline, question)
                st.session_state["v2_notice"] = "근거를 검색했습니다 (LLM 미사용). 초안을 입력한 뒤 검증하세요."
        st.session_state["v2_question"] = question
    except Exception as exc:
        st.session_state["v2_error"] = f"{type(exc).__name__}: {exc}\n\n" + traceback.format_exc()[-1500:]


def run_verification(offline: bool) -> None:
    st.session_state["v2_error"] = ""
    st.session_state["v2_staff_confirmed"] = False      # 새 결과에 이전 확인이 승계되면 안 된다
    left = max(0, DEFAULT_BUDGET - st.session_state.get("v2_calls_used", 0))
    try:
        with st.spinner("claim 검증 → 쉬운 설명 → 역검증 중…"):
            svc = get_service(online=not offline)
            res = svc.run(st.session_state["v2_question"], st.session_state["v2_draft"].strip(),
                          st.session_state.get("v2_evidences", []), max_calls=0 if offline else left)
        st.session_state["v2_calls_used"] = st.session_state.get("v2_calls_used", 0) + res.api_calls
        st.session_state["v2_result"] = res
    except Exception as exc:
        st.session_state["v2_result"] = None
        st.session_state["v2_error"] = f"{type(exc).__name__}: {exc}\n\n" + traceback.format_exc()[-1500:]


def main() -> None:
    st.set_page_config(page_title="FinGuide-RAG v2", layout="wide", initial_sidebar_state="expanded")
    v1.apply_custom_theme()
    for k, v in (("v2_question", ""), ("v2_draft", ""), ("v2_evidences", []), ("v2_result", None),
                 ("v2_error", ""), ("v2_notice", ""), ("v2_calls_used", 0)):
        st.session_state.setdefault(k, v)

    render_header()
    offline = render_sidebar()
    left, right = st.columns([0.48, 0.52], gap="large")

    with left:
        st.markdown('<div class="fg-card-title">1. 고객 질문과 근거</div>', unsafe_allow_html=True)
        question = st.text_input("고객 질문", key="v2_question_input",
                                 placeholder="예: 정기예금을 만기 전에 해지하면 이자는 어떻게 되나요?")
        source = st.radio("초안 준비 방식", ["직접 작성", "v1 답변 불러오기"], horizontal=True, key="v2_source",
                          help="직접 작성: 근거만 검색하고(LLM 미사용) 직원이 초안을 씁니다. "
                               "v1 답변 불러오기: v1 파이프라인이 만든 답변을 초안으로 씁니다.")
        if st.button("근거 검색" if source == "직접 작성" else "근거 검색 + v1 답변 불러오기", key="v2_fetch"):
            if not question.strip():
                st.warning("질문을 입력하세요.")
            else:
                fetch_evidence(question.strip(), offline, source == "v1 답변 불러오기")
                st.session_state["v2_draft_input"] = st.session_state["v2_draft"]
        if st.session_state["v2_notice"] and st.session_state["v2_question"]:
            st.caption(st.session_state["v2_notice"])

        if st.session_state["v2_question"]:
            st.markdown('<div class="fg-card-title">2. 직원 답변 초안</div>', unsafe_allow_html=True)
            draft = st.text_area("초안", key="v2_draft_input", height=160,
                                 placeholder="고객에게 안내할 답변 초안을 입력하세요.")
            if st.button("검증 · 쉬운 설명 실행", key="v2_run", type="primary"):
                if not draft.strip():
                    st.warning("초안을 입력하세요.")
                else:
                    st.session_state["v2_draft"] = draft
                    run_verification(offline)

        if st.session_state["v2_error"]:
            st.error("실행에 실패했습니다.")
            with st.expander("오류 상세"):
                st.code(st.session_state["v2_error"])

        res = st.session_state["v2_result"]
        if res:
            for n in res.notes:
                st.warning(n)
            st.markdown('<div class="fg-card-title">3. 초안 검증 (claim 단위)</div>', unsafe_allow_html=True)
            render_claims(res)

    with right:
        res = st.session_state["v2_result"]
        if res:
            st.markdown('<div class="fg-card-title">4. 고객용 쉬운 설명 · 역검증</div>', unsafe_allow_html=True)
            render_simplify(res)
            st.markdown('<div class="fg-card-title">5. 직원 확인 영역</div>', unsafe_allow_html=True)
            render_staff_area(res)
            render_confirm(res)
        evs = st.session_state["v2_evidences"]
        if evs:
            with st.expander(f"검색된 근거 {len(evs)}건", expanded=not bool(res)):
                for e in evs:
                    st.markdown(f"**{v1.esc(e.get('citation') or e.get('chunk_id'))}** "
                                f"<span class='fg-hint'>{v1.esc(e.get('doc_type', ''))}</span>",
                                unsafe_allow_html=True)
                    st.markdown(v1.truncate_text(e.get("text", ""), 800))


main()
