# -*- coding: utf-8 -*-
"""
app.py — FinGuide-RAG 데모 콘솔

실행
----
    streamlit run app.py

설계 원칙
--------
1. 이 파일은 표시 계층이다. 검색·거절·생성 로직을 재구현하거나 수정하지
   않는다. Pipeline.answer() 를 호출하고 결과를 렌더링할 뿐이다.

2. 하드코딩된 답변·근거·상태가 없다. 예시 질문 버튼은 입력창을 채우는
   프리셋일 뿐이며, 화면에 보이는 모든 값은 실제 파이프라인 산출물이다.

3. 상품 일치 여부를 UI 가 판정하지 않는다. 자동 상품 판정은 오탐률이
   높다는 것이 이미 측정됐다(20건 중 16건). 따라서 질문에서 인식된
   상품과 근거 문서의 상품을 나란히 제시하고 판단은 직원에게 맡긴다.

4. Streamlit 은 위젯 조작마다 스크립트를 다시 실행한다. 결과는 전부
   session_state 에 보관해 체크박스나 토글 조작이 재호출을 일으키지
   않게 한다.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# 경로·환경
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "interim" / "product_catalog.json"

#: 예시 질문 후보. 실제로 돌려본 뒤 대표 5개로 추리는 것을 전제로 한다.
EXAMPLE_QUESTIONS = [
    "하나은행 인터넷뱅킹에서 1천만원 넘게 이체하려면 어떻게 해야 하나요?",
    "은행이 돈을 다 갚으라고 통보를 늦게 하면 저는 언제부터 바로 갚아야 하나요?",
    "제 개인 신용등급이 대출 금리에 어떻게 반영되나요?",
    "지금 하나은행 신용대출 금리는 몇 퍼센트인가요?",
    "정기예금을 중도해지하면 이자는 어떻게 계산되나요?",
    "자유적금 만기 전에 해지하면 우대금리는 어떻게 되나요?",
]

STATUS_STYLE = {
    "answered": ("✅ 답변 가능 — 공개 문서 근거 확인됨", "success"),
    "partial": ("⚠ 일부 항목 미확인 — 문서에서 확인되지 않은 내용이 있습니다", "warning"),
    "refused": ("⛔ 답변 보류 — 근거가 불충분하거나 실시간·개인정보 조회가 필요한 질문입니다", "error"),
}

REFUSAL_LABEL = {
    "personalized": "개인 정보·이력 조회가 필요한 질문",
    "time_variant": "시점에 따라 답이 달라지는 질문 (현재 금리 등)",
    "out_of_scope": "보유 문서 범위를 벗어난 질문",
    "blank_value": "근거 문서의 해당 값이 공란",
    "low_confidence": "검색 신뢰도가 기준에 미달",
}

STAGE_LABEL = {
    "pattern": "0단계 · 질문 패턴 (LLM 호출 없음)",
    "retrieval": "1단계 · 검색 신호 (LLM 호출 없음)",
    "llm": "2단계 · LLM 근거 검증",
    "no_generator": "생성기 미연결 (근거만 표시)",
}

_RE_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")


def load_env() -> None:
    """.env 를 직접 파싱한다. 값은 화면에 절대 표시하지 않는다."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    path = ROOT / ".env"
    if not path.exists():
        return
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def ensure_importable() -> None:
    """src 레이아웃 패키지를 임포트 가능하게 한다."""
    try:
        import finguide_rag  # noqa: F401
        return
    except ImportError:
        pass
    src = ROOT / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """dict / dataclass / 일반 객체에서 값을 안전하게 꺼낸다."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    return default if value is None else value


def to_result_dict(result: Any) -> dict:
    """파이프라인 산출물을 공통 dict 로 변환한다.

    Answer.to_dict() 가 있으면 그것을 쓰고, 없으면 dataclass·pydantic·
    일반 객체를 차례로 시도한다. 어떤 경우에도 예외를 던지지 않는다.
    """
    if result is None:
        return {}

    for attr in ("to_dict", "model_dump", "dict"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    if is_dataclass(result) and not isinstance(result, type):
        try:
            return asdict(result)
        except Exception:
            pass

    if isinstance(result, dict):
        return dict(result)

    return {k: v for k, v in vars(result).items()
            if not k.startswith("_")} if hasattr(result, "__dict__") else {}


def truncate_text(text: str, max_chars: int = 3000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n… (이하 {len(text) - max_chars:,}자 생략)"


def format_score(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def sanitize_for_debug(data: Any, text_limit: int = 800) -> Any:
    """디버그 표시용 정리. 비밀 값 제거, 긴 원문 절단."""
    secret = ("api_key", "apikey", "token_secret", "password", "secret",
              "authorization", "openai_api_key")

    def walk(node: Any, depth: int = 0) -> Any:
        if depth > 8:
            return "…"
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if any(s in str(k).lower() for s in secret):
                    out[k] = "***"
                elif str(k).lower() in ("text", "chunk_text", "raw_text"):
                    out[k] = truncate_text(str(v), text_limit)
                else:
                    out[k] = walk(v, depth + 1)
            return out
        if isinstance(node, list):
            return [walk(v, depth + 1) for v in node[:20]]
        if isinstance(node, (str, int, float, bool)) or node is None:
            return node
        return str(node)[:200]

    return walk(data)


def derive_status(result: dict) -> str:
    """answered / partial / refused 를 결정한다.

    partial 은 파이프라인이 직접 내보내는 값이 아니라, 생성기가
    '질문이 물었으나 근거에서 확인되지 않은 항목'으로 신고한
    not_found 가 비어있지 않은 경우로 정의한다.
    """
    decision = str(safe_get(result, "decision", "") or "").lower()
    if decision in ("refuse", "refused", "거절"):
        return "refused"
    if safe_get(result, "not_found") or []:
        return "partial"
    return "answered"


def evidence_list(result: dict) -> list[dict]:
    items = safe_get(result, "evidences", []) or []
    return [e if isinstance(e, dict) else to_result_dict(e) for e in items]


def evidence_field(ev: dict, *names: str, default: str = "") -> str:
    """필드명이 구현마다 다를 수 있으므로 후보를 순서대로 찾는다."""
    for name in names:
        value = ev.get(name)
        if value not in (None, ""):
            return str(value)
    return default


# ---------------------------------------------------------------------------
# 무거운 리소스
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="인덱스와 모델을 불러오는 중입니다…")
def get_pipeline(use_llm: bool):
    """파이프라인을 조립한다. use_llm=False 면 LLM 없이 검색+규칙만 쓴다."""
    ensure_importable()
    load_env()
    from finguide_rag.generation.pipeline import Pipeline

    client = None
    if use_llm:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY 를 찾지 못했습니다. .env 를 확인하거나 "
                "사이드바에서 오프라인 모드를 켜 주세요."
            )
        from openai import OpenAI
        client = OpenAI()

    return Pipeline.build(ROOT, client=client, use_llm=use_llm)


@st.cache_resource(show_spinner=False)
def get_product_tools():
    """상품 어절 대조 도구. 카탈로그가 없으면 (None, []) 를 돌려준다."""
    ensure_importable()
    try:
        from finguide_rag.generation.product_match import ProductMatcher
        from finguide_rag.generation.product_catalog import ProductCatalog
    except Exception:
        return None, []

    if not CATALOG_PATH.exists():
        return None, []

    try:
        matcher = ProductMatcher.from_catalog(CATALOG_PATH)
        catalog = ProductCatalog.load(CATALOG_PATH)
        names = [e.canonical_name for e in catalog.all_named]
        return matcher, names
    except Exception:
        return None, []


@st.cache_resource(show_spinner=False)
def load_metrics() -> dict:
    """reports/ 에서 측정된 지표를 찾아온다. 하드코딩하지 않는다."""

    def deep_find(node: Any, keys: tuple[str, ...], depth: int = 0):
        if depth > 8:
            return None
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).lower() in keys and isinstance(v, (int, float)):
                    return v
            for v in node.values():
                found = deep_find(v, keys, depth + 1)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for v in node[:50]:
                found = deep_find(v, keys, depth + 1)
                if found is not None:
                    return found
        return None

    def newest(pattern: str):
        hits = sorted((ROOT / "reports").glob(pattern),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for path in hits:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
        return None

    out: dict[str, Any] = {}

    hybrid = newest("hybrid_tuning_final_doclevel_*.json")
    if isinstance(hybrid, dict):
        results = hybrid.get("results", {})
        chosen = None
        for key, val in results.items():
            if "weighted" in str(key) and "0.5" in str(key) and "z" not in str(key):
                chosen = val
                break
        overall = (chosen or {}).get("overall", {}) if isinstance(chosen, dict) else {}
        if overall:
            out["문서 Recall@5"] = overall.get("recall@5")
            out["MRR"] = overall.get("mrr")
            out["검색 n"] = overall.get("n")

    refusal = newest("refusal_eval_*.json")
    if refusal is not None:
        far = deep_find(refusal, ("false_answer_rate", "far"))
        bal = deep_find(refusal, ("balanced_accuracy", "balacc"))
        if far is not None:
            out["False Answer Rate"] = far
        if bal is not None:
            out["Balanced Accuracy"] = bal

    return out


# ---------------------------------------------------------------------------
# 상품 표시 (판정하지 않는다)
# ---------------------------------------------------------------------------

def question_products(matcher, names: list[str], question: str) -> list[str]:
    """질문에 언급된 카탈로그 상품을 찾는다.

    ProductMatcher 가 카탈로그에서 학습한 '변별력 없는 어절' 목록을 그대로
    쓴다. 판정이 아니라 표시용이므로, 상품의 변별력 있는 어절이 모두
    질문에 접두 일치로 등장할 때만 인식한다(보수적).
    """
    if matcher is None or not names or not question:
        return []

    q_tokens = [t.lower() for t in _RE_TOKEN.findall(question) if len(t) >= 2]
    if not q_tokens:
        return []

    def prefix_hit(token: str) -> bool:
        return any(token.startswith(q) or q.startswith(token) for q in q_tokens)

    found: list[str] = []
    for name in names:
        try:
            informative = matcher.informative(name)
        except Exception:
            continue
        if not informative:
            continue
        if all(prefix_hit(t) for t in informative):
            found.append(name)
    return found


def evidence_products(matcher, evidences: list[dict]) -> list[str]:
    """근거 문서에서 상품명을 뽑아 중복을 제거한다."""
    if matcher is None or not evidences:
        return []
    seen: list[str] = []
    for ev in evidences:
        name = evidence_field(ev, "doc_display_name", "citation")
        if not name:
            continue
        try:
            product = matcher.product_of(name)
        except Exception:
            product = name
        product = (product or "").strip()
        if product and product not in seen:
            seen.append(product)
    return seen


# ---------------------------------------------------------------------------
# 렌더링
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.title("FinGuide-RAG")
    st.caption("은행 직원용 근거 제시형 금융 RAG 콘솔")
    st.write(
        "상품설명서·약관·FAQ 원문 근거를 함께 제공하여 직원이 직접 검증할 수 "
        "있도록 설계된 B2E RAG 데모입니다."
    )
    st.info(
        "※ 본 데모는 공개 문서 기반 프로토타입이며, 실제 고객 안내 전에는 "
        "은행 내부 시스템과 담당 부서 확인이 필요합니다.",
        icon="ℹ️",
    )


def render_sidebar() -> tuple[bool, bool]:
    with st.sidebar:
        st.subheader("측정된 성능")
        metrics = load_metrics()
        if metrics:
            for label, value in metrics.items():
                if value is None:
                    continue
                shown = f"{value:.3f}" if isinstance(value, float) else str(value)
                st.metric(label, shown)
            st.caption("reports/ 의 최신 평가 결과에서 읽어옵니다.")
        else:
            st.caption("reports/ 에서 평가 결과를 찾지 못했습니다.")

        st.divider()
        st.subheader("실행 설정")
        offline = st.checkbox(
            "오프라인 모드 (LLM 없이 검색·규칙만)",
            value=False,
            help="LLM 근거 검증과 답변 생성을 끕니다. 검색 결과와 규칙 기반 "
                 "거절만 확인할 때 씁니다.",
        )
        debug = st.checkbox("엔지니어링 메타데이터 보기", value=False)
        return offline, debug


def render_example_buttons() -> None:
    st.caption("예시 질문")
    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(question, key=f"ex_{i}", use_container_width=True):
                st.session_state["question_input"] = question


def render_status_badge(result: dict, evidences: list[dict]) -> None:
    status = derive_status(result)
    message, kind = STATUS_STYLE[status]
    getattr(st, kind)(message)

    doc_types = {evidence_field(e, "doc_type") for e in evidences}
    doc_types.discard("")
    if doc_types and doc_types <= {"약관", "FAQ"}:
        st.info("📘 공통 규정 근거 — 특정 상품이 아닌 공통 약관·FAQ 기준입니다.")

    if any(e.get("is_latest") is False for e in evidences):
        st.warning("🕓 최신본이 아닐 수 있는 근거가 포함되어 있습니다. 시행일을 확인하세요.")


def render_source_notice(result: dict, evidences: list[dict],
                         question: str) -> str:
    """답변 최상단 근거 안내문을 렌더링하고, 복사용 텍스트를 돌려준다."""
    notice = str(safe_get(result, "source_notice", "") or "").strip()

    if not notice:
        titles: list[str] = []
        for ev in evidences:
            name = evidence_field(ev, "doc_display_name", "citation")
            if name and name not in titles:
                titles.append(name)
        if titles:
            joined = ", ".join(f"『{t}』" for t in titles)
            notice = f"※ 이 답변은 다음 근거 문서에 기반합니다: {joined}."
        else:
            notice = "※ 표시할 근거 문서가 없습니다."

    st.markdown(f"**{notice}**")

    matcher, names = get_product_tools()
    q_products = question_products(matcher, names, question)
    e_products = evidence_products(matcher, evidences)

    if q_products or e_products:
        left = ", ".join(f"『{p}』" for p in q_products) if q_products else "특정 상품 미인식"
        right = ", ".join(f"『{p}』" for p in e_products) if e_products else "없음"
        st.markdown(
            f"- 질문에서 인식된 상품 *(카탈로그 대조, 참고용)*: {left}\n"
            f"- 근거 문서의 상품: {right}"
        )
        st.caption(
            "두 값이 다르면 우측 근거 원문을 확인하세요. 자동 상품 판정은 "
            "오탐률이 높은 것으로 측정되어(20건 중 16건), 시스템이 불일치를 "
            "단정하지 않고 두 값을 그대로 제시합니다."
        )

    return notice


def render_answer_panel(result: dict, evidences: list[dict],
                        question: str) -> None:
    render_status_badge(result, evidences)
    notice = render_source_notice(result, evidences, question)

    answer_text = str(safe_get(result, "answer", "") or "")
    st.markdown("#### 답변")
    st.write(answer_text or "(답변 본문 없음)")

    reason = safe_get(result, "refusal_reason")
    stage = str(safe_get(result, "stage", "") or "")
    if reason:
        label = REFUSAL_LABEL.get(str(reason), str(reason))
        st.markdown(f"**거절 사유** · `{reason}` — {label}")
    if stage:
        st.caption(f"판정 단계: {STAGE_LABEL.get(stage, stage)}")

    not_found = safe_get(result, "not_found") or []
    if not_found:
        st.markdown("**문서에서 확인되지 않은 항목**")
        for item in not_found:
            st.markdown(f"- {item}")
        st.caption("위 항목은 근거 문서에 없어 답변에 포함하지 않았습니다. "
                   "내부 시스템이나 담당 부서 확인이 필요합니다.")

    st.divider()
    checked = st.checkbox("우측 근거 원문을 확인했습니다.", key="evidence_checked")
    if not checked:
        st.warning("고객 안내 전 우측 근거 문서를 확인하는 것을 권장합니다.")
        return

    copy_text = f"{notice}\n\n{answer_text}"
    if not_found:
        copy_text += "\n\n[문서에서 확인되지 않은 항목]\n" + "\n".join(
            f"- {x}" for x in not_found)
    st.markdown("#### 고객 안내용")
    st.text_area("복사해서 사용하세요", value=copy_text, height=200,
                 label_visibility="collapsed")


def render_evidence_panel(evidences: list[dict], result: dict) -> None:
    st.markdown("### 근거 원문")

    if not evidences:
        stage = str(safe_get(result, "stage", "") or "")
        if stage == "pattern":
            st.info(
                "이 질문은 검색 이전 단계(질문 패턴)에서 판정되었습니다.\n\n"
                "검색과 LLM 을 모두 호출하지 않았으므로 표시할 근거가 없습니다. "
                "의도된 동작입니다."
            )
        else:
            st.warning("표시할 근거가 없습니다.")
        return

    for i, ev in enumerate(evidences):
        title = evidence_field(ev, "citation", "doc_display_name",
                               default="(문서명 없음)")
        rank = ev.get("rank", i + 1)
        header = f"근거 {rank} · {title}"

        body = lambda ev=ev: _render_evidence_body(ev)
        if i == 0:
            st.markdown(f"**{header}**")
            body()
        else:
            with st.expander(header, expanded=False):
                body()


def _render_evidence_body(ev: dict) -> None:
    cols = st.columns(3)
    cols[0].caption(f"문서유형 · {evidence_field(ev, 'doc_type', default='N/A')}")
    cols[1].caption(
        "시행일 · "
        + evidence_field(ev, "effective_date", "valid_from", "date", "시행일",
                         default="N/A")
    )
    cols[2].caption(f"score · {format_score(ev.get('score'))}")

    section = evidence_field(ev, "section")
    if section:
        st.caption(f"위치 · {section}")

    ids = " / ".join(filter(None, [
        f"doc_id: {evidence_field(ev, 'doc_id', default='N/A')}",
        f"chunk_id: {evidence_field(ev, 'chunk_id', default='N/A')}",
    ]))
    st.caption(ids)

    if ev.get("is_latest") is False:
        st.warning("최신본이 아닐 수 있습니다.")

    st.text(truncate_text(evidence_field(ev, "text"), 3000) or "(원문 없음)")


def render_debug_panel(result: dict, evidences: list[dict]) -> None:
    st.divider()
    st.markdown("### 엔지니어링 메타데이터")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["검색 점수", "거절 판정", "처리 시간", "raw result"])

    with tab1:
        if evidences:
            rows = [{
                "rank": e.get("rank"),
                "doc_title": evidence_field(e, "doc_display_name", "citation",
                                            default="N/A"),
                "hybrid_score": format_score(e.get("score")),
                "doc_type": evidence_field(e, "doc_type", default="N/A"),
                "effective_date": evidence_field(
                    e, "effective_date", "valid_from", "date", default="N/A"),
                "is_latest": e.get("is_latest", "N/A"),
            } for e in evidences]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(
                "hybrid_score 는 weighted(min-max) α=0.5 로 결합된 최종 점수입니다. "
                "dense·sparse 개별 점수는 현재 파이프라인이 반환하지 않습니다."
            )
        else:
            st.caption("근거가 없어 표시할 점수가 없습니다.")

    with tab2:
        stage = str(safe_get(result, "stage", "") or "N/A")
        st.markdown(
            f"- **최종 상태**: `{derive_status(result)}`\n"
            f"- **decision**: `{safe_get(result, 'decision', 'N/A')}`\n"
            f"- **판정 단계**: `{stage}` — {STAGE_LABEL.get(stage, 'N/A')}\n"
            f"- **거절 사유**: `{safe_get(result, 'refusal_reason') or 'N/A'}`"
        )
        signals = safe_get(result, "signals") or {}
        if signals:
            st.markdown("**거절 판정에 쓰인 신호**")
            st.dataframe(
                [{"신호": k, "값": v} for k, v in sanitize_for_debug(signals).items()],
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("신호 값이 반환되지 않았습니다.")

    with tab3:
        st.markdown(
            f"- **total_ms**: {safe_get(result, 'latency_ms', 'N/A')}\n"
            f"- **tokens**: {safe_get(result, 'tokens', 'N/A')}"
        )
        st.caption(
            "단계별 소요 시간(retrieval/refusal/generation)은 현재 파이프라인이 "
            "계측하지 않습니다."
        )

    with tab4:
        st.json(sanitize_for_debug(result))
        st.caption("근거 원문은 800자로 잘랐습니다. 전체는 우측 근거 패널에서 "
                   "확인하세요.")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="FinGuide-RAG", layout="wide",
                       initial_sidebar_state="expanded")

    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("asked_question", "")
    st.session_state.setdefault("error", "")

    render_header()
    offline, debug = render_sidebar()
    st.divider()

    left, right = st.columns([0.45, 0.55])

    with left:
        render_example_buttons()
        st.text_area("질문", key="question_input", height=100,
                     placeholder="고객 문의 내용을 입력하세요")
        run = st.button("답변 생성", type="primary", use_container_width=True)

        if run:
            question = st.session_state["question_input"].strip()
            if not question:
                st.warning("질문을 입력하세요.")
            else:
                # 새 질문마다 근거 확인 체크를 반드시 해제한다.
                # 이전 답변에 대한 확인이 새 답변으로 승계되면 이 화면의
                # 목적 자체가 무너진다.
                st.session_state["evidence_checked"] = False
                st.session_state["error"] = ""
                try:
                    with st.status("처리 중…", expanded=False) as status:
                        status.update(label="파이프라인 준비 중…")
                        pipeline = get_pipeline(use_llm=not offline)
                        status.update(label="문서 검색 · 거절 판정 · 답변 생성 중…")
                        answer = pipeline.answer(question)
                        status.update(label="완료", state="complete")
                    st.session_state["result"] = to_result_dict(answer)
                    st.session_state["asked_question"] = question
                except Exception as exc:
                    st.session_state["result"] = None
                    st.session_state["error"] = (
                        f"{type(exc).__name__}: {exc}\n\n"
                        + "".join(traceback.format_exc()[-1500:])
                    )

        if st.session_state["error"]:
            st.error("파이프라인 실행에 실패했습니다.")
            with st.expander("오류 상세"):
                st.code(st.session_state["error"])
            st.caption("인덱스가 없거나 API 키가 설정되지 않았을 수 있습니다. "
                       "사이드바의 오프라인 모드로 검색만 확인할 수 있습니다.")

        result = st.session_state["result"]
        if result:
            st.divider()
            render_answer_panel(result, evidence_list(result),
                                st.session_state["asked_question"])

    with right:
        result = st.session_state["result"]
        if result:
            render_evidence_panel(evidence_list(result), result)
        else:
            st.markdown("### 근거 원문")
            st.caption("답변을 생성하면 참조한 근거 문서의 원문이 여기에 "
                       "표시됩니다.")

    if debug and st.session_state["result"]:
        render_debug_panel(st.session_state["result"],
                           evidence_list(st.session_state["result"]))


if __name__ == "__main__":
    main()
