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

2. 하드코딩된 답변·근거·상태가 없다. 예시 질문은 입력창을 채우는
   프리셋일 뿐이며, 화면에 보이는 모든 값은 실제 파이프라인 산출물이다.

3. 상품 일치 여부를 UI 가 판정하지 않는다. 자동 상품 판정은 오탐률이
   높다는 것이 이미 측정됐다(20건 중 16건). 따라서 질문에서 인식된
   상품과 근거 문서의 상품을 나란히 제시하고 판단은 직원에게 맡긴다.

4. Streamlit 은 위젯 조작마다 스크립트를 다시 실행한다. 결과는 전부
   session_state 에 보관해 체크박스나 토글 조작이 재호출을 일으키지
   않게 한다.

테마 규칙
--------
이 앱은 라이트/다크 어느 쪽에서도 같은 모습으로 보이도록 CSS 에서 배경과
글자색을 모두 명시한다. config.toml 에 [theme] 을 두면 Streamlit 설정
메뉴의 테마 전환이 사라지므로 두지 않는다.

CSS 우선순위 주의
---------------
다크모드에서 글자가 사라지는 것을 막으려면 본문 글자색에 !important 가
필요하다. 그런데 그 규칙은 어두운 배경 위에 흰 글씨를 쓰는 요소(배너,
네이비 버튼, 사이드바)까지 덮어쓴다. 따라서 밝은 글씨를 쓰는 규칙에도
반드시 !important 를 붙이고, 본문 규칙보다 뒤에 배치한다.
"""

from __future__ import annotations

import csv
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
# 경로
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "interim" / "product_catalog.json"
REFUSAL_EVAL_PATH = ROOT / "data" / "eval" / "refusal_eval.csv"

#: 답변 생성 버튼의 위젯 key. CSS 에서 .st-key-<key> 로 잡아 색을 바꾼다.
SUBMIT_KEY = "fg_submit"

# ---------------------------------------------------------------------------
# 테마 — 한국외국어대학교 상징색
#   Primary   딥 네이비    R0  G45  B86   -> #002D56
#   Secondary 브론즈 골드  R141 G113 B80  -> #8D7150
# ---------------------------------------------------------------------------

PRIMARY = "#002D56"
PRIMARY_SOFT = "#084874"
PRIMARY_DEEP = "#001E3A"
SECONDARY = "#8D7150"
SECONDARY_LIFT = "#A88A63"
SECONDARY_SOFT = "#BFA77C"
INK = "#1B2733"
MUTED = "#5B6B7A"
LINE = "#E2E8EE"
CANVAS = "#F7F9FB"
ON_NAVY = "#E7EEF5"
ON_NAVY_MUTED = "rgba(231,238,245,0.72)"

STATUS_STYLE = {
    "answered": {
        "label": "답변 가능", "detail": "공개 문서 근거 확인됨", "icon": "✅",
        "fg": "#0F5132", "bg": "#E8F5EC", "border": "#B6DCC4",
    },
    "partial": {
        "label": "일부 항목 미확인",
        "detail": "문서에서 확인되지 않은 내용이 있습니다", "icon": "⚠",
        "fg": "#7A5B00", "bg": "#FFF6E0", "border": "#EBD9A6",
    },
    "refused": {
        "label": "답변 보류",
        "detail": "근거가 불충분하거나 실시간·개인정보 조회가 필요한 질문입니다",
        "icon": "⛔", "fg": "#7F1D1D", "bg": "#FDECEC", "border": "#F0C3C3",
    },
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

#: "『문서명』 (2026-07-28 수집)" 처럼 끝에 붙는 수집일 표기.
_RE_COLLECTED = re.compile(r"\s*[(（]\s*\d{4}[-.]\d{1,2}[-.]\d{1,2}\s*수집\s*[)）]\s*$")


# ---------------------------------------------------------------------------
# 환경
# ---------------------------------------------------------------------------

def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    return ""


def load_env() -> None:
    """.env 를 직접 파싱한다. 값은 화면에 절대 표시하지 않는다."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in read_text(path).splitlines():
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
    value = obj.get(key, default) if isinstance(obj, dict) \
        else getattr(obj, key, default)
    return default if value is None else value


def to_result_dict(result: Any) -> dict:
    """파이프라인 산출물을 공통 dict 로 변환한다."""
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

    if hasattr(result, "__dict__"):
        return {k: v for k, v in vars(result).items() if not k.startswith("_")}
    return {}


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


def strip_collected_suffix(text: str) -> str:
    """문서명 끝에 붙은 '(YYYY-MM-DD 수집)' 표기를 떼어낸다.

    수집일은 문서의 시행일이 아니라 크롤링 시점이다. 직원이 고객 안내
    근거로 삼을 정보가 아니므로 화면에 노출하지 않는다.
    """
    return _RE_COLLECTED.sub("", (text or "").strip()).strip()


def has_collected_date(*texts: str) -> bool:
    """수집일 표기가 붙어 있었는가 = 시행일이 없는 문서인가."""
    return any(_RE_COLLECTED.search((t or "").strip()) for t in texts)


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
    """answered / partial / refused 를 결정한다."""
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


def evidence_title(ev: dict) -> str:
    raw = evidence_field(ev, "doc_display_name", "citation",
                         default="(문서명 없음)")
    return strip_collected_suffix(raw) or "(문서명 없음)"


def esc(text: Any) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# 테마
# ---------------------------------------------------------------------------

def apply_custom_theme() -> None:
    st.markdown(f"""
    <style>
      .stApp {{ background: {CANVAS} !important; }}
      .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}

      /* =============================================================
         [1] 본문 글자색 고정
         배경을 밝은색으로 강제하므로 글자색도 고정한다. 이 블록이
         없으면 OS 다크모드에서 흰 배경 위 흰 글씨가 된다.
         !important 가 필요하지만, 그 대가로 어두운 배경 위 흰 글씨
         요소까지 덮어쓴다. 그래서 [2] 블록에서 되돌린다.
         ============================================================= */
      .block-container h1, .block-container h2,
      .block-container h3, .block-container h4,
      .block-container p, .block-container li,
      .block-container label, .block-container label p,
      .block-container label span,
      .block-container [data-testid="stMarkdownContainer"] p,
      .block-container [data-testid="stWidgetLabel"] p {{
        color: {INK} !important;
      }}
      .block-container [data-testid="stCaptionContainer"],
      .block-container [data-testid="stCaptionContainer"] p {{
        color: {MUTED} !important;
      }}
      .stTabs [data-baseweb="tab"] p {{ color: {MUTED} !important; }}
      .stTabs [aria-selected="true"] p {{ color: {PRIMARY} !important; }}
      [data-testid="stExpanderDetails"] p,
      [data-testid="stExpanderDetails"] li {{ color: {INK} !important; }}
      [data-testid="stJson"] {{
        background: #FFFFFF !important; border: 1px solid {LINE};
        border-radius: 8px;
      }}
      [data-testid="stCode"] pre, [data-testid="stCode"] code {{
        background: #F1F5F8 !important; color: {INK} !important;
      }}

      /* 알림 상자는 배경이 반투명이라 다크모드에서 특히 취약하다.
         배경·테두리·글자색을 함께 못박는다. */
      [data-testid="stAlert"] {{
        border-radius: 9px !important; border: 1px solid {LINE} !important;
        background: #FFFFFF !important;
      }}
      [data-testid="stAlert"] p, [data-testid="stAlert"] div,
      [data-testid="stAlert"] span {{ color: {INK} !important; }}
      [data-testid="stAlert"][class*="warning"],
      div[data-baseweb="notification"][kind="warning"] {{
        background: #FFF6E0 !important; border-color: #EBD9A6 !important;
      }}
      [data-testid="stAlert"][class*="error"] {{
        background: #FDECEC !important; border-color: #F0C3C3 !important;
      }}
      [data-testid="stAlert"][class*="info"] {{
        background: #F2F7FA !important; border-color: #D6E5EE !important;
      }}

      /* 카드·안내 */
      .fg-card {{
        background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px;
        padding: 18px 20px; box-shadow: 0 1px 3px rgba(16,32,48,0.05);
        margin-bottom: 14px;
      }}
      .fg-card-title {{
        font-size: 0.78rem; font-weight: 700; color: {PRIMARY};
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-bottom: 10px; padding-bottom: 8px;
        border-bottom: 1px solid {LINE};
      }}
      .fg-notice {{
        background: #FFFFFF; border-left: 3px solid {SECONDARY};
        border-radius: 8px; padding: 12px 16px; margin-top: 12px;
        color: {MUTED}; font-size: 0.86rem; line-height: 1.6;
        border-top: 1px solid {LINE}; border-right: 1px solid {LINE};
        border-bottom: 1px solid {LINE};
      }}

      /* 상태 배지 */
      .fg-badge {{
        border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
        border: 1px solid; display: flex; align-items: baseline; gap: 10px;
      }}
      .fg-badge .b-icon {{ font-size: 1.05rem; }}
      .fg-badge .b-label {{ font-weight: 700; font-size: 1rem; }}
      .fg-badge .b-detail {{ font-size: 0.86rem; opacity: 0.85; }}
      .fg-subbadge {{
        display: inline-block; border-radius: 999px; padding: 5px 13px;
        font-size: 0.8rem; font-weight: 600; border: 1px solid {SECONDARY_SOFT};
        color: {SECONDARY}; background: #FBF8F1; margin: 0 6px 8px 0;
      }}

      /* 근거 안내·답변 */
      .fg-source {{
        background: #F2F7FA; border: 1px solid #D6E5EE;
        border-left: 3px solid {PRIMARY}; border-radius: 8px;
        padding: 13px 16px; margin-bottom: 12px; color: {INK};
        font-size: 0.9rem; line-height: 1.65;
      }}
      .fg-source b {{ color: {PRIMARY}; }}
      .fg-answer {{
        background: #FFFFFF; border: 1px solid {LINE}; border-radius: 10px;
        padding: 18px 20px; font-size: 0.97rem; line-height: 1.8; color: {INK};
      }}
      .fg-ev-meta {{
        color: {MUTED}; font-size: 0.8rem; line-height: 1.75;
        padding-bottom: 8px; margin-bottom: 10px;
        border-bottom: 1px dashed {LINE};
      }}
      .fg-ev-meta code {{
        background: #EEF3F7; color: {PRIMARY}; padding: 1px 6px;
        border-radius: 4px; font-size: 0.76rem;
      }}
      .fg-ev-text {{
        background: {CANVAS}; border: 1px solid {LINE}; border-radius: 8px;
        padding: 15px 17px; font-size: 0.87rem; line-height: 1.95;
        color: #2C3A47; white-space: pre-wrap; word-break: break-word;
        max-height: 460px; overflow-y: auto;
      }}
      .fg-hint {{ color: {MUTED}; font-size: 0.82rem; line-height: 1.65; }}

      /* 버튼 기본 (밝은 배경) */
      .stButton > button {{
        border-radius: 8px; border: 1px solid {LINE}; background: #FFFFFF;
        font-size: 0.85rem; font-weight: 500; padding: 9px 14px;
        transition: all 0.12s ease; text-align: left; line-height: 1.45;
      }}
      .stButton > button p {{ color: {INK} !important; }}
      .stButton > button:hover {{
        border-color: {PRIMARY}; background: #F4F9FC;
        box-shadow: 0 2px 6px rgba(0,45,86,0.10);
      }}
      .stButton > button:hover p {{ color: {PRIMARY} !important; }}

      /* 입력 */
      .stTextArea textarea {{
        border-radius: 9px; border: 1px solid {LINE}; font-size: 0.94rem;
        line-height: 1.6; background: #FFFFFF !important; color: {INK} !important;
      }}
      .stTextArea textarea:focus {{
        border-color: {PRIMARY}; box-shadow: 0 0 0 2px rgba(0,45,86,0.10);
      }}

      /* =============================================================
         expander(근거 헤더) / st.status

         테마 추종을 시도했으나 폐기했다. 이유를 남긴다.
           - Streamlit 메뉴(Settings > Theme)의 토글은 OS 설정을 바꾸지
             않으므로 @media (prefers-color-scheme) 로 감지되지 않는다.
           - var(--text-color) 는 이 앱에서 값을 읽지 못했다.
           - st.context.theme 은 1.60 기준 type 값이 None 이라 쓸 수 없다.
         이 앱은 배경·사이드바·버튼·카드를 모두 고정색으로 쓰는 라이트
         기준 화면이다. 이 두 곳만 테마를 따라가면 오히려 어긋나므로
         전체와 같은 밝은 카드로 고정한다. 다크모드에서도 동일하게 보인다.

         선택자에 data-testid 를 쓰지 않는 이유: Streamlit 은
         data-testid="stExpander" 를 <details> 가 아니라 그 바깥 <div> 에
         붙이므로 details[data-testid=...] 는 아무것도 매칭하지 못한다.
         ============================================================= */
      .stApp details {{
        border: 1px solid {LINE} !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
        margin-bottom: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(16,32,48,0.05);
      }}
      .stApp details > summary {{
        background: #FFFFFF !important;
        border-radius: 0 !important;
        padding: 12px 15px !important;
      }}
      .stApp details > summary:hover {{ background: #F4F9FC !important; }}
      .stApp details > summary,
      .stApp details > summary *,
      .stApp details > summary p,
      .stApp details > summary span,
      .stApp details > summary div,
      .stApp details > summary label,
      .stApp details > summary [data-testid="stMarkdownContainer"] p,
      .stApp details > summary [data-testid="stMarkdownContainer"] span {{
        color: {PRIMARY} !important;
        font-weight: 600;
        font-size: 0.92rem;
      }}
      /* 펼침 화살표, 스피너, 완료 체크 아이콘 */
      .stApp details > summary svg,
      .stApp details > summary svg *,
      .stApp details > summary i {{
        fill: {PRIMARY} !important;
        stroke: {PRIMARY} !important;
        color: {PRIMARY} !important;
        opacity: 1 !important;
      }}
      .stApp details > div,
      .stApp details > div > div {{ background: #FFFFFF !important; }}
      .stApp details > div p,
      .stApp details > div li,
      .stApp details > div span,
      .stApp details > div [data-testid="stMarkdownContainer"] p,
      .stApp details > div [data-testid="stMarkdownContainer"] li {{
        color: {INK} !important;
      }}

      /* status 위젯이 details 가 아닌 버전을 위한 보조 규칙 */
      [data-testid="stStatusWidget"],
      [data-testid="stStatus"] {{
        background: #FFFFFF !important;
        border: 1px solid {LINE} !important;
        border-radius: 10px !important;
      }}
      [data-testid="stStatusWidget"] *,
      [data-testid="stStatus"] * {{ color: {INK} !important; }}
      [data-testid="stStatusWidget"] svg,
      [data-testid="stStatus"] svg {{
        fill: {PRIMARY} !important; stroke: {PRIMARY} !important;
      }}

      hr {{ border-color: {LINE}; }}

      /* =============================================================
         [2] 어두운 배경 위 밝은 글씨 — [1] 을 되돌리는 구간.
         반드시 [1] 보다 뒤에 오고 !important 를 붙여야 한다.
         ============================================================= */

      /* 상단 배너 */
      .fg-banner {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_SOFT} 100%);
        border-radius: 14px; padding: 26px 30px 22px 30px;
        box-shadow: 0 6px 20px rgba(0,45,86,0.18);
        border-top: 3px solid {SECONDARY_SOFT};
      }}
      .fg-banner h1 {{
        color: #FFFFFF !important; margin: 0 0 4px 0; font-size: 1.85rem;
        font-weight: 700; letter-spacing: -0.02em;
      }}
      .fg-banner .sub {{
        color: {SECONDARY_SOFT} !important; font-size: 0.98rem;
        font-weight: 600; margin-bottom: 10px;
      }}
      /* 선택자를 길게 쓰는 이유:
         [1] 블록의 `.block-container [data-testid="stMarkdownContainer"] p`
         가 클래스 2 + 요소 1 의 명시도를 갖는다. !important 끼리 부딪히면
         명시도가 높은 쪽이 이기므로, 배너 규칙도 그보다 높여야 한다. */
      .block-container [data-testid="stMarkdownContainer"] .fg-banner p,
      .block-container [data-testid="stMarkdownContainer"] .fg-banner .desc,
      .block-container .fg-banner p,
      .fg-banner .desc, .fg-banner p {{
        color: #FFFFFF !important; font-size: 0.92rem; line-height: 1.6;
        margin: 0; opacity: 0.94;
      }}

      /* 선택된 카테고리 칩 — 네이비 바탕 + 흰 글씨 */
      .stButton > button[kind="primary"] {{
        background: {PRIMARY} !important; border: none !important;
        font-weight: 600; text-align: center;
        box-shadow: 0 3px 10px rgba(0,45,86,0.24);
      }}
      .stButton > button[kind="primary"]:hover {{
        background: {PRIMARY_SOFT} !important;
      }}
      .stButton > button[kind="primary"] p,
      .stButton > button[kind="primary"] span,
      .stButton > button[kind="primary"] div,
      .stButton > button[kind="primary"]:hover p {{
        color: #FFFFFF !important;
      }}

      /* 답변 생성 버튼 — 브론즈 골드 */
      .st-key-{SUBMIT_KEY} button,
      .st-key-{SUBMIT_KEY} button[kind="primary"] {{
        background: linear-gradient(135deg, {SECONDARY} 0%, {SECONDARY_LIFT} 100%)
          !important;
        border: none !important; font-weight: 700 !important;
        text-align: center; letter-spacing: 0.02em;
        box-shadow: 0 3px 12px rgba(141,113,80,0.32) !important;
      }}
      .st-key-{SUBMIT_KEY} button:hover,
      .st-key-{SUBMIT_KEY} button[kind="primary"]:hover {{
        background: {SECONDARY_LIFT} !important;
        box-shadow: 0 4px 14px rgba(141,113,80,0.42) !important;
      }}
      .st-key-{SUBMIT_KEY} button p,
      .st-key-{SUBMIT_KEY} button span,
      .st-key-{SUBMIT_KEY} button div,
      .st-key-{SUBMIT_KEY} button:hover p {{ color: #FFFFFF !important; }}

      /* 사이드바 — 딥 네이비 바탕, 브론즈 골드 포인트 */
      section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PRIMARY} 0%, {PRIMARY_DEEP} 100%)
          !important;
        border-right: none;
      }}
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] span,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] label p,
      section[data-testid="stSidebar"] label span,
      section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {ON_NAVY} !important;
      }}
      section[data-testid="stSidebar"] .fg-card-title {{
        color: {SECONDARY_SOFT} !important;
        border-bottom: 1px solid rgba(191,167,124,0.32);
      }}
      section[data-testid="stSidebar"] .fg-metric {{
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.13);
        border-left: 3px solid {SECONDARY_SOFT};
        border-radius: 8px; padding: 10px 13px; margin-bottom: 8px;
      }}
      section[data-testid="stSidebar"] .fg-metric .m-label,
      section[data-testid="stSidebar"] .fg-metric .m-note {{
        color: {ON_NAVY_MUTED} !important;
      }}
      section[data-testid="stSidebar"] .fg-metric .m-label {{
        font-size: 0.74rem; font-weight: 600;
      }}
      section[data-testid="stSidebar"] .fg-metric .m-value {{
        font-size: 1.3rem; font-weight: 700; color: #FFFFFF !important;
        line-height: 1.25;
      }}
      section[data-testid="stSidebar"] .fg-metric .m-note {{ font-size: 0.7rem; }}
      section[data-testid="stSidebar"] .fg-hint {{
        color: {ON_NAVY_MUTED} !important;
      }}
      section[data-testid="stSidebar"] [data-testid="stCheckbox"] svg {{
        fill: {SECONDARY_SOFT};
      }}
      section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.16);
      }}
    </style>
    """, unsafe_allow_html=True)


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
        if not os.environ.get("OPENAI_API_KEY"):
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
        return matcher, [e.canonical_name for e in catalog.all_named]
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
        folder = ROOT / "reports"
        if not folder.is_dir():
            return None
        for path in sorted(folder.glob(pattern),
                           key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                return json.loads(read_text(path))
            except Exception:
                continue
        return None

    out: list[tuple[str, Any, str]] = []

    hybrid = newest("hybrid_tuning_final_doclevel_*.json")
    if isinstance(hybrid, dict):
        chosen = None
        for key, val in (hybrid.get("results") or {}).items():
            if "weighted" in str(key) and "0.5" in str(key) and "z" not in str(key):
                chosen = val
                break
        overall = (chosen or {}).get("overall", {}) if isinstance(chosen, dict) else {}
        if overall.get("recall@5") is not None:
            out.append(("문서 Recall@5", overall["recall@5"],
                        f"검색 · n={overall.get('n', '?')}"))
        if overall.get("mrr") is not None:
            out.append(("MRR", overall["mrr"], "검색 순위 품질"))

    refusal = newest("refusal_eval_*.json")
    if refusal is not None:
        far = deep_find(refusal, ("false_answer_rate", "far"))
        bal = deep_find(refusal, ("balanced_accuracy", "balacc"))
        if far is not None:
            out.append(("False Answer Rate (FNR)", far,
                        "거절해야 할 질문에 답한 비율"))
        if bal is not None:
            out.append(("Balanced Accuracy", bal, "거절 판정 · 이진 분류"))

    return {"items": out}


@st.cache_resource(show_spinner=False)
def build_example_question_map() -> dict[str, list[str]]:
    """카테고리별 예시 질문.

    질문을 새로 지어내지 않는다. 아래 목록은 실제 평가셋에 포함되어
    파이프라인이 처리한 이력이 있는 질문들이다. 거절 시연 카테고리는
    거절 평가셋 CSV 에서 직접 읽어온다.
    """
    catalog: dict[str, list[str]] = {
        "예금": [
            "정기예금을 중도해지하면 이자는 어떻게 계산되나요?",
            "자유저축예금 이자는 언제 받을 수 있나요?",
            "하나은행 예금 토큰은 모바일 앱에서 어떻게 가입해요?",
            "저축예금 통장에 사기 의심이 생기면 거래가 어떻게 제한돼요?",
            "하나은행 입출금 예금 이율은 바뀌면 언제부터 적용돼요?",
        ],
        "적금": [
            "자유적금 만기 전에 해지하면 우대금리는 어떻게 되나요?",
            "사업자 주거래 우대통장 가입 후 최초 다음달까지 제공되는 수수료 우대서비스는 무엇인가요?",
        ],
        "청약": [
            "주택청약종합저축 가입하고 청약 1순위 되려면 어느 정도 기간과 납입횟수가 필요한가요?",
            "청년 주택드림 청약통장 가입할 때 소득 증명서류로는 어떤 게 필요한가요?",
            "주택청약저축에서 선납한 금액은 언제 인정회차로 산정되나요?",
        ],
        "대출": [
            "은행에서 빌린 돈을 10일 넘게 못 갚으면 어떻게 되나요?",
            "대출받은 뒤에 원금은 안 내고 이자만 낼 수 있는 기간이 있나요?",
            "전세자금대출에서 중도상환해약금 산정 기준은 무엇인가요?",
            "하나은행 마이너스통장 대출 이자는 어떤 기준으로 출금되나요?",
            "하나은행 가계 대출할 때 인지세는 누가 부담해요?",
        ],
        "외환 / 해외송금": [
            "해외 ATM 출금 서비스의 1회 출금 표준한도는 얼마인가요?",
            "해외에서 돈 받을 때 해외 사람이 우리 은행에 보내려면 뭘 알려줘야 해?",
            "외화 송금한 후에 돈 보내는 걸 취소하거나 바꿀 수 있나요?",
            "하나은행 환전지갑에서 외화 찾을 때 신분증 꼭 가져가야 해요? 대리인은 어떻게 되나요?",
        ],
        "전자금융 / 인터넷뱅킹": [
            "하나은행 인터넷뱅킹에서 1천만원 넘게 이체하려면 어떻게 해야 하나요?",
            "하나원큐에서 특정 계좌 조회 못하게 하려면 어떻게 해?",
            "하나은행 심플이체에서 이체 내역을 수정할 수 있나요?",
            "내 계좌에 돈이 부족하면 인터넷으로 돈 보내기가 안 되나요?",
        ],
        "마이데이터 / 인증": [
            "휴대폰 번호는 안 바꾸고 기기만 바꿨는데 금융인증서 클라우드 계속 쓸 수 있나요?",
            "하나 합(마이데이터 서비스) 이용 시 만 14세 미만 손님의 거래 제한 내용은 무엇인가요?",
            "OTP 이용 중 '보정거래 필요' 오류가 발생하면 어떻게 해야 하나요?",
            "마이데이터서비스 약관이 갑자기 바뀌면 은행이 어떻게 알려줘요?",
        ],
        "퇴직연금 / 펀드": [
            "개인형IRP에서 만기 자금 운용 지시가 없을 때 어떻게 운용되나요?",
            "소득공제 장기펀드를 5년 안에 중간에 해지하면 세금이 얼마나 나와요?",
            "하나은행 디폴트옵션 안정투자형 포트폴리오 1 상품이 위험등급이 바뀌거나 승인이 취소되면 어떻게 알려주나요?",
            "일임형 개인종합자산관리계좌(ISA) 계약 만료 후 수수료는 어떻게 처리됩니까?",
        ],
        "공통 약관 / 기타": [
            "은행이 돈을 다 갚으라고 통보를 늦게 하면 저는 언제부터 바로 갚아야 하나요?",
            "통장 없이 거래할 때 도장이나 서명은 꼭 제출해야 하나요?",
            "하나은행에서 전자금융 거래 내역을 못 받을 때 은행이 어떻게 알려줘야 해?",
            "은행이 서류를 잃어버리거나 다치게 하면 제가 가진 다른 증서로 빚을 갚아야 하나요?",
        ],
    }

    # 거절 시연은 거절 평가셋에서 직접 읽는다. 지어낸 질문은 실제로
    # 거절되지 않을 수 있고, 그러면 시연이 성립하지 않는다.
    refuse_questions: list[str] = []
    if REFUSAL_EVAL_PATH.exists():
        text = read_text(REFUSAL_EVAL_PATH)
        try:
            rows = list(csv.DictReader(text.splitlines())) if text else []
        except csv.Error:
            rows = []
        picked: dict[str, str] = {}
        for row in rows:
            if str(row.get("expected", "")).strip().lower() != "refuse":
                continue
            kind = str(row.get("refusal_type", "")).strip()
            question = str(row.get("question", "")).strip()
            if kind and question and kind not in picked:
                picked[kind] = question
        refuse_questions = list(picked.values())

    if refuse_questions:
        catalog["거절 시연"] = refuse_questions

    return catalog


# ---------------------------------------------------------------------------
# 상품 표시 (판정하지 않는다)
# ---------------------------------------------------------------------------

def question_products(matcher, names: list[str], question: str) -> list[str]:
    """질문에 언급된 카탈로그 상품을 찾는다(보수적, 표시 전용)."""
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
        if informative and all(prefix_hit(t) for t in informative):
            found.append(name)
    return found


def evidence_products(matcher, evidences: list[dict]) -> list[str]:
    """근거 문서에서 상품명을 뽑아 중복을 제거한다."""
    if matcher is None or not evidences:
        return []
    seen: list[str] = []
    for ev in evidences:
        name = strip_collected_suffix(
            evidence_field(ev, "doc_display_name", "citation"))
        if not name:
            continue
        try:
            product = (matcher.product_of(name) or "").strip()
        except Exception:
            product = name
        if product and product not in seen:
            seen.append(product)
    return seen


# ---------------------------------------------------------------------------
# 렌더링 — 공통
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(f"""
    <div class="fg-banner">
      <h1>FinGuide-RAG</h1>
      <div class="sub">은행 직원용 근거 제시형 금융 RAG 콘솔</div>
      <p class="desc">
        상품설명서·약관·FAQ 원문 근거를 함께 제공하여 직원이 직접 검증할 수
        있도록 설계된 B2E RAG 데모입니다.
      </p>
    </div>
    <div class="fg-notice">
      ※ 본 데모는 공개 문서 기반 프로토타입이며, 실제 고객 안내 전에는
      은행 내부 시스템과 담당 부서 확인이 필요합니다.
    </div>
    """, unsafe_allow_html=True)


def render_metrics_sidebar() -> tuple[bool, bool]:
    with st.sidebar:
        st.markdown('<div class="fg-card-title">측정된 성능</div>',
                    unsafe_allow_html=True)

        items = load_metrics().get("items", [])
        if items:
            for label, value, note in items:
                shown = f"{value:.3f}" if isinstance(value, float) else str(value)
                st.markdown(f"""
                <div class="fg-metric">
                  <div class="m-label">{esc(label)}</div>
                  <div class="m-value">{esc(shown)}</div>
                  <div class="m-note">{esc(note)}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown(
                '<div class="fg-hint">reports/ 의 최신 평가 결과에서 '
                '읽어옵니다.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="fg-hint">reports/ 에서 평가 결과를 찾지 '
                '못했습니다.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="fg-card-title">실행 설정</div>',
                    unsafe_allow_html=True)
        offline = st.checkbox(
            "오프라인 모드 (LLM 없이 검색·규칙만)", value=False,
            help="LLM 근거 검증과 답변 생성을 끕니다. 검색 결과와 규칙 기반 "
                 "거절만 확인할 때 씁니다.")
        debug = st.checkbox("엔지니어링 메타데이터 보기", value=False)

        st.markdown("---")
        st.markdown(
            '<div class="fg-hint">공개 문서 308건 · 청크 2,999개<br>'
            'multilingual-e5-small + BM25 하이브리드</div>',
            unsafe_allow_html=True)
        return offline, debug


# ---------------------------------------------------------------------------
# 렌더링 — 질문 입력
# ---------------------------------------------------------------------------

def render_category_selector(categories: list[str]) -> None:
    """카테고리 칩. 클릭 즉시 선택 색상이 반영되도록 곧바로 rerun 한다.

    st.button 은 렌더링과 반환이 한 번에 일어난다. 따라서 클릭이 발생한
    실행 회차에서는 이미 옛 상태로 그려진 뒤이므로, rerun 하지 않으면
    다음 상호작용까지 색이 바뀌지 않는다(두 번 눌러야 하는 것처럼 보임).
    """
    st.markdown('<div class="fg-card-title">카테고리 선택</div>',
                unsafe_allow_html=True)

    per_row = 3
    for start in range(0, len(categories), per_row):
        cols = st.columns(per_row)
        for col, name in zip(cols, categories[start:start + per_row]):
            with col:
                selected = st.session_state.get("selected_category") == name
                if st.button(
                    ("● " if selected else "") + name,
                    key=f"cat_{name}", use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    st.session_state["selected_category"] = (
                        None if selected else name)
                    st.rerun()


def render_example_questions_by_category(mapping: dict[str, list[str]]) -> None:
    category = st.session_state.get("selected_category")
    if not category:
        st.markdown(
            '<div class="fg-hint">카테고리를 선택하면 해당 분야의 예시 질문이 '
            '표시됩니다. 직접 입력해도 됩니다.</div>', unsafe_allow_html=True)
        return

    if category == "거절 시연":
        st.markdown(
            '<div class="fg-hint">거절 평가셋에서 유형별로 한 건씩 가져온 '
            '질문입니다. 답변 대신 보류 사유가 표시되는 것이 정상 동작입니다.'
            '</div>', unsafe_allow_html=True)

    for i, question in enumerate(mapping.get(category, [])):
        if st.button(question, key=f"q_{category}_{i}",
                     use_container_width=True):
            st.session_state["question_input"] = question
            st.rerun()


def render_question_input() -> bool:
    st.markdown('<div class="fg-card-title">고객 문의 입력</div>',
                unsafe_allow_html=True)
    st.text_area("질문", key="question_input", height=110,
                 placeholder="고객 문의 내용을 입력하세요",
                 label_visibility="collapsed")
    return st.button("답변 생성", key=SUBMIT_KEY, type="primary",
                     use_container_width=True)


# ---------------------------------------------------------------------------
# 렌더링 — 답변
# ---------------------------------------------------------------------------

def render_status_badge(result: dict, evidences: list[dict]) -> None:
    style = STATUS_STYLE[derive_status(result)]
    st.markdown(f"""
    <div class="fg-badge" style="background:{style['bg']};
         border-color:{style['border']}; color:{style['fg']};">
      <span class="b-icon">{style['icon']}</span>
      <span>
        <span class="b-label" style="color:{style['fg']};">{style['label']}</span>
        <span class="b-detail" style="color:{style['fg']};">
          — {style['detail']}</span>
      </span>
    </div>""", unsafe_allow_html=True)

    subs: list[str] = []
    doc_types = {evidence_field(e, "doc_type") for e in evidences}
    doc_types.discard("")
    if doc_types and doc_types <= {"약관", "FAQ"}:
        subs.append("📘 공통 규정 근거 — 특정 상품이 아닌 공통 약관·FAQ 기준")
    if any(e.get("is_latest") is False for e in evidences):
        subs.append("🕓 최신본이 아닐 수 있는 근거 포함 — 시행일 확인 필요")
    if subs:
        st.markdown(
            "".join(f'<span class="fg-subbadge">{esc(s)}</span>' for s in subs),
            unsafe_allow_html=True)


def render_source_notice(result: dict, evidences: list[dict],
                         question: str) -> str:
    """답변 최상단 근거 안내문을 렌더링하고, 복사용 텍스트를 돌려준다."""
    notice = str(safe_get(result, "source_notice", "") or "").strip()

    if not notice:
        titles: list[str] = []
        for ev in evidences:
            name = evidence_title(ev)
            if name and name != "(문서명 없음)" and name not in titles:
                titles.append(name)
        notice = (
            "※ 이 답변은 다음 근거 문서에 기반합니다: "
            + ", ".join(f"『{t}』" for t in titles) + "."
            if titles else "※ 표시할 근거 문서가 없습니다."
        )

    matcher, names = get_product_tools()
    q_products = question_products(matcher, names, question)
    e_products = evidence_products(matcher, evidences)

    rows = ""
    if q_products or e_products:
        left = ", ".join(f"『{p}』" for p in q_products) or "특정 상품 미인식"
        right = ", ".join(f"『{p}』" for p in e_products) or "없음"
        rows = (
            f'<div style="margin-top:9px; font-size:0.86rem; line-height:1.75;">'
            f'질문에서 인식된 상품 <span style="opacity:.7;">(카탈로그 대조, '
            f'참고용)</span> · {esc(left)}<br>'
            f'근거 문서의 상품 · {esc(right)}</div>'
        )

    st.markdown(f'<div class="fg-source"><b>{esc(notice)}</b>{rows}</div>',
                unsafe_allow_html=True)

    if rows:
        st.markdown(
            '<div class="fg-hint">두 값이 다르면 우측 근거 원문을 확인하세요. '
            '자동 상품 판정은 오탐률이 높은 것으로 측정되어(20건 중 16건), '
            '시스템이 불일치를 단정하지 않고 두 값을 그대로 제시합니다.</div>',
            unsafe_allow_html=True)

    return notice


def render_answer_panel(result: dict, evidences: list[dict],
                        question: str) -> None:
    render_status_badge(result, evidences)
    notice = render_source_notice(result, evidences, question)

    answer_text = str(safe_get(result, "answer", "") or "")
    st.markdown(
        f'<div class="fg-answer">{esc(answer_text) or "(답변 본문 없음)"}</div>',
        unsafe_allow_html=True)

    reason = safe_get(result, "refusal_reason")
    stage = str(safe_get(result, "stage", "") or "")
    meta: list[str] = []
    if reason:
        meta.append(f"거절 사유 · <code>{esc(reason)}</code> "
                    f"{esc(REFUSAL_LABEL.get(str(reason), ''))}")
    if stage:
        meta.append(f"판정 단계 · {esc(STAGE_LABEL.get(stage, stage))}")
    if meta:
        st.markdown(f'<div class="fg-ev-meta" style="margin-top:12px;">'
                    f'{"<br>".join(meta)}</div>', unsafe_allow_html=True)

    not_found = safe_get(result, "not_found") or []
    if not_found:
        items = "".join(f"<li>{esc(x)}</li>" for x in not_found)
        st.markdown(f"""
        <div class="fg-card" style="border-left:3px solid {SECONDARY};
             margin-top:12px;">
          <div class="fg-card-title" style="color:{SECONDARY};">
            문서에서 확인되지 않은 항목</div>
          <ul style="margin:0 0 8px 18px; font-size:0.9rem; line-height:1.75;">
            {items}</ul>
          <div class="fg-hint">위 항목은 근거 문서에 없어 답변에 포함하지
          않았습니다. 내부 시스템이나 담당 부서 확인이 필요합니다.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    checked = st.checkbox("우측 근거 원문을 확인했습니다.", key="evidence_checked")
    if not checked:
        st.warning("고객 안내 전 우측 근거 문서를 확인하는 것을 권장합니다.")
        return

    copy_text = f"{notice}\n\n{answer_text}"
    if not_found:
        copy_text += "\n\n[문서에서 확인되지 않은 항목]\n" + "\n".join(
            f"- {x}" for x in not_found)
    st.markdown('<div class="fg-card-title">고객 안내용</div>',
                unsafe_allow_html=True)
    st.text_area("복사해서 사용하세요", value=copy_text, height=210,
                 label_visibility="collapsed")


# ---------------------------------------------------------------------------
# 렌더링 — 근거
# ---------------------------------------------------------------------------

def render_evidence_panel(evidences: list[dict], result: dict | None) -> None:
    st.markdown('<div class="fg-card-title">근거 원문</div>',
                unsafe_allow_html=True)

    if result is None:
        st.markdown(
            '<div class="fg-hint">답변을 생성하면 참조한 근거 문서의 원문이 '
            '여기에 표시됩니다. 직원은 이 원문을 직접 확인한 뒤 고객에게 '
            '안내합니다.</div>', unsafe_allow_html=True)
        return

    if not evidences:
        if str(safe_get(result, "stage", "") or "") == "pattern":
            st.info(
                "이 질문은 검색 이전 단계(질문 패턴)에서 판정되었습니다.\n\n"
                "검색과 LLM 을 모두 호출하지 않았으므로 표시할 근거가 없습니다. "
                "의도된 동작입니다.")
        else:
            st.warning("표시할 근거가 없습니다.")
        return

    for i, ev in enumerate(evidences):
        rank = ev.get("rank", i + 1)
        with st.expander(f"근거 {rank}  ·  {evidence_title(ev)}", expanded=True):
            _render_evidence_body(ev)


def _render_evidence_body(ev: dict) -> None:
    meta = [f"문서유형 · {esc(evidence_field(ev, 'doc_type', default='N/A'))}"]

    # 시행일: 값이 있고, 그 문서가 '수집일'로 표기되는 문서가 아닐 때만
    # 보여준다. 수집일은 크롤링 시점이지 문서의 효력 발생일이 아니므로
    # 고객 안내 근거로 오인될 수 있다.
    effective = evidence_field(ev, "effective_date", "valid_from", "date",
                               "시행일")
    collected = has_collected_date(ev.get("citation", ""),
                                   ev.get("doc_display_name", ""))
    if effective and not collected:
        meta.append(f"시행일 · {esc(effective)}")

    meta.append(f"score · {esc(format_score(ev.get('score')))}")

    section = evidence_field(ev, "section")
    if section:
        meta.append(f"위치 · {esc(section)}")

    ids = f"<code>{esc(evidence_field(ev, 'chunk_id', default='N/A'))}</code>"
    doc_id = evidence_field(ev, "doc_id")
    if doc_id:
        ids = f"<code>{esc(doc_id)}</code> / " + ids

    st.markdown(
        f'<div class="fg-ev-meta">{" &nbsp;·&nbsp; ".join(meta)}<br>{ids}</div>',
        unsafe_allow_html=True)

    if ev.get("is_latest") is False:
        st.warning("최신본이 아닐 수 있습니다. 시행일을 확인하세요.")

    body = truncate_text(evidence_field(ev, "text"), 3000) or "(원문 없음)"
    st.markdown(f'<div class="fg-ev-text">{esc(body)}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 렌더링 — 개발자 모드
# ---------------------------------------------------------------------------

def render_debug_panel(result: dict, evidences: list[dict]) -> None:
    st.markdown("---")
    st.markdown('<div class="fg-card-title">엔지니어링 메타데이터</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["검색 점수", "거절 판정", "처리 시간", "raw result"])

    with tab1:
        if evidences:
            st.dataframe([{
                "rank": e.get("rank"),
                "doc_title": evidence_title(e),
                "hybrid_score": format_score(e.get("score")),
                "doc_type": evidence_field(e, "doc_type", default="N/A"),
                "effective_date": evidence_field(
                    e, "effective_date", "valid_from", "date", default="N/A"),
                "is_latest": e.get("is_latest", "N/A"),
            } for e in evidences], use_container_width=True, hide_index=True)
            st.caption(
                "hybrid_score 는 weighted(min-max) α=0.5 로 결합된 최종 "
                "점수입니다. dense·sparse 개별 점수는 현재 파이프라인이 "
                "반환하지 않습니다.")
        else:
            st.caption("근거가 없어 표시할 점수가 없습니다.")

    with tab2:
        stage = str(safe_get(result, "stage", "") or "N/A")
        st.markdown(
            f"- **최종 상태**: `{derive_status(result)}`\n"
            f"- **decision**: `{safe_get(result, 'decision', 'N/A')}`\n"
            f"- **판정 단계**: `{stage}` — {STAGE_LABEL.get(stage, 'N/A')}\n"
            f"- **거절 사유**: `{safe_get(result, 'refusal_reason') or 'N/A'}`")
        signals = safe_get(result, "signals") or {}
        if signals:
            st.markdown("**거절 판정에 쓰인 신호**")
            st.dataframe(
                [{"신호": k, "값": v}
                 for k, v in sanitize_for_debug(signals).items()],
                use_container_width=True, hide_index=True)
        else:
            st.caption("신호 값이 반환되지 않았습니다.")

    with tab3:
        st.markdown(
            f"- **total_ms**: {safe_get(result, 'latency_ms', 'N/A')}\n"
            f"- **tokens**: {safe_get(result, 'tokens', 'N/A')}")
        st.caption("단계별 소요 시간(retrieval/refusal/generation)은 현재 "
                   "파이프라인이 계측하지 않습니다.")

    with tab4:
        st.json(sanitize_for_debug(result))
        st.caption("근거 원문은 800자로 잘랐습니다. 전체는 우측 근거 패널에서 "
                   "확인하세요.")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def run_pipeline(question: str, offline: bool) -> None:
    """파이프라인을 호출하고 결과를 session_state 에 저장한다."""
    # 새 질문마다 근거 확인 체크를 반드시 해제한다. 이전 답변에 대한
    # 확인이 새 답변으로 승계되면 이 화면의 목적 자체가 무너진다.
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
            + "".join(traceback.format_exc()[-1500:]))


def main() -> None:
    st.set_page_config(page_title="FinGuide-RAG", layout="wide",
                       initial_sidebar_state="expanded")
    apply_custom_theme()

    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("selected_category", None)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("asked_question", "")
    st.session_state.setdefault("error", "")

    render_header()
    offline, debug = render_metrics_sidebar()
    st.markdown("<br>", unsafe_allow_html=True)

    examples = build_example_question_map()
    left, right = st.columns([0.45, 0.55], gap="large")

    with left:
        render_category_selector(list(examples.keys()))
        render_example_questions_by_category(examples)
        st.markdown("---")

        if render_question_input():
            question = st.session_state["question_input"].strip()
            if not question:
                st.warning("질문을 입력하세요.")
            else:
                run_pipeline(question, offline)

        if st.session_state["error"]:
            st.error("파이프라인 실행에 실패했습니다.")
            with st.expander("오류 상세"):
                st.code(st.session_state["error"])
            st.caption("인덱스가 없거나 API 키가 설정되지 않았을 수 있습니다. "
                       "사이드바의 오프라인 모드로 검색만 확인할 수 있습니다.")

        result = st.session_state["result"]
        if result:
            st.markdown("---")
            render_answer_panel(result, evidence_list(result),
                                st.session_state["asked_question"])

    with right:
        result = st.session_state["result"]
        render_evidence_panel(evidence_list(result) if result else [], result)

    if debug and st.session_state["result"]:
        render_debug_panel(st.session_state["result"],
                           evidence_list(st.session_state["result"]))


if __name__ == "__main__":
    main()
