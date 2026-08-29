# -*- coding: utf-8 -*-
"""
45_check_examples.py — 데모 예시 질문 전량 점검

목적
----
app.py 의 카테고리별 예시 질문을 실제 파이프라인에 통과시켜, 시연에서
의도대로 동작하는지 확인한다. 특히 다음 두 가지를 본다.

  1. 일반 카테고리 질문이 답변되는가 (거절되면 시연이 밋밋해진다)
  2. 거절 시연 질문이 실제로 거절되는가 (답해버리면 시연이 무너진다)

질문 목록은 app.py 소스를 AST 로 파싱해 가져온다. 목록을 두 곳에
복사해 두면 한쪽만 고쳤을 때 어긋나기 때문이다. app.py 를 import 하지
않는 이유는 streamlit 위젯이 실행되기 때문이다.

사용법
-----
    python scripts/45_check_examples.py --dry-run   # 대상과 예상 비용만
    python scripts/45_check_examples.py --yes
    python scripts/45_check_examples.py --only "거절 시연"

출력
----
    reports/example_check_<날짜>.md  + 콘솔 요약

의존성: 표준 라이브러리 + 프로젝트 파이프라인
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_MARKERS = ("pyproject.toml", "src", "scripts", "reports")

#: 이 카테고리의 질문은 거절되어야 정상이다.
REFUSAL_CATEGORY = "거절 시연"


def find_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if sum(1 for m in ROOT_MARKERS if (cand / m).exists()) >= 2:
            return cand
    return Path.cwd().resolve()


def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    return ""


def load_env(root: Path) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    for line in read_text(root / ".env").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def extract_examples(app_path: Path) -> dict[str, list[str]]:
    """app.py 에서 예시 질문 딕셔너리를 꺼낸다(실행하지 않는다)."""
    src = read_text(app_path)
    if not src:
        sys.exit(f"[중단] app.py 를 읽지 못했습니다: {app_path}")

    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef)
                and node.name == "build_example_question_map"):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Dict) and sub.keys:
                try:
                    return ast.literal_eval(sub)
                except ValueError:
                    continue
    sys.exit("[중단] 예시 질문 목록을 찾지 못했습니다.")


def describe(answer: Any) -> dict[str, Any]:
    decision = getattr(answer, "decision", None)
    decision = getattr(decision, "value", decision)
    not_found = list(getattr(answer, "not_found", []) or [])
    evidences = list(getattr(answer, "evidences", []) or [])

    if str(decision).lower() in ("refuse", "refused"):
        status = "refused"
    elif not_found:
        status = "partial"
    else:
        status = "answered"

    top = evidences[0] if evidences else None
    return {
        "status": status,
        "stage": getattr(answer, "stage", "") or "",
        "reason": getattr(answer, "refusal_reason", "") or "",
        "n_evidence": len(evidences),
        "top_doc": (getattr(top, "doc_display_name", "")
                    or getattr(top, "citation", "")) if top else "",
        "top_score": round(float(getattr(top, "score", 0.0)), 4) if top else None,
        "not_found": not_found,
        "answer": (getattr(answer, "answer", "") or "").strip(),
        "latency_ms": getattr(answer, "latency_ms", 0),
    }


def verdict(category: str, info: dict) -> tuple[str, str]:
    """시연 관점의 판정. (기호, 사유)"""
    if category == REFUSAL_CATEGORY:
        if info["status"] == "refused":
            return "OK", f"거절됨 ({info['reason'] or '사유 미기재'})"
        return "문제", "거절되어야 하는데 답변함"
    if info["status"] == "refused":
        return "문제", f"답변되어야 하는데 거절됨 ({info['reason']})"
    if info["status"] == "partial":
        return "주의", f"부분 답변 (미확인 {len(info['not_found'])}건)"
    if info["n_evidence"] == 0:
        return "주의", "근거 없이 답변"
    return "OK", f"근거 {info['n_evidence']}건"


def main() -> None:
    parser = argparse.ArgumentParser(description="예시 질문 전량 점검")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--only", default=None, help="특정 카테고리만 검사")
    args = parser.parse_args()

    root = find_root()
    load_env(root)
    examples = extract_examples(root / "app.py")

    if args.only:
        examples = {k: v for k, v in examples.items() if k == args.only}
        if not examples:
            sys.exit(f"[중단] 카테고리를 찾지 못했습니다: {args.only}")

    total = sum(len(v) for v in examples.values())
    print(f"대상: {len(examples)}개 카테고리 / {total}개 질문")
    print("예상 LLM 호출: 질문당 최대 2회 (거절 검증 + 답변 생성)")

    if args.dry_run:
        for cat, qs in examples.items():
            print(f"  {cat} · {len(qs)}문항")
        print("\n[dry-run] 실제 호출 없이 종료합니다.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[중단] OPENAI_API_KEY 를 찾지 못했습니다 (.env 확인).")

    if not args.yes:
        if input("진행할까요? [y/N] ").strip().lower() not in ("y", "yes"):
            print("취소했습니다.")
            return

    src_dir = root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from openai import OpenAI
    from finguide_rag.generation.pipeline import Pipeline

    print("파이프라인 준비 중…")
    pipeline = Pipeline.build(root, client=OpenAI(), use_llm=True)

    rows: list[tuple[str, str, dict, str, str]] = []
    started = time.perf_counter()
    done = 0
    for category, questions in examples.items():
        for question in questions:
            done += 1
            print(f"  ({done}/{total}) {category} · {question[:34]}…", flush=True)
            try:
                info = describe(pipeline.answer(question))
                mark, why = verdict(category, info)
            except Exception as exc:
                info = {"status": "error", "stage": "", "reason": "",
                        "n_evidence": 0, "top_doc": "", "top_score": None,
                        "not_found": [], "answer": "", "latency_ms": 0}
                mark, why = "오류", f"{type(exc).__name__}: {exc}"[:120]
            rows.append((category, question, info, mark, why))

    elapsed = time.perf_counter() - started

    # 리포트
    out: list[str] = [f"# 예시 질문 점검 — {datetime.now():%Y-%m-%d %H:%M}\n"]
    counts: dict[str, int] = {}
    for *_, mark, _ in rows:
        counts[mark] = counts.get(mark, 0) + 1
    out.append(f"- 질문 {len(rows)}개 / 소요 {elapsed:.0f}초")
    out.append("- 판정: " + " · ".join(f"{k} {v}건" for k, v in counts.items()))

    out.append("\n## 문제·주의 항목\n")
    flagged = [r for r in rows if r[3] != "OK"]
    if flagged:
        out.append("| 판정 | 카테고리 | 질문 | 사유 |")
        out.append("|---|---|---|---|")
        for cat, q, _, mark, why in flagged:
            out.append(f"| **{mark}** | {cat} | {q} | {why} |")
    else:
        out.append("전 항목 정상입니다.")

    out.append("\n## 전체 결과\n")
    out.append("| 판정 | 카테고리 | 질문 | status | stage | 사유 | 근거 | top1 문서 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for cat, q, info, mark, _ in rows:
        out.append(
            f"| {mark} | {cat} | {q[:44]} | {info['status']} | "
            f"{info['stage'] or '-'} | {info['reason'] or '-'} | "
            f"{info['n_evidence']} | {str(info['top_doc'])[:36] or '-'} |")

    content = "\n".join(out)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / f"example_check_{datetime.now():%Y-%m-%d}.md"
    path.write_text(content, encoding="utf-8")

    print()
    print(content)
    print()
    print("=" * 60)
    print(f"저장: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
