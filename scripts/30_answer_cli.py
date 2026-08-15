"""
30_answer_cli.py

목적
----
검색 -> 거절 판정 -> 답변 생성 전체 흐름을 직접 확인한다.

거절과 답변이 하나의 진입점(Pipeline.answer)으로 통합되어 있으므로,
어떤 질문을 넣어도 같은 타입(Answer)이 돌아온다. 거절되는 질문과
답변되는 질문을 번갈아 넣어 보면 흐름이 한눈에 들어온다.

사용법
-----
    python scripts/30_answer_cli.py
        대화형. 질문을 계속 입력한다. 빈 줄이나 q 로 종료.

    python scripts/30_answer_cli.py -q "정기예금 중도해지하면 이자는 어떻게 되나요?"
        한 건만 처리하고 종료.

    python scripts/30_answer_cli.py --demo
        미리 준비된 질문 5건을 순서대로 처리한다.
        거절 유형별로 하나씩 섞여 있어 동작 확인에 좋다.

    python scripts/30_answer_cli.py --no-llm
        LLM 없이 검색과 규칙 단계만. 비용 0.

    python scripts/30_answer_cli.py -q "..." --json
        Answer 를 JSON 으로 출력. 평가 스크립트 연동 확인용.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.pipeline import Pipeline  # noqa: E402

# 거절 유형과 정상 답변이 고루 섞이도록 골랐다.
DEMO_QUESTIONS = [
    ("정기예금을 중도해지하면 이자는 어떻게 계산되나요?", "답변 기대"),
    ("제 신용등급이면 얼마까지 대출받을 수 있나요?", "거절: personalized"),
    ("오늘 정기예금 금리가 몇 퍼센트인가요?", "거절: time_variant"),
    ("다른 은행이랑 비교하면 어떤 점이 더 좋나요?", "거절: out_of_scope"),
    ("예금거래에서 이자는 언제 지급되나요?", "답변 기대"),
]


def show(ans, as_json: bool) -> None:
    if as_json:
        print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))
        return

    print(ans.format_display())
    print(f"\n  ── {ans.decision.value} / stage={ans.stage} / "
          f"{ans.tokens} tokens / {ans.latency_ms}ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model", default="e5-small")
    ap.add_argument("--index", default=None)
    ap.add_argument("--bm25", default="default")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--llm-model", default="gpt-4.1-mini")
    ap.add_argument("--no-llm", action="store_true",
                    help="LLM 검증·생성 없이 검색과 규칙만 (비용 0)")
    args = ap.parse_args()

    client = None
    if not args.no_llm:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI()

    print("=" * 76)
    print("FinGuide-RAG — 근거 기반 답변")
    print("=" * 76)
    print(f"  검색 weighted α={args.alpha} / 생성 "
          f"{'사용 안 함' if args.no_llm else args.llm_model}")
    print("  인덱스 로딩 중...")

    pipeline = Pipeline.build(
        PROJECT_ROOT,
        model=args.model,
        index=args.index,
        bm25=args.bm25,
        alpha=args.alpha,
        client=client,
        llm_model=args.llm_model,
        use_llm=not args.no_llm,
    )
    print(f"  준비 완료 (청크 {len(pipeline.chunk_texts)}개)\n")

    # --- 단건 ---
    if args.question:
        show(pipeline.answer(args.question), args.json)
        return

    # --- 데모 ---
    if args.demo:
        for i, (q, expect) in enumerate(DEMO_QUESTIONS, 1):
            print("=" * 76)
            print(f"[{i}/{len(DEMO_QUESTIONS)}] {q}")
            print(f"  기대: {expect}")
            print("-" * 76)
            show(pipeline.answer(q), args.json)
            print()
        return

    # --- 대화형 ---
    print("질문을 입력하세요. 빈 줄 또는 q 로 종료합니다.\n")
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"q", "quit", "exit"}:
            break
        print("-" * 76)
        show(pipeline.answer(q), args.json)
        print()


if __name__ == "__main__":
    main()
