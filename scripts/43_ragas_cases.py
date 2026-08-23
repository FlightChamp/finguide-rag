# -*- coding: utf-8 -*-
"""
40_ragas_cases.py — RAGAS 결과 중 사람 라벨과 어긋난 케이스의 본문 추출

목적
----
39번 측정에서 드러난 불일치 케이스(표준 지표 미탐 / 표준 지표 오탐)의
질문·답변·근거·진술 단위 판정을 한자리에 모아 README에 실을 사례를 만든다.

API 호출을 하지 않는다. 39번이 남긴 판정 캐시를 그대로 읽는다. 비용 0.

선정 기준 (--ids 미지정 시 자동)
------------------------------
- 미탐: 사람이 fail 판정했으나 Faithfulness >= 임계값 → 표준 지표가 통과시킨 오류
- 오탐: 사람이 pass 판정했으나 Faithfulness <  임계값 → 표준 지표만 문제 삼은 건

사용법
-----
    python scripts/40_ragas_cases.py                # 자동 선정
    python scripts/40_ragas_cases.py --ids 24,03    # 특정 건만
    python scripts/40_ragas_cases.py --ctx-chars 1200

출력
----
- reports/ragas_cases_<날짜>.md
- 콘솔에도 동일 내용 출력

의존성: 표준 라이브러리만 사용
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

MEASURE_SCRIPT = "42_ragas_measure.py"
DEFAULT_CTX_CHARS = 800
DEFAULT_ANSWER_CHARS = 1500

#: 사람 라벨 컬럼과 표시명
LABEL_COLUMNS = [
    ("human_환각", "사람_환각"),
    ("human_상품일치", "사람_상품일치"),
    ("human_수치정확", "사람_수치정확"),
    ("v2_환각", "v2_환각"),
    ("v2_상품일치", "v2_상품일치"),
    ("v2_수치정확", "v2_수치정확"),
]


def load_measure_module(scripts_dir: Path):
    """39번 스크립트를 모듈로 불러온다(캐시 키 계산을 동일하게 맞추기 위해)."""
    path = scripts_dir / MEASURE_SCRIPT
    if not path.exists():
        sys.exit(f"[중단] {MEASURE_SCRIPT} 를 찾지 못했습니다: {path}")
    spec = importlib.util.spec_from_file_location("ragas_measure", path)
    if spec is None or spec.loader is None:
        sys.exit(f"[중단] {MEASURE_SCRIPT} 를 불러오지 못했습니다.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… (이하 {len(text) - limit:,}자 생략)"


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 불일치 케이스 본문 추출")
    parser.add_argument("--ids", default=None,
                        help="쉼표로 구분한 id 목록 (미지정 시 자동 선정)")
    parser.add_argument("--input", default="data/interim/g_eval_answers.json")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Faithfulness 통과 임계값 (기본: 39번 설정값)")
    parser.add_argument("--ctx-chars", type=int, default=DEFAULT_CTX_CHARS,
                        help=f"근거 표시 최대 글자 수 (기본 {DEFAULT_CTX_CHARS})")
    parser.add_argument("--model", default=None, help="판정 모델 (기본: 39번 설정값)")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    rm = load_measure_module(scripts_dir)

    root = rm.find_project_root()
    model = args.model or rm.CHAT_MODEL
    threshold = args.threshold if args.threshold is not None \
        else rm.FAITHFUL_PASS_THRESHOLD

    print(f"[1/4] 루트 {root} / 모델 {model} / 임계값 {threshold}")

    # 1) 샘플과 라벨
    source = root / args.input
    samples = rm.load_samples(source)
    rm.join_labels(samples, root)
    by_id = {rm.norm_id(s["id"]): s for s in samples}

    # 2) 39번 측정 결과
    measure_path = rm.newest(root, "reports/ragas_measure_*.csv")
    if measure_path is None:
        sys.exit("[중단] reports/ragas_measure_*.csv 가 없습니다. 39번을 먼저 실행하세요.")
    rows = rm.read_csv_rows(measure_path)
    scores = {rm.row_id(r): r for r in rows if rm.row_id(r)}
    print(f"[2/4] 측정 결과 {len(scores)}건 로드: {measure_path.name}")

    # 3) 캐시
    cache_path = (root / "data" / "interim"
                  / f"ragas_cache_{model.replace('.', '')}.json")
    cache = rm.JudgeCache(cache_path)
    if not cache.data:
        print(f"[경고] 판정 캐시가 비어 있습니다: {cache_path}")
        print("       진술 단위 판정 없이 질문·답변·근거만 출력합니다.")

    # 4) 대상 선정
    selected: list[tuple[str, str]] = []  # (id, 분류)
    if args.ids:
        for raw in args.ids.split(","):
            key = rm.norm_id(raw)
            if key in by_id:
                selected.append((key, "직접 지정"))
            else:
                print(f"[경고] id {raw} 를 찾지 못했습니다.")
    else:
        for key, row in scores.items():
            faith = to_float(row.get("ragas_faithfulness"))
            if faith is None:
                continue
            sample = by_id.get(key)
            if sample is None:
                continue
            human = [sample.get(col, "") for col, _ in LABEL_COLUMNS
                     if col.startswith("human_")]
            has_fail = "fail" in human
            has_pass_only = ("fail" not in human) and ("pass" in human)
            if has_fail and faith >= threshold:
                selected.append((key, "미탐 — 사람 fail, 표준 지표 통과"))
            elif has_pass_only and faith < threshold:
                selected.append((key, "오탐 — 사람 pass, 표준 지표 탈락"))

    print(f"[3/4] 대상 {len(selected)}건 선정")
    if not selected:
        sys.exit("[중단] 조건에 맞는 케이스가 없습니다. --ids 로 직접 지정해 보세요.")

    # 5) 리포트 작성
    out: list[str] = []
    add = out.append
    add(f"# RAGAS 불일치 케이스 — {datetime.now():%Y-%m-%d %H:%M}\n")
    add(f"- 측정 결과: `{measure_path.name}` / 임계값 {threshold}")
    add(f"- 판정 캐시: `{cache_path.name}` (API 호출 없음)")
    add(f"- 대상 {len(selected)}건\n")

    add("| id | 분류 | Faithfulness | AnswerRel | 사람_환각 | 사람_상품일치 | 사람_수치정확 |")
    add("|---|---|---|---|---|---|---|")
    for key, kind in selected:
        row = scores.get(key, {})
        sample = by_id[key]
        add(f"| {sample['id']} | {kind} | "
            f"{row.get('ragas_faithfulness', '-')} | "
            f"{row.get('ragas_answer_relevancy', '-')} | "
            f"{sample.get('human_환각', '-') or '-'} | "
            f"{sample.get('human_상품일치', '-') or '-'} | "
            f"{sample.get('human_수치정확', '-') or '-'} |")

    for key, kind in selected:
        sample = by_id[key]
        row = scores.get(key, {})
        add(f"\n---\n\n## id {sample['id']} — {kind}\n")
        add(f"- 문서유형 {sample.get('doc_type') or '-'} / "
            f"난이도 {sample.get('difficulty') or '-'} / "
            f"판정단계 {sample.get('stage') or '-'}")
        add(f"- Faithfulness {row.get('ragas_faithfulness', '-')} "
            f"({row.get('supported', '?')}/{row.get('judged', '?')}) / "
            f"AnswerRel {row.get('ragas_answer_relevancy', '-')}")
        labels = " / ".join(
            f"{shown}={sample.get(col) or '-'}" for col, shown in LABEL_COLUMNS)
        add(f"- 라벨: {labels}")

        add(f"\n**질문**\n\n> {sample['question']}\n")
        add("**답변**\n")
        add("```")
        add(truncate(sample["answer"], DEFAULT_ANSWER_CHARS))
        add("```")

        contexts = sample.get("contexts", [])
        add(f"\n**근거 {len(contexts)}건**\n")
        for i, ctx in enumerate(contexts, start=1):
            add(f"근거 {i}:")
            add("```")
            add(truncate(ctx, args.ctx_chars))
            add("```")

        # 진술 단위 판정 (캐시에서 복원)
        stmt_key = rm.cache_key("stmt", model, sample["question"], sample["answer"])
        stmt_result = cache.data.get(stmt_key) or {}
        statements = [s for s in stmt_result.get("statements", [])
                      if isinstance(s, str)]
        if statements:
            ctx_block = "\n\n".join(
                f"[근거 {i}]\n{c}" for i, c in enumerate(contexts, start=1))
            stmt_block = "\n".join(
                f"{i}. {s}" for i, s in enumerate(statements, start=1))
            verify_key = rm.cache_key("verify", model, ctx_block, stmt_block)
            verdicts = (cache.data.get(verify_key) or {}).get("verdicts", [])
            verdict_by_idx = {}
            for i, v in enumerate(verdicts, start=1):
                if isinstance(v, dict):
                    verdict_by_idx[int(v.get("idx", i) or i)] = v

            add("\n**진술 단위 판정**\n")
            add("| # | 진술 | 지지 | 판정 근거 |")
            add("|---|---|---|---|")
            for i, statement in enumerate(statements, start=1):
                v = verdict_by_idx.get(i, {})
                supported = v.get("supported")
                mark = "O" if supported is True else ("X" if supported is False else "?")
                reason = str(v.get("reason", "")).replace("|", "/")
                add(f"| {i} | {statement.replace('|', '/')} | {mark} | {reason} |")
        else:
            add("\n(진술 판정을 캐시에서 찾지 못했습니다 — "
                "39번 실행 이후 답변이 바뀌었을 수 있습니다)")

        # 역질문
        qgen_key = rm.cache_key("qgen", model, sample["answer"])
        qgen = cache.data.get(qgen_key) or {}
        generated = [q for q in qgen.get("questions", []) if isinstance(q, str)]
        if generated:
            add(f"\n**역생성 질문** (noncommittal={qgen.get('noncommittal')})\n")
            for q in generated:
                add(f"- {q}")

    content = "\n".join(out)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out_path = reports / f"ragas_cases_{datetime.now():%Y-%m-%d}.md"
    out_path.write_text(content, encoding="utf-8")

    print("[4/4] 완료\n")
    print(content)
    print()
    print("=" * 60)
    print(f"저장: {out_path}")
    print(f"분량: {len(content):,}자")
    print("=" * 60)


if __name__ == "__main__":
    main()
