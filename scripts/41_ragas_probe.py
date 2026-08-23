# -*- coding: utf-8 -*-
"""
38_ragas_probe.py — RAGAS 병행 측정을 위한 사전 탐지

목적
----
RAGAS 지표(faithfulness, answer relevancy)를 계산하려면 샘플마다
  (1) 질문  (2) 검색된 근거 청크 목록  (3) 생성된 답변
세 가지가 필요하다. 이 스크립트는 프로젝트 안에서 그 세 필드를 담고 있는
파일을 찾아내고, 어떤 키 이름으로 저장돼 있는지만 보고한다.

측정은 하지 않는다. 값도 거의 싣지 않는다(80자 이상은 잘라냄).
목적은 "다음 스크립트를 어떤 스키마에 맞춰 짜야 하는가"를 확정하는 것.

수집 항목
--------
1. 실행 환경 (Python 버전, ragas/langchain/datasets/openai 설치 여부와 버전)
2. reports/ · data/ 안의 JSON / JSONL / CSV 중 평가 산출물로 보이는 파일
3. 각 후보의 레코드 수, 키 목록, 필드 역할 추정, 값 샘플(잘림)
4. RAGAS 입력 적합도 판정 (질문/근거/답변 3필드가 모두 있는가)

사용법
-----
    python scripts/38_ragas_probe.py

    # 탐색 범위 추가
    python scripts/38_ragas_probe.py --dirs reports data outputs

의존성: 표준 라이브러리만 사용 (ragas 미설치 상태에서도 동작)
출력  : 콘솔 + reports/ragas_probe_<YYYY-MM-DD>.md
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

#: 기본 탐색 디렉터리 (프로젝트 루트 기준)
DEFAULT_DIRS = ["reports", "data", "outputs", "eval", "evaluation"]

#: 설치 여부를 확인할 패키지
CHECK_PACKAGES = [
    "ragas", "langchain", "langchain-core", "langchain-openai",
    "datasets", "openai", "pandas", "numpy",
    "sentence-transformers", "faiss-cpu", "pydantic",
]

#: 필드 역할 추정용 키워드 (소문자 부분일치)
ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "question": ("question", "query", "질문", "q_text", "user_input", "prompt"),
    "contexts": ("context", "chunk", "retrieved", "passage", "evidence",
                 "근거", "docs", "documents", "source"),
    "answer":   ("answer", "response", "generated", "output", "prediction",
                 "답변", "generation"),
    "ground_truth": ("ground_truth", "gold", "reference", "정답", "label"),
    "refusal":  ("refus", "abstain", "거절", "reject", "is_refused"),
    "verdict":  ("hallucin", "환각", "mismatch", "불일치", "numeric",
                 "수치", "judge", "verdict", "score"),
}

#: 값 미리보기 최대 길이
PREVIEW_CHARS = 80

#: 파일당 검사할 최대 레코드 수
SCAN_LIMIT = 200

ROOT_MARKERS = ("pyproject.toml", "src", "scripts", "reports")


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if sum(1 for m in ROOT_MARKERS if (cand / m).exists()) >= 2:
            return cand
    return Path.cwd().resolve()


def read_text(path: Path, limit: int | None = None) -> str | None:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            text = path.read_text(encoding=enc)
            return text[:limit] if limit else text
        except (UnicodeDecodeError, OSError):
            continue
    return None


def preview(value: object) -> str:
    """값을 짧게 요약한다. 긴 본문은 싣지 않는다."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        inner = preview(value[0]) if value else ""
        return f"list[{len(value)}] 첫원소: {inner}"
    if isinstance(value, dict):
        return f"dict(keys={list(value)[:6]})"
    text = " ".join(str(value).split())
    if len(text) > PREVIEW_CHARS:
        return text[:PREVIEW_CHARS] + " …"
    return text


def guess_role(key: str) -> str | None:
    low = key.lower()
    for role, hints in ROLE_HINTS.items():
        if any(h in low for h in hints):
            return role
    return None


def size_label(path: Path) -> str:
    n = path.stat().st_size
    if n < 1024:
        return f"{n}B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    return f"{n / 1024 ** 2:.1f}MB"


# ---------------------------------------------------------------------------
# 환경 점검
# ---------------------------------------------------------------------------

def probe_environment() -> list[str]:
    lines = [f"- Python: {sys.version.split()[0]}",
             f"- 실행 파일: {sys.executable}"]
    for name in CHECK_PACKAGES:
        try:
            lines.append(f"- {name}: {md.version(name)}")
        except md.PackageNotFoundError:
            lines.append(f"- {name}: (미설치)")
    return lines


# ---------------------------------------------------------------------------
# 파일 파싱
# ---------------------------------------------------------------------------

def load_records(path: Path) -> tuple[list[dict], str] | None:
    """파일에서 레코드 리스트를 뽑는다. (records, 형식설명) 또는 None."""
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        text = read_text(path)
        if text is None:
            return None
        records = []
        for line in text.splitlines()[:SCAN_LIMIT]:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
        total = sum(1 for ln in text.splitlines() if ln.strip())
        return (records, f"JSONL, 총 {total}줄") if records else None

    if suffix == ".json":
        text = read_text(path)
        if text is None:
            return None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, list):
            recs = [r for r in obj[:SCAN_LIMIT] if isinstance(r, dict)]
            return (recs, f"JSON 배열, 총 {len(obj)}건") if recs else None
        if isinstance(obj, dict):
            # 최상위 dict 안에서 레코드 배열을 찾는다
            for key, value in obj.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    recs = [r for r in value[:SCAN_LIMIT] if isinstance(r, dict)]
                    return recs, f"JSON dict → '{key}' 배열, 총 {len(value)}건"
            return [obj], "JSON 단일 dict (요약 파일로 추정)"
        return None

    if suffix == ".csv":
        text = read_text(path)
        if text is None:
            return None
        try:
            rows = list(csv.DictReader(text.splitlines()))
        except csv.Error:
            return None
        if not rows:
            return None
        return rows[:SCAN_LIMIT], f"CSV, 총 {len(rows)}행"

    return None


def analyse(records: list[dict]) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """키별 역할 추정과 샘플 값을 뽑는다."""
    first = records[0]
    roles: dict[str, list[str]] = {}
    samples: list[tuple[str, str]] = []
    for key, value in first.items():
        role = guess_role(str(key))
        if role:
            roles.setdefault(role, []).append(str(key))
        samples.append((str(key), preview(value)))
    return roles, samples


def ragas_readiness(roles: dict[str, list[str]]) -> tuple[str, str]:
    """RAGAS 입력으로 쓸 수 있는지 판정."""
    need = ["question", "contexts", "answer"]
    missing = [n for n in need if n not in roles]
    if not missing:
        return "적합", "질문·근거·답변 3필드 모두 확인"
    if len(missing) == 3:
        return "부적합", "3필드 모두 없음"
    return "부분", f"누락: {', '.join(missing)}"


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 사전 탐지")
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                        help="탐색할 디렉터리 (프로젝트 루트 기준)")
    parser.add_argument("--max-files", type=int, default=40,
                        help="보고할 최대 파일 수 (기본 40)")
    args = parser.parse_args()

    root = find_project_root()
    print(f"[1/3] 프로젝트 루트: {root}")

    out: list[str] = []
    add = out.append

    add(f"# RAGAS 사전 탐지 — {datetime.now():%Y-%m-%d %H:%M}\n")
    add(f"- 루트: `{root}`\n")

    # 1) 환경
    print("[2/3] 실행 환경 점검")
    add("\n## 1. 실행 환경\n")
    out.extend(probe_environment())

    # 2) 후보 파일
    print("[3/3] 평가 산출물 스캔")
    add("\n## 2. 평가 산출물 후보\n")

    candidates: list[Path] = []
    for name in args.dirs:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in (".json", ".jsonl", ".csv"):
                candidates.append(path)

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        add("(탐색 대상 디렉터리에서 json/jsonl/csv 파일을 찾지 못했습니다)")
        add(f"\n탐색한 경로: {', '.join(args.dirs)}")
    else:
        add(f"총 {len(candidates)}개 발견, 최신 {min(len(candidates), args.max_files)}개 분석\n")

    fit_table: list[tuple[str, str, str, str]] = []

    for path in candidates[:args.max_files]:
        rel = path.relative_to(root).as_posix()
        loaded = load_records(path)
        if loaded is None:
            add(f"\n### `{rel}`\n")
            add(f"- 크기 {size_label(path)} / 파싱 불가 또는 레코드 없음")
            continue

        records, fmt = loaded
        roles, samples = analyse(records)
        status, reason = ragas_readiness(roles)
        fit_table.append((rel, status, reason, fmt))

        add(f"\n### `{rel}`\n")
        add(f"- 크기 {size_label(path)} / {fmt} / "
            f"수정 {datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M}")
        add(f"- **RAGAS 적합도: {status}** — {reason}")
        if roles:
            add("- 역할 추정:")
            for role, keys in roles.items():
                add(f"  - {role}: {', '.join(f'`{k}`' for k in keys)}")
        add(f"- 키 {len(samples)}개와 첫 레코드 값(잘림):")
        add("")
        add("| 키 | 값 미리보기 |")
        add("|---|---|")
        for key, val in samples:
            add(f"| `{key}` | {val.replace('|', '/')} |")

    # 3) 요약
    add("\n## 3. 요약 — RAGAS 입력 후보\n")
    if fit_table:
        add("| 파일 | 적합도 | 사유 | 형식 |")
        add("|---|---|---|---|")
        for rel, status, reason, fmt in sorted(
                fit_table, key=lambda r: {"적합": 0, "부분": 1, "부적합": 2}[r[1]]):
            add(f"| `{rel}` | {status} | {reason} | {fmt} |")
    else:
        add("(분석 가능한 후보 없음)")

    content = "\n".join(out)

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out_path = reports / f"ragas_probe_{datetime.now():%Y-%m-%d}.md"
    out_path.write_text(content, encoding="utf-8")

    # 콘솔에도 전문 출력 (붙여넣기 편하도록)
    print()
    print(content)
    print()
    print("=" * 60)
    print(f"저장: {out_path}")
    print(f"분량: {len(content):,}자")
    print("=" * 60)


if __name__ == "__main__":
    main()
