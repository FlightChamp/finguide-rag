# -*- coding: utf-8 -*-
"""
37_readme_snapshot.py — README 전면 개편용 저장소 스냅샷 수집기

목적
----
README를 다시 쓰기 위해 필요한 "사실 재료"를 한 파일로 모은다.
추측이나 기억이 아니라 실제 파일 시스템 상태를 근거로 문서를 쓰기 위한 사전 단계.

수집 항목
--------
1. 프로젝트 디렉터리 트리 (대용량/캐시 디렉터리 제외)
2. 현재 README.md 전문
3. pyproject.toml / requirements*.txt
4. scripts/ 파일 목록 + 각 파일의 첫 docstring 요약
5. src/finguide_rag/ 모듈 목록 + 각 모듈의 첫 docstring 요약
6. reports/ 산출물 목록 (크기·수정일) + JSON 리포트의 최상위 키 구조
7. data/ 하위 파일 개수 요약 (내용은 읽지 않음)

사용법
-----
    python 37_readme_snapshot.py

프로젝트 루트 자동 탐지에 실패하면 아래처럼 직접 지정:

    python 37_readme_snapshot.py --root "C:\\Programming\\MyProject\\financial_rag_project"

의존성: 표준 라이브러리만 사용 (외부 패키지 불필요)
출력  : <루트>/reports/readme_snapshot_<YYYY-MM-DD>.md
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

#: 트리 탐색에서 통째로 건너뛸 디렉터리 이름
SKIP_DIRS = {
    ".venv", "venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".idea", ".vscode", ".ipynb_checkpoints",
    "site-packages", ".egg-info",
}

#: 내용을 읽지 않고 개수만 세는 디렉터리 (대용량 데이터/인덱스)
COUNT_ONLY_DIRS = {"data", "indexes", "index", "cache", "outputs", "artifacts"}

#: 트리 출력 최대 깊이
MAX_TREE_DEPTH = 4

#: 한 디렉터리에서 트리에 표시할 최대 항목 수
MAX_ENTRIES_PER_DIR = 40

#: docstring 요약 시 가져올 최대 글자 수
DOCSTRING_CHARS = 300

#: 루트 판별에 사용할 마커
ROOT_MARKERS = ("pyproject.toml", "src/finguide_rag", "src", "scripts")


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def read_text(path: Path, limit: int | None = None) -> str:
    """인코딩 문제에 강한 텍스트 읽기. 실패 시 사유 문자열을 반환한다."""
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            text = path.read_text(encoding=enc)
            if limit is not None and len(text) > limit:
                cut = len(text) - limit
                text = text[:limit] + f"\n\n... (이하 {cut:,}자 생략)"
            return text
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            return f"(읽기 실패: {exc})"
    return "(읽기 실패: 지원하는 인코딩으로 디코딩되지 않음)"


def find_root(explicit: str | None) -> Path:
    """프로젝트 루트를 탐지한다. 실패하면 SystemExit."""
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            sys.exit(f"[중단] 지정한 경로가 디렉터리가 아닙니다: {root}")
        return root

    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents]
    for cand in candidates:
        hits = sum(1 for m in ROOT_MARKERS if (cand / m).exists())
        if hits >= 2:
            return cand
    # 마커를 못 찾으면 현재 작업 디렉터리로 폴백
    cwd = Path.cwd().resolve()
    print(f"[경고] 루트 마커를 찾지 못해 현재 디렉터리를 루트로 간주합니다: {cwd}")
    return cwd


def summarize_docstring(path: Path) -> str:
    """파이썬 파일의 모듈 docstring 첫 부분을 뽑는다."""
    src = read_text(path)
    if src.startswith("(읽기 실패"):
        return src
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return f"(파싱 실패: {exc.msg})"
    doc = ast.get_docstring(tree)
    if not doc:
        return "(모듈 docstring 없음)"
    doc = " ".join(doc.split())
    if len(doc) > DOCSTRING_CHARS:
        doc = doc[:DOCSTRING_CHARS] + " …"
    return doc


def human_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:,.0f}{unit}" if unit == "B" else f"{num/1:,.0f}{unit}"
        num /= 1024
    return f"{num:.0f}GB"


def size_label(path: Path) -> str:
    try:
        n = path.stat().st_size
    except OSError:
        return "?"
    if n < 1024:
        return f"{n}B"
    if n < 1024 ** 2:
        return f"{n/1024:.1f}KB"
    return f"{n/1024**2:.1f}MB"


def mtime_label(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "?"


# ---------------------------------------------------------------------------
# 수집 단계
# ---------------------------------------------------------------------------

def build_tree(root: Path) -> list[str]:
    """디렉터리 트리를 문자열 리스트로 만든다."""
    lines: list[str] = [f"{root.name}/"]

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > MAX_TREE_DEPTH:
            lines.append(prefix + "└── … (깊이 제한)")
            return
        try:
            entries = sorted(
                (p for p in directory.iterdir() if p.name not in SKIP_DIRS
                 and not p.name.endswith(".egg-info")),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except OSError as exc:
            lines.append(prefix + f"└── (접근 실패: {exc})")
            return

        truncated = False
        if len(entries) > MAX_ENTRIES_PER_DIR:
            entries = entries[:MAX_ENTRIES_PER_DIR]
            truncated = True

        for i, entry in enumerate(entries):
            last = (i == len(entries) - 1) and not truncated
            connector = "└── " if last else "├── "
            if entry.is_dir():
                if entry.name in COUNT_ONLY_DIRS:
                    try:
                        n_files = sum(1 for _ in entry.rglob("*") if _.is_file())
                    except OSError:
                        n_files = -1
                    note = f"(파일 {n_files}개, 내용 생략)" if n_files >= 0 else "(집계 실패)"
                    lines.append(prefix + connector + f"{entry.name}/ {note}")
                else:
                    lines.append(prefix + connector + entry.name + "/")
                    walk(entry, prefix + ("    " if last else "│   "), depth + 1)
            else:
                lines.append(prefix + connector + f"{entry.name}  [{size_label(entry)}]")
        if truncated:
            lines.append(prefix + "└── … (항목 수 제한으로 일부 생략)")

    walk(root, "", 1)
    return lines


def collect_py_summaries(directory: Path) -> list[tuple[str, str]]:
    """디렉터리 내 .py 파일의 (이름, docstring 요약) 목록."""
    if not directory.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for path in sorted(directory.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(directory).as_posix()
        out.append((rel, summarize_docstring(path)))
    return out


def json_shape(path: Path, max_keys: int = 25) -> str:
    """JSON 리포트의 구조(최상위 키와 타입)를 요약한다. 값은 크게 싣지 않는다."""
    raw = read_text(path)
    if raw.startswith("(읽기 실패"):
        return raw
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"(JSON 파싱 실패: {exc})"

    def describe(value: object) -> str:
        if isinstance(value, dict):
            return f"dict({len(value)} keys)"
        if isinstance(value, list):
            return f"list({len(value)})"
        if isinstance(value, (int, float, bool)) or value is None:
            return f"{type(value).__name__} = {value!r}"
        text = str(value)
        return f"str = {text[:60]!r}" + (" …" if len(text) > 60 else "")

    if isinstance(obj, dict):
        items = list(obj.items())[:max_keys]
        lines = [f"  - {k}: {describe(v)}" for k, v in items]
        if len(obj) > max_keys:
            lines.append(f"  - … (키 {len(obj) - max_keys}개 더)")
        return "\n".join(lines)
    if isinstance(obj, list):
        head = obj[0] if obj else None
        return f"  - 최상위가 list({len(obj)}), 첫 원소: {describe(head)}"
    return f"  - {describe(obj)}"


def collect_reports(root: Path) -> list[str]:
    reports = root / "reports"
    lines: list[str] = []
    if not reports.is_dir():
        return ["(reports/ 디렉터리 없음)"]

    files = sorted(
        (p for p in reports.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    lines.append(f"총 {len(files)}개 파일 (최신순)\n")
    lines.append("| 파일 | 크기 | 수정일 |")
    lines.append("|---|---|---|")
    for path in files[:60]:
        rel = path.relative_to(reports).as_posix()
        lines.append(f"| {rel} | {size_label(path)} | {mtime_label(path)} |")
    if len(files) > 60:
        lines.append(f"\n… (파일 {len(files) - 60}개 더)")

    json_files = [p for p in files if p.suffix.lower() == ".json"][:8]
    if json_files:
        lines.append("\n### 최신 JSON 리포트 구조 (값이 아니라 키만)\n")
        for path in json_files:
            lines.append(f"**{path.relative_to(reports).as_posix()}**")
            lines.append("```")
            lines.append(json_shape(path))
            lines.append("```")

    csv_files = [p for p in files if p.suffix.lower() == ".csv"][:6]
    if csv_files:
        lines.append("\n### 최신 CSV 리포트 헤더\n")
        for path in csv_files:
            head = read_text(path, limit=2000).splitlines()
            header = head[0] if head else "(빈 파일)"
            n_rows = max(len(head) - 1, 0)
            lines.append(f"- `{path.relative_to(reports).as_posix()}` "
                         f"(표시분 {n_rows}행 이상)\n  - 헤더: `{header}`")
    return lines


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="README 개편용 저장소 스냅샷 수집기")
    parser.add_argument("--root", default=None,
                        help="프로젝트 루트 경로 (미지정 시 자동 탐지)")
    parser.add_argument("--readme-limit", type=int, default=20000,
                        help="README 본문 수집 최대 글자 수 (기본 20000)")
    args = parser.parse_args()

    root = find_root(args.root)
    print(f"[1/6] 프로젝트 루트: {root}")

    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"readme_snapshot_{datetime.now():%Y-%m-%d}.md"

    parts: list[str] = []
    add = parts.append

    add(f"# README 스냅샷 — {datetime.now():%Y-%m-%d %H:%M}\n")
    add(f"- 루트: `{root}`")
    add(f"- 파이썬: {sys.version.split()[0]}")
    add(f"- 생성 스크립트: `37_readme_snapshot.py`\n")

    # 1) 트리
    print("[2/6] 디렉터리 트리 수집")
    add("\n## 1. 디렉터리 트리\n")
    add("```")
    parts.extend(build_tree(root))
    add("```")

    # 2) README
    print("[3/6] README 수집")
    add("\n## 2. 현재 README.md 전문\n")
    readme = root / "README.md"
    if readme.exists():
        add(f"- 크기: {size_label(readme)} / 수정일: {mtime_label(readme)}\n")
        add("````markdown")
        add(read_text(readme, limit=args.readme_limit))
        add("````")
    else:
        add("(README.md 없음)")

    # 3) 설정 파일
    print("[4/6] 설정 파일 수집")
    add("\n## 3. 프로젝트 설정 파일\n")
    found_config = False
    for name in ("pyproject.toml", "setup.cfg", "setup.py",
                 "requirements.txt", "requirements-dev.txt", ".env.example"):
        path = root / name
        if path.exists():
            found_config = True
            add(f"### {name}\n")
            add("```")
            add(read_text(path, limit=4000))
            add("```")
    if not found_config:
        add("(설정 파일을 찾지 못함)")

    # 4) 스크립트 / 모듈
    print("[5/6] 스크립트·모듈 docstring 수집")
    add("\n## 4. scripts/ 목록\n")
    scripts = collect_py_summaries(root / "scripts")
    if scripts:
        add("| 파일 | 모듈 docstring 요약 |")
        add("|---|---|")
        for name, doc in scripts:
            add(f"| `{name}` | {doc.replace('|', '/')} |")
    else:
        add("(scripts/ 디렉터리를 찾지 못함)")

    add("\n## 5. src/finguide_rag/ 모듈\n")
    pkg = root / "src" / "finguide_rag"
    modules = collect_py_summaries(pkg)
    if modules:
        add("| 모듈 | 모듈 docstring 요약 |")
        add("|---|---|")
        for name, doc in modules:
            add(f"| `{name}` | {doc.replace('|', '/')} |")
    else:
        add(f"(`{pkg}` 를 찾지 못함)")

    # 5) reports
    print("[6/6] reports/ 산출물 수집")
    add("\n## 6. reports/ 산출물\n")
    parts.extend(collect_reports(root))

    content = "\n".join(parts)
    out_path.write_text(content, encoding="utf-8")

    print()
    print("=" * 60)
    print(f"완료: {out_path}")
    print(f"분량: {len(content):,}자 / {content.count(chr(10)):,}줄")
    print("=" * 60)
    if len(content) > 120_000:
        print("[안내] 분량이 커서 채팅에 통째로 붙여넣기 어려울 수 있습니다.")
        print("       '1. 디렉터리 트리'와 '2. 현재 README' 섹션만 먼저 붙여넣어도 됩니다.")


if __name__ == "__main__":
    main()
