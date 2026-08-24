# -*- coding: utf-8 -*-
"""
44_patch_answer_signals.py — Answer 에 거절 신호(signals) 필드 추가

목적
----
거절 판정의 근거가 되는 신호(top1 점수, 점수 격차, 공란 비율 등)는
RefusalJudge 가 이미 계산해 RefusalResult.signals 에 담고 있고,
build_answer 도 그중 llm_tokens 를 꺼내 쓰고 있다. 다만 Answer 에는
실려 나오지 않아 데모 UI 에서 볼 수 없다.

이 패치는 계산 로직을 전혀 건드리지 않는다. 이미 있는 값을 Answer 에
그대로 옮겨 담기만 한다.

안전장치
-------
- 치환 대상 문자열이 정확히 1회 나타나는지 먼저 전부 검사한다.
  하나라도 어긋나면 아무것도 바꾸지 않고 중단한다.
- 변경 전 파일을 .bak_<타임스탬프> 로 백업한다.
- 이미 패치된 파일이면 아무 일도 하지 않는다.
- --dry-run 으로 변경 예정 지점만 확인할 수 있다.

사용법
-----
    python scripts/44_patch_answer_signals.py --dry-run
    python scripts/44_patch_answer_signals.py

의존성: 표준 라이브러리만 사용
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET = Path("src") / "finguide_rag" / "generation" / "generator.py"
ROOT_MARKERS = ("pyproject.toml", "src", "scripts", "reports")

#: 이미 패치되었는지 판단하는 표식
SENTINEL = "signals: dict = field(default_factory=dict)"


#: (설명, 찾을 문자열, 바꿀 문자열)
PATCHES: list[tuple[str, str, str]] = [
    (
        "Answer dataclass 에 signals 필드 추가",
        """    stage: str = ""                          # pattern / retrieval / llm
    tokens: int = 0
    latency_ms: int = 0
""",
        """    stage: str = ""                          # pattern / retrieval / llm
    tokens: int = 0
    latency_ms: int = 0
    # 거절 판정에 쓰인 검색 신호(top1 점수, 점수 격차, 공란 비율 등).
    # RefusalJudge 가 계산한 값을 그대로 옮겨 담는다. 판정 로직과는
    # 무관하며, 데모 UI 와 진단 스크립트가 판단 근거를 보기 위한 통로다.
    signals: dict = field(default_factory=dict)
""",
    ),
    (
        "to_dict 에 signals 포함",
        """            "stage": self.stage,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
""",
        """            "stage": self.stage,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "signals": dict(self.signals or {}),
""",
    ),
    (
        "거절 경로 반환에 signals 전달",
        """            tokens=int((refusal.signals or {}).get("llm_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
""",
        """            tokens=int((refusal.signals or {}).get("llm_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
            signals=dict(refusal.signals or {}),
        )
""",
    ),
    (
        "생성기 없음 경로 반환에 signals 전달",
        """            stage=refusal.stage or "no_generator",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
""",
        """            stage=refusal.stage or "no_generator",
            latency_ms=int((time.perf_counter() - started) * 1000),
            signals=dict(refusal.signals or {}),
        )
""",
    ),
    (
        "정상 답변 경로 반환에 signals 전달",
        """        tokens=verify_tokens + gen_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
""",
        """        tokens=verify_tokens + gen_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
        signals=dict(refusal.signals or {}),
    )
""",
    ),
]


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
        except UnicodeDecodeError:
            continue
    sys.exit(f"[중단] 인코딩을 판별하지 못했습니다: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer.signals 패치")
    parser.add_argument("--dry-run", action="store_true",
                        help="변경하지 않고 검사만 수행")
    args = parser.parse_args()

    root = find_root()
    path = root / TARGET
    if not path.exists():
        sys.exit(f"[중단] 대상 파일이 없습니다: {path}")

    print(f"대상: {path}")
    original = read_text(path)

    if SENTINEL in original:
        print("이미 패치된 파일입니다. 변경하지 않고 종료합니다.")
        return

    # 1) 전수 검사 — 하나라도 어긋나면 중단
    problems: list[str] = []
    for label, old, _ in PATCHES:
        count = original.count(old)
        mark = "OK" if count == 1 else "실패"
        print(f"  [{mark}] {label} (일치 {count}회)")
        if count != 1:
            problems.append(f"{label}: {count}회 발견 (1회여야 함)")

    if problems:
        print("\n[중단] 아래 지점을 찾지 못했습니다. 파일을 변경하지 않았습니다.")
        for p in problems:
            print(f"  - {p}")
        print("\n파일이 이미 수정되었거나 줄바꿈/들여쓰기가 다를 수 있습니다.")
        print("해당 부분을 그대로 보내주시면 패치를 맞춰 드리겠습니다.")
        sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] 5개 지점 모두 확인되었습니다. 실제 변경은 하지 않았습니다.")
        return

    # 2) 백업
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup)
    print(f"\n백업: {backup.name}")

    # 3) 치환
    patched = original
    for _, old, new in PATCHES:
        patched = patched.replace(old, new, 1)

    path.write_text(patched, encoding="utf-8")
    print(f"패치 완료: {len(PATCHES)}개 지점")

    # 4) 구문 검사
    try:
        compile(patched, str(path), "exec")
        print("구문 검사 통과")
    except SyntaxError as exc:
        shutil.copy2(backup, path)
        sys.exit(f"[중단] 구문 오류가 발생해 원복했습니다: {exc}")

    print("\n다음 확인 명령:")
    print("  python -c \"from src.finguide_rag.generation.generator import Answer;"
          " import dataclasses; print([f.name for f in dataclasses.fields(Answer)])\"")


if __name__ == "__main__":
    main()
