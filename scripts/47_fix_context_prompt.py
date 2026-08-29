# -*- coding: utf-8 -*-
"""
47_fix_context_prompt.py — 근거 유용성 판정 프롬프트 결함 수정

발견한 결함
----------
46번으로 추가한 Context Precision 의 판정 사유를 전수로 읽어 세 가지
결함을 찾았다. 값 자체가 아니라 판정 사유를 읽어야 보이는 것들이었다.

  1. 중복을 이유로 탈락시킨다
     id02  "근거 1과 중복되는 내용이며" -> not useful
     id08  "제출 가능한 서류 목록을 안내하고 있으나 (근거 1보다 덜 구체)"
     RAGAs 의 Context Precision 은 "질문에 답하는 데 유용한가"를 묻는다.
     "가장 좋은 근거인가"가 아니다. 중복이어도 답에 쓰이면 유용하다.

  2. 근거를 뭉뚱그려 판정하고 verdict 를 1개만 반환한다
     id09, id13  "근거들은 모두 ... " -> verdict 1개
     나머지 rank 는 '판정 없음'이 되어 not useful 로 집계됐다. 실제
     의미는 정반대다. 프롬프트가 rank 별 개별 판정을 강제하지 않았다.

  3. 누락 처리가 결과를 왜곡한다
     누락을 보수적으로 not useful 처리했는데, 2번 때문에 누락이
     구조적으로 발생하고 있었다. 누락은 이제 오류로 드러내고,
     프롬프트에서 애초에 발생하지 않게 막는다.

수정 내용
--------
- 근거마다 반드시 개별 verdict 를 내도록 개수를 명시하고 강제한다.
- 중복이 탈락 사유가 아님을 명시한다.
- 상품이 다르면 유용하지 않다는 기준은 유지한다(이번 검증의 핵심).
- 판정이 누락되면 콘솔에 경고를 남긴다.

캐시
----
프롬프트가 바뀌면 캐시 키도 바뀌므로 ctxprec 판정만 다시 호출된다.
stmt / verify / qgen 캐시 45 건은 그대로 적중한다.

사용법
-----
    python scripts/47_fix_context_prompt.py --dry-run
    python scripts/47_fix_context_prompt.py
    python scripts/42_ragas_measure.py --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET = Path("scripts") / "42_ragas_measure.py"
ROOT_MARKERS = ("pyproject.toml", "src", "scripts", "reports")
SENTINEL = "각 근거를 독립적으로 판정한다"


OLD_BLOCK = '''        system = (
            "너는 검색 근거 평가기다. 각 근거가 주어진 질문에 답하는 데 "
            "실제로 유용한지 판정한다. 질문과 주제가 비슷하다는 이유만으로 "
            "유용하다고 하지 마라. 질문이 특정 상품을 가리키는데 근거가 "
            "다른 상품에 대한 것이면 유용하지 않다. "
            'JSON만 출력한다: {"verdicts": [{"rank": 1, "useful": true, '
            '"reason": "..."}]}'
        )
        block = "\\n\\n".join(
            f"[근거 {i}]\\n{c}" for i, c in enumerate(contexts, start=1))
        user = f"[질문]\\n{question}\\n\\n{block}"
        key = cache_key("ctxprec", self.model, question, block)
        result = self._chat_json(key, system, user)
        verdicts = [v for v in result.get("verdicts", []) if isinstance(v, dict)]

        # 판정이 누락된 rank 는 유용하지 않은 것으로 본다(보수적).
        by_rank = {}
        for i, v in enumerate(verdicts, start=1):
            try:
                rank = int(v.get("rank", i) or i)
            except (TypeError, ValueError):
                rank = i
            by_rank[rank] = v
        return [by_rank.get(i, {"rank": i, "useful": False,
                                "reason": "판정 없음"})
                for i in range(1, len(contexts) + 1)]'''


NEW_BLOCK = '''        n = len(contexts)
        system = (
            "너는 검색 근거 평가기다. 각 근거가 주어진 질문에 답하는 데 "
            "유용한지 판정한다.\\n"
            "판정 기준:\\n"
            "- 각 근거를 독립적으로 판정한다. 다른 근거와 내용이 겹치거나 "
            "덜 구체적이라는 이유로 유용하지 않다고 하지 마라. 그 근거 "
            "하나만 놓고 봤을 때 질문에 답하는 데 쓸 수 있으면 유용하다.\\n"
            "- 질문과 주제가 비슷하다는 이유만으로 유용하다고 하지 마라. "
            "질문이 묻는 항목 자체를 다루지 않으면 유용하지 않다.\\n"
            "- 질문이 특정 상품을 가리키는데 근거가 다른 상품에 대한 "
            "것이면 유용하지 않다.\\n"
            f"근거는 {n}개다. 반드시 {n}개의 판정을 rank 1부터 {n}까지 "
            "빠짐없이 낸다. 여러 근거를 묶어서 한 번에 판정하지 마라.\\n"
            'JSON만 출력한다: {"verdicts": [{"rank": 1, "useful": true, '
            '"reason": "..."}]}'
        )
        block = "\\n\\n".join(
            f"[근거 {i}]\\n{c}" for i, c in enumerate(contexts, start=1))
        user = f"[질문]\\n{question}\\n\\n{block}\\n\\n판정 {n}개를 내십시오."
        key = cache_key("ctxprec", self.model, question, block, str(n))
        result = self._chat_json(key, system, user)
        verdicts = [v for v in result.get("verdicts", []) if isinstance(v, dict)]

        by_rank = {}
        for i, v in enumerate(verdicts, start=1):
            try:
                rank = int(v.get("rank", i) or i)
            except (TypeError, ValueError):
                rank = i
            by_rank[rank] = v

        # 누락은 숨기지 않고 드러낸다. 이전 구현은 누락을 조용히
        # not useful 로 처리해, 판정기가 근거를 묶어 답하던 결함이
        # 값에 그대로 반영됐다.
        missing = [i for i in range(1, n + 1) if i not in by_rank]
        if missing:
            print(f"        [경고] 근거 판정 누락 rank {missing} "
                  f"— 유용하지 않음으로 처리합니다.")
        return [by_rank.get(i, {"rank": i, "useful": False,
                                "reason": "판정 누락"})
                for i in range(1, n + 1)]'''


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
    sys.exit(f"[중단] 인코딩 판별 실패: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="유용성 판정 프롬프트 수정")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = find_root()
    path = root / TARGET
    if not path.exists():
        sys.exit(f"[중단] 대상 파일이 없습니다: {path}")

    print(f"대상: {path}")
    original = read_text(path)

    if SENTINEL in original:
        print("이미 수정된 파일입니다. 변경하지 않고 종료합니다.")
        return

    count = original.count(OLD_BLOCK)
    print(f"  [{'OK' if count == 1 else '실패'}] 판정 프롬프트 블록 "
          f"(일치 {count}회)")
    if count != 1:
        print("\\n[중단] 치환 지점을 찾지 못했습니다. 파일을 변경하지 "
              "않았습니다.")
        print("46번 패치가 적용되지 않았거나 이미 수정된 상태일 수 있습니다.")
        sys.exit(1)

    if args.dry_run:
        print("\\n[dry-run] 확인 완료. 실제 변경은 하지 않았습니다.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup)
    print(f"\\n백업: {backup.name}")

    patched = original.replace(OLD_BLOCK, NEW_BLOCK, 1)
    path.write_text(patched, encoding="utf-8")
    print("수정 완료")

    try:
        compile(patched, str(path), "exec")
        print("구문 검사 통과")
    except SyntaxError as exc:
        shutil.copy2(backup, path)
        sys.exit(f"[중단] 구문 오류로 원복했습니다: {exc}")

    print("\\n다음 실행:")
    print("  python scripts/42_ragas_measure.py --yes")
    print("  (ctxprec 판정 15건만 재호출됩니다. 나머지 45건은 캐시 적중)")


if __name__ == "__main__":
    main()
