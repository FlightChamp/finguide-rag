"""
35_apply_review.py

목적
----
검수 판단 근거 문서에서 판정을 읽어, 스크립트가 생성한 검수 템플릿에
옮겨 넣는다.

왜 필요한가
---------
검수 결과가 두 곳에 존재한다.

    판단 근거 문서   사람이 읽는 글. 기준, 논거, 케이스별 설명이 담긴다.
    검수 템플릿      기계가 읽는 라벨. <!-- ITEM NN --> 마커와
                     "- 환각 판정: pass" 형식으로 되어 있다.

둘 다 필요하다. 근거 없이 라벨만 남으면 나중에 왜 그렇게 판정했는지
알 수 없고, 라벨 없이 글만 남으면 자동 대조가 불가능하다.

그런데 사람이 판정을 손으로 옮겨 적으면 오타가 난다. 10건 × 5항목이면
50칸이고, 한 칸만 틀려도 일치율 계산이 어긋난다. 옮기는 일은 기계가 한다.

읽는 형식
--------
근거 문서에서 다음 두 형식을 모두 인식한다.

    1. 케이스별 블록
       ## [03] 정상 답변
       - 환각 판정: fail
       - 메모: ...

    2. 판정표
       | 03 | 정상 답변 | fail | fail | fail | fail | fail | 메모 |

둘 다 있으면 블록 형식을 우선한다. 표는 요약이고 블록이 원본이기 때문이다.
두 형식의 판정이 다르면 경고한다. 문서 안에서 값이 어긋나 있다는 뜻이므로
사람이 확인해야 한다.

안전장치
-------
원본 템플릿은 .bak 으로 백업한다.
템플릿에 있으나 근거 문서에 없는 문항은 건드리지 않고 보고한다.
판정값이 pass/fail/na 가 아니면 채우지 않고 보고한다.

사용법
-----
    python scripts/35_apply_review.py
    python scripts/35_apply_review.py --dry-run     # 미리보기만
    python scripts/35_apply_review.py --source data/eval/xxx.md --target data/eval/yyy.md
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

DEFAULT_SOURCE = EVAL_DIR / "g_eval_recheck_rationale.md"
DEFAULT_TARGET = EVAL_DIR / "g_eval_recheck.md"

CRITERIA = ["환각", "상품일치", "수치정확", "미확인신고", "실무활용"]
VALID = {"pass", "fail", "na"}

# 근거 문서의 케이스별 블록 헤더:  ## [03] 정상 답변
BLOCK_HEAD = re.compile(r"^#{1,4}\s*\[(\d{1,2})\]", re.M)
# 판정 줄:  - 환각 판정: fail
VERDICT = re.compile(r"^-\s*(\S+?)\s*판정:\s*([A-Za-z가-힣?]+)", re.M)
# 메모 줄
MEMO = re.compile(r"^-\s*메모:\s*(.*)$", re.M)
# 판정표 행
TABLE_ROW = re.compile(
    r"^\|\s*(\d{1,2})\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|(.*)\|\s*$",
    re.M,
)
# 템플릿의 문항 마커
ITEM_MARK = re.compile(r"<!--\s*ITEM\s*(\d+)\s*-->")


def parse_blocks(text: str) -> dict[str, dict]:
    """케이스별 블록에서 판정을 읽는다."""
    out: dict[str, dict] = {}
    marks = list(BLOCK_HEAD.finditer(text))
    for n, m in enumerate(marks):
        iid = f"{int(m.group(1)):02d}"
        end = marks[n + 1].start() if n + 1 < len(marks) else len(text)
        chunk = text[m.start():end]

        rec: dict = {}
        for name, val in VERDICT.findall(chunk):
            if name in CRITERIA:
                rec[name] = val.strip().lower()
        if not rec:
            continue
        mm = MEMO.search(chunk)
        rec["memo"] = mm.group(1).strip() if mm else ""
        # 같은 번호가 여러 번 나오면 판정칸이 더 많이 채워진 쪽을 쓴다.
        if iid in out and sum(1 for c in CRITERIA if c in out[iid]) >= \
                sum(1 for c in CRITERIA if c in rec):
            continue
        out[iid] = rec
    return out


def parse_table(text: str) -> dict[str, dict]:
    """판정표에서 판정을 읽는다."""
    out: dict[str, dict] = {}
    for m in TABLE_ROW.findall(text):
        verdicts = [v.strip().lower() for v in m[2:7]]
        if not all(v in VALID for v in verdicts):
            continue
        out[f"{int(m[0].strip()):02d}"] = {
            **dict(zip(CRITERIA, verdicts)),
            "memo": m[7].strip(),
        }
    return out


def fill(template: str, verdicts: dict[str, dict]) -> tuple[str, list, list]:
    """템플릿의 판정 줄을 채운다."""
    marks = list(ITEM_MARK.finditer(template))
    if not marks:
        sys.exit("템플릿에 <!-- ITEM NN --> 마커가 없습니다. 대상 파일을 확인하세요.")

    filled, skipped = [], []
    pieces, cursor = [], 0

    for n, m in enumerate(marks):
        iid = f"{int(m.group(1)):02d}"
        end = marks[n + 1].start() if n + 1 < len(marks) else len(template)
        pieces.append(template[cursor:m.start()])
        chunk = template[m.start():end]
        cursor = end

        rec = verdicts.get(iid)
        if not rec:
            skipped.append((iid, "근거 문서에 없음"))
            pieces.append(chunk)
            continue

        changed = []

        def sub(match: re.Match) -> str:
            name = match.group(1)
            if name not in CRITERIA:
                return match.group(0)
            val = rec.get(name)
            if val not in VALID:
                return match.group(0)
            changed.append(name)
            # 주석 등 줄 뒤쪽은 그대로 둔다.
            return match.group(0).replace(f"판정: {match.group(2)}",
                                          f"판정: {val}", 1)

        chunk = VERDICT.sub(sub, chunk)

        memo = rec.get("memo", "")
        if memo:
            chunk = MEMO.sub(lambda _: f"- 메모: {memo}", chunk, count=1)

        pieces.append(chunk)
        if len(changed) == len(CRITERIA):
            filled.append(iid)
        else:
            missing = [c for c in CRITERIA if c not in changed]
            skipped.append((iid, f"일부만 채움 (누락: {', '.join(missing)})"))
            filled.append(iid)

    pieces.append(template[cursor:])
    return "".join(pieces), filled, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="판단 근거 문서")
    ap.add_argument("--target", default=None, help="채울 검수 템플릿")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.source) if args.source else DEFAULT_SOURCE
    tgt = Path(args.target) if args.target else DEFAULT_TARGET
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    if not tgt.is_absolute():
        tgt = PROJECT_ROOT / tgt

    print("=" * 76)
    print("검수 판정 이관")
    print("=" * 76)

    if not src.exists():
        sys.exit(f"근거 문서 없음: {src}")
    if not tgt.exists():
        sys.exit(f"템플릿 없음: {tgt}\n34번을 먼저 실행하세요.")

    text = src.read_text(encoding="utf-8")
    blocks = parse_blocks(text)
    table = parse_table(text)
    print(f"  근거 문서  {src.name}")
    print(f"    케이스 블록 {len(blocks)}건 / 판정표 {len(table)}건")

    # 두 형식이 어긋나면 사람이 확인해야 한다.
    conflicts = []
    for iid in sorted(set(blocks) & set(table)):
        for c in CRITERIA:
            b, t = blocks[iid].get(c), table[iid].get(c)
            if b and t and b != t:
                conflicts.append((iid, c, b, t))
    if conflicts:
        print(f"\n  [경고] 블록과 표의 판정이 다른 곳 {len(conflicts)}건")
        for iid, c, b, t in conflicts:
            print(f"    [{iid}] {c}: 블록={b} / 표={t}  -> 블록 값을 사용")
        print("    문서 안에서 값이 어긋나 있습니다. 확인하십시오.")

    verdicts = {**table, **blocks}   # 블록 우선
    print(f"  이관 대상  {len(verdicts)}건")

    template = tgt.read_text(encoding="utf-8")
    result, filled, skipped = fill(template, verdicts)

    print(f"\n  템플릿  {tgt.name}")
    print(f"    채움 {len(filled)}건")
    for iid in filled:
        v = verdicts[iid]
        line = "  ".join(f"{c}={v.get(c, '-')}" for c in CRITERIA)
        print(f"      [{iid}] {line}")
    if skipped:
        print(f"    미처리 {len(skipped)}건")
        for iid, why in skipped:
            print(f"      [{iid}] {why}")

    remaining = result.count("판정: ?")
    if remaining:
        print(f"\n  [주의] 아직 '?' 로 남은 판정칸 {remaining}개")

    if args.dry_run:
        print("\n  --dry-run 이므로 저장하지 않았습니다.")
        return

    backup = tgt.with_suffix(".md.bak")
    shutil.copy2(tgt, backup)
    tgt.write_text(result, encoding="utf-8")
    print(f"\n  저장 완료. 백업 → {backup.relative_to(PROJECT_ROOT)}")
    print(f"  확인: python scripts/33_calibrate_judge.py --review "
          f"{tgt.relative_to(PROJECT_ROOT).as_posix()}")
    print("=" * 76)


if __name__ == "__main__":
    main()
