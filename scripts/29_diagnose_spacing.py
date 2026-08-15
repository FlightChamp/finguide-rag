"""
29_diagnose_spacing.py  (v2)

목적
----
청크 원문의 띄어쓰기 손상을 측정하고, 복원 정책 두 가지를 비교한다.

v1 에서 확인된 것
---------------
손상은 전체의 7.3%(220/2999)이며 설명서에 집중된다.
FAQ 0건, 약관 2건, 설명서 218건. 근거 발췌의 핵심인 약관은 사실상
온전하다.

그런데 Kiwi 복원을 전체에 적용하니 2998/2999 청크가 바뀌었다. 멀쩡한
93%까지 건드리면서 과분할이 발생했다.

    가입일 -> 가입 일 / 총급여액을 -> 총 급여 액을 / 주민등록증 -> 주민 등록증

또한 불변식 위반이 270건 나왔는데, 원인은 Kiwi 가 줄바꿈을 제거하는
것이었다.

    "있어야\n한다"  ->  "있어야한다"

v1 의 불변식이 공백(" ")만 제거해 비교했기 때문에 줄바꿈 소실을
"내용 변경"으로 잡아냈다. 검사는 옳게 동작했으나 기준이 느슨했다.

v2 의 정책
---------
naive       전체 청크에 그대로 적용 (v1 방식, 비교군)
selective   아래 세 가지를 모두 적용

    1. 손상 청크에만 적용한다 (공백 비율 < 0.10)
       멀쩡한 텍스트를 건드리지 않으므로 과분할이 발생하지 않는다.

    2. 줄 단위로 적용한다
       개행으로 분리해 각 줄에만 복원하고 다시 합친다. Kiwi 가 줄바꿈을
       보지 못하므로 줄 구조 소실이 구조적으로 불가능하다.

    3. 모든 공백문자를 제거해 비교한다
       공백문자 전체를 제거한 뼈대가 원문과 같아야 통과. 실패하면
       원문을 그대로 쓴다.

이 스크립트는 아무것도 수정하지 않는다. 측정만 한다.

사용법
-----
    python scripts/29_diagnose_spacing.py
    python scripts/29_diagnose_spacing.py --samples 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"

# 한국어 정상 문서의 공백 비율은 대략 12~18% 다.
# 이보다 크게 낮으면 추출 과정에서 공백이 소실된 것으로 본다.
HEALTHY_SPACE_RATIO = 0.10

LONG_RUN_CHARS = 12
RE_HANGUL_RUN = re.compile(r"[가-힣]{%d,}" % LONG_RUN_CHARS)
RE_WS = re.compile(r"\s")


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} 없음.")
    out = []
    for line in CHUNKS_PATH.open(encoding="utf-8"):
        if line.strip():
            out.append(json.loads(line))
    return out


def space_ratio(text: str) -> float:
    return text.count(" ") / len(text) if text else 0.0


def longest_run(text: str) -> int:
    return max((len(r) for r in RE_HANGUL_RUN.findall(text)), default=0)


def longest_run_of(group: list[dict]) -> int:
    return max((longest_run(c["text"]) for c in group), default=0)


def skeleton(text: str) -> str:
    """모든 공백문자를 제거한 뼈대. 불변식 비교의 기준."""
    return RE_WS.sub("", text)


def restore_naive(kiwi, text: str) -> str:
    """v1 방식. 전체에 그대로 적용."""
    return kiwi.space(text)


def restore_selective(kiwi, text: str) -> tuple[str, str]:
    """v2 방식. 손상 청크에만, 줄 단위로, 불변식 검사 후 적용.

    반환: (결과 텍스트, 사유)
      untouched  손상 기준 미달로 건드리지 않음
      restored   복원 적용됨
      reverted   불변식 위반으로 원문 유지
      error      복원 중 예외 발생, 원문 유지
    """
    if space_ratio(text) >= HEALTHY_SPACE_RATIO:
        return text, "untouched"

    try:
        lines = text.split("\n")
        out = [kiwi.space(ln) if ln.strip() else ln for ln in lines]
        restored = "\n".join(out)
    except Exception:
        return text, "error"

    if skeleton(restored) != skeleton(text):
        return text, "reverted"
    return restored, "restored"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    print("=" * 76)
    print("띄어쓰기 진단 v2 — 복원 정책 비교")
    print("=" * 76)

    chunks = load_chunks()
    print(f"  청크 {len(chunks)}개")

    # ---------- 1. 손상 분포 ----------
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_type[c.get("doc_type", "?")].append(c)

    print("\n" + "=" * 76)
    print("문서 유형별 손상 분포")
    print("=" * 76)
    print(f"  {'유형':<10}{'청크':>7}{'공백비율':>10}{'손상청크':>14}{'최장연속':>10}")
    print("  " + "-" * 51)

    damaged_ids = set()
    for dt in sorted(by_type):
        g = by_type[dt]
        avg = sum(space_ratio(c["text"]) for c in g) / len(g)
        dmg = [c for c in g if space_ratio(c["text"]) < HEALTHY_SPACE_RATIO]
        damaged_ids |= {c["chunk_id"] for c in dmg}
        print(f"  {dt:<10}{len(g):>7}{avg:>10.3f}"
              f"{len(dmg):>8} ({len(dmg) / len(g):>4.0%}){longest_run_of(g):>8}자")

    print(f"\n  손상 기준: 공백 비율 < {HEALTHY_SPACE_RATIO:.2f}")
    print(f"  전체 손상: {len(damaged_ids)} / {len(chunks)} "
          f"({len(damaged_ids) / len(chunks):.1%})")

    # 약관 손상 청크는 근거 발췌의 핵심이므로 개별 확인한다.
    yak = [c for c in chunks
           if c.get("doc_type") == "약관" and c["chunk_id"] in damaged_ids]
    if yak:
        print(f"\n  [약관 손상 {len(yak)}건 — 근거 발췌 핵심이므로 개별 확인]")
        for c in yak:
            print(f"    {c['chunk_id']}  공백 {space_ratio(c['text']):.3f}")
            print(f"      {c.get('doc_display_name', '')} / "
                  f"{c.get('section', '') or '구간명 없음'}")
            print(f"      {c['text'][:100]}")

    # ---------- 2. Kiwi 준비 ----------
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        sys.exit("  kiwipiepy 가 설치되어 있지 않습니다.")

    kiwi = Kiwi()
    if not hasattr(kiwi, "space"):
        sys.exit("  이 kiwipiepy 버전에는 space() 가 없습니다.")

    # ---------- 3. 두 정책 비교 ----------
    print("\n" + "=" * 76)
    print("정책 비교")
    print("=" * 76)
    print("  naive      전체 적용 (v1)")
    print("  selective  손상 청크만 + 줄 단위 + 불변식 검사 (v2)\n")

    naive_changed = naive_violation = 0
    sel_counts: dict[str, int] = defaultdict(int)
    sel_added = 0

    for i, c in enumerate(chunks, 1):
        original = c["text"]

        try:
            nv = restore_naive(kiwi, original)
            if nv != original:
                naive_changed += 1
            if skeleton(nv) != skeleton(original):
                naive_violation += 1
        except Exception:
            naive_violation += 1

        sv, why = restore_selective(kiwi, original)
        sel_counts[why] += 1
        if why == "restored":
            sel_added += sv.count(" ") - original.count(" ")

        if i % 500 == 0:
            print(f"    {i}/{len(chunks)}")

    n = len(chunks)
    print(f"\n  {'정책':<12}{'변경':>12}{'불변식 위반':>14}")
    print("  " + "-" * 38)
    print(f"  {'naive':<12}{naive_changed:>6} ({naive_changed / n:>3.0%})"
          f"{naive_violation:>10}건")
    print(f"  {'selective':<12}{sel_counts['restored']:>6} "
          f"({sel_counts['restored'] / n:>3.0%}){sel_counts['reverted']:>10}건")

    print(f"\n  [selective 상세]")
    for k in ("untouched", "restored", "reverted", "error"):
        if sel_counts[k]:
            print(f"    {k:<12} {sel_counts[k]:>5}건")
    print(f"    추가된 공백  {sel_added:,}개")

    if sel_counts["reverted"] == 0 and sel_counts["error"] == 0:
        print("\n  -> selective 정책은 위반 0건. 손상 청크만 안전하게 복원된다.")
    else:
        print(f"\n  -> 되돌린 청크 {sel_counts['reverted']}건은 원문을 그대로 쓴다.")
        print("     표시 계층에서 같은 검사를 수행하므로 안전하다.")

    # ---------- 4. 예시 ----------
    print("\n" + "=" * 76)
    print("복원 예시 (selective)")
    print("=" * 76)

    dmg = sorted((c for c in chunks if c["chunk_id"] in damaged_ids),
                 key=lambda c: space_ratio(c["text"]))
    for c in dmg[:args.samples]:
        sv, why = restore_selective(kiwi, c["text"])
        print(f"\n  [{c['chunk_id']}] {c.get('doc_type', '?')} "
              f"공백 {space_ratio(c['text']):.3f} -> {why}")
        print(f"    전: {c['text'][:140]}")
        print(f"    후: {sv[:140]}")

    print("\n" + "=" * 76)
    print("멀쩡한 청크 확인 — selective 는 건드리지 않아야 한다")
    print("=" * 76)
    healthy = [c for c in chunks if c["chunk_id"] not in damaged_ids]
    for c in healthy[:args.samples]:
        sv, why = restore_selective(kiwi, c["text"])
        nv = restore_naive(kiwi, c["text"])
        print(f"\n  [{c['chunk_id']}] selective={why}")
        print(f"    원문:      {c['text'][:110]}")
        if nv != c["text"]:
            print(f"    naive라면: {nv[:110]}")

    print("\n" + "=" * 76)


if __name__ == "__main__":
    main()
