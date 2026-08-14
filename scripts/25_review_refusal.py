"""
25_review_refusal.py

목적
----
생성된 거절 평가셋을 검수한다. 질문과 근거 청크를 나란히 보여주고
사용자가 유지·제외·유형변경을 판단한다.

왜 사람이 봐야 하는가
------------------
생성 단계에서 LLM 역검증을 거치지만 판정이 불안정하다. 실제로
"인지세는 대출금액별로 얼마인가요"는 통과했는데 "인지세는 대출금액별로
각각 몇 원인가요"는 탈락하는 사례가 있었다. 거의 같은 질문이다.

특히 blank_value 유형은 **원문에 정말 값이 없는지** 눈으로 확인해야 한다.
상품설명서에는 중도상환수수료율이나 연체가산율이 명시된 경우가 많아,
'몇 퍼센트인가요' 질문이 실제로는 답변 가능할 수 있다.

이 평가셋이 거절 로직의 유일한 기준이 되므로 품질이 중요하다.

사용법
-----
    python scripts/25_review_refusal.py            # 검수
    python scripts/25_review_refusal.py --all      # 대조군까지 전부
    python scripts/25_review_refusal.py --summary  # 현황만
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_CSV = PROJECT_ROOT / "data" / "eval" / "refusal_eval_draft.csv"
CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"

TYPES = ["blank_value", "out_of_scope", "personalized", "time_variant"]

TYPE_DESC = {
    "blank_value": "문서에 항목은 있으나 값이 공란",
    "out_of_scope": "문서에 아예 없는 정보",
    "personalized": "개인 정보가 있어야 답할 수 있음",
    "time_variant": "시점에 따라 달라지는 값",
    "answerable": "답할 수 있는 질문 (대조군)",
}


def load_rows() -> list[dict]:
    if not DRAFT_CSV.exists():
        sys.exit(f"{DRAFT_CSV} 없음. 먼저 24_generate_refusal_set.py 를 실행하세요.")
    with DRAFT_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("reviewed", "")
    return rows


def load_chunks() -> dict[str, dict]:
    if not CHUNKS_PATH.exists():
        return {}
    out = {}
    for line in CHUNKS_PATH.open(encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            out[c["chunk_id"]] = c
    return out


def save_rows(rows: list[dict]) -> None:
    try:
        with DRAFT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        sys.exit(f"\n저장 실패: {DRAFT_CSV.name} 이 열려 있습니다. 닫고 다시 실행하세요.")


def wrap(text: str, width: int = 74, indent: str = "      ") -> str:
    text = text.replace("\n", " ")
    lines = []
    while text:
        lines.append(indent + text[:width])
        text = text[width:]
    return "\n".join(lines)


def highlight_blanks(text: str) -> str:
    """공란 표시를 눈에 띄게 바꾼다.

    blank_value 판정의 핵심 근거이므로 원문에서 바로 확인할 수 있어야 한다.
    """
    import re
    return re.sub(r"\(\s*\)|（\s*）|_{3,}|☐|□", lambda m: f"[[{m.group()}]]", text)


def show_item(idx: int, total: int, row: dict, chunks: dict) -> None:
    print("\n" + "=" * 84)
    print(f"[{idx}/{total}]  {row['query_id']}  |  {row['refusal_type']}  "
          f"|  {TYPE_DESC.get(row['refusal_type'], '')}")
    print("=" * 84)

    print("\n  [질문]")
    print(wrap(row["question"]))

    if row.get("why_unanswerable"):
        print(f"\n  [생성 근거] {row['why_unanswerable']}")

    if row.get("verified") == "N":
        print("\n  !! LLM 역검증에서 '답변 가능' 판정을 받았습니다.")

    print(f"\n  [출처] {row.get('doc_display_name', '')[:64]}")

    src = chunks.get(row.get("source_chunk_id", ""), {})
    if src:
        body = src.get("text", "")[:600]
        if row["refusal_type"] == "blank_value":
            body = highlight_blanks(body)
            print("  [근거 청크]  ([[ ]] 로 표시된 곳이 공란입니다)")
        else:
            print("  [근거 청크]")
        print(wrap(body))
    else:
        print("  [근거 청크] 찾을 수 없음")

    print("\n  [판단 기준]")
    if row["refusal_type"] == "blank_value":
        print("    이 문서에 질문이 묻는 '값'이 실제로 없는가?")
        print("    값이 명시되어 있다면 답변 가능하므로 제외해야 한다.")
    elif row["refusal_type"] == "out_of_scope":
        print("    하나은행 문서 전체를 뒤져도 답이 없는가?")
        print("    다른 문서나 FAQ에 있을 법하면 제외한다.")
    else:
        print("    실제로 답할 수 없는 질문인가?")


def print_summary(rows: list[dict]) -> None:
    kept = [r for r in rows if r.get("keep", "Y").upper() != "N"]
    dropped = [r for r in rows if r.get("keep", "Y").upper() == "N"]

    refuse = [r for r in kept if r["expected"] == "refuse"]
    answer = [r for r in kept if r["expected"] == "answer"]

    print("\n" + "=" * 70)
    print("거절 평가셋 현황")
    print("=" * 70)
    print(f"  전체     : {len(rows)}건")
    print(f"  유지     : {len(kept)}건  (거절 {len(refuse)} / 대조군 {len(answer)})")
    print(f"  제외     : {len(dropped)}건")

    print("\n  [거절 유형별]")
    counts = Counter(r["refusal_type"] for r in refuse)
    for t in TYPES:
        n = counts.get(t, 0)
        bar = "█" * n
        print(f"    {t:<16} {n:>3}건  {bar}")

    if len(refuse) < 20:
        print("\n  !! 거절 질문이 20건 미만입니다. 지표가 불안정할 수 있습니다.")
    if not answer:
        print("\n  !! 대조군이 없습니다. 과잉 거절을 측정할 수 없습니다.")

    if dropped:
        print("\n  [제외된 항목]")
        for r in dropped[:12]:
            note = f" — {r['review_note']}" if r.get("review_note") else ""
            print(f"    {r['query_id']} [{r['refusal_type']:<13}] "
                  f"{r['question'][:40]}{note[:36]}")
        if len(dropped) > 12:
            print(f"    ... 외 {len(dropped) - 12}건")
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="대조군까지 전부 검수")
    ap.add_argument("--summary", action="store_true", help="현황만 출력")
    args = ap.parse_args()

    rows = load_rows()

    if args.summary:
        print_summary(rows)
        return

    chunks = load_chunks()

    targets = [
        (i, r) for i, r in enumerate(rows)
        if r.get("reviewed", "").upper() != "Y"
        and (args.all or r["expected"] == "refuse")
    ]

    if not targets:
        print("검수할 항목이 없습니다.")
        print_summary(rows)
        return

    print("=" * 84)
    print("거절 평가셋 검수")
    print("=" * 84)
    print(f"  {len(targets)}건을 검수합니다.")
    print()
    print("  입력키")
    print("    y = 유지 (거절 대상이 맞음)")
    print("    n = 제외 (실제로는 답할 수 있음)")
    print("    t = 유형 변경 후 유지")
    print("    s = 판단 보류 (다음 실행 때 다시 나타남)")
    print("    q = 중단하고 저장")
    print("=" * 84)

    changed = 0
    for order, (idx, row) in enumerate(targets, 1):
        show_item(order, len(targets), row, chunks)

        while True:
            try:
                ans = input("\n  거절 대상이 맞습니까? [y/n/t/s/q] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "q"
            if ans in ("y", "n", "t", "s", "q", ""):
                break
            print("  y, n, t, s, q 중 하나를 입력하세요.")

        if ans == "q":
            print("\n  검수를 중단합니다. 지금까지의 결과를 저장합니다.")
            break
        if ans == "s":
            continue

        if ans == "t":
            print("\n  유형을 고르세요")
            for i, t in enumerate(TYPES, 1):
                print(f"    {i}. {t:<16} {TYPE_DESC[t]}")
            try:
                pick = input("  번호 > ").strip()
                if pick.isdigit() and 1 <= int(pick) <= len(TYPES):
                    old_type = rows[idx]["refusal_type"]
                    rows[idx]["refusal_type"] = TYPES[int(pick) - 1]
                    rows[idx]["review_note"] = f"유형 변경: {old_type} -> {TYPES[int(pick) - 1]}"
                    changed += 1
                    print(f"  -> {TYPES[int(pick) - 1]} 로 변경, 유지")
            except (EOFError, KeyboardInterrupt):
                pass
            rows[idx]["keep"] = "Y"
            rows[idx]["reviewed"] = "Y"
            continue

        if ans == "n":
            rows[idx]["keep"] = "N"
            rows[idx]["reviewed"] = "Y"
            try:
                note = input("  제외 사유 (엔터로 생략) > ").strip()
            except (EOFError, KeyboardInterrupt):
                note = ""
            rows[idx]["review_note"] = note or "검수 중 제외"
            changed += 1
            print("  -> 제외")
        else:
            rows[idx]["keep"] = "Y"
            rows[idx]["reviewed"] = "Y"
            print("  -> 유지")

    save_rows(rows)
    print(f"\n저장 완료 ({changed}건 변경)")

    print_summary(rows)
    print("\n다음: python scripts/26_finalize_refusal.py")


if __name__ == "__main__":
    main()
