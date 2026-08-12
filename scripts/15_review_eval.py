"""
15_review_eval.py

목적
----
생성된 평가셋 초안을 검수한다. 문제가 의심되는 항목만 골라 질문과
근거 청크를 나란히 보여주고, 사용자가 유지/제외를 판단한다.

왜 도구가 필요한가
----------------
Excel로 검수하려면 질문을 읽고 원본 청크와 대조해야 하는데, 청크 내용을
보려면 chunks.jsonl 을 별도로 뒤져야 한다. 이 왕복이 병목이다.
여기서는 둘을 한 화면에 보여주므로 판단만 하면 된다.

검수 대상 선별 기준
-----------------
- overlap_ratio 0.6 초과 : 청크 문장을 베꼈을 가능성
- hard_issue 존재        : 난이도 지시 위반
- 근거가 20위 밖         : 질문이 잘못됐거나 정말 어려운 경우 (구별 필요)
- 정답이 3개 이상 추가됨  : LLM 판정이 관대했을 가능성

사용법
-----
    python scripts/15_review_eval.py           # 의심 항목만 검수
    python scripts/15_review_eval.py --all     # 전체 110건 검수
    python scripts/15_review_eval.py --summary # 현재 상태만 확인
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_draft.csv"
CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"

OVERLAP_THRESHOLD = 0.6
MANY_RELEVANT = 3


# ------------------------------------------------------------------
# 로드
# ------------------------------------------------------------------


def load_rows() -> list[dict]:
    if not DRAFT_CSV.exists():
        sys.exit(f"{DRAFT_CSV} 없음. 먼저 14_generate_eval.py 를 실행하세요.")
    with DRAFT_CSV.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_chunks() -> dict[str, dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} 없음.")
    out = {}
    for line in CHUNKS_PATH.open(encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            out[c["chunk_id"]] = c
    return out


def save_rows(rows: list[dict]) -> None:
    """검수 결과를 원본 CSV에 덮어쓴다."""
    try:
        with DRAFT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        sys.exit(
            f"\n저장 실패: {DRAFT_CSV.name} 이 열려 있습니다.\n"
            "Excel 등에서 파일을 닫고 다시 실행하세요."
        )


# ------------------------------------------------------------------
# 검수 대상 선별
# ------------------------------------------------------------------


def flag_reasons(row: dict) -> list[str]:
    """이 항목을 검수해야 하는 이유를 반환한다. 비어 있으면 검수 불필요."""
    reasons: list[str] = []

    try:
        overlap = float(row.get("overlap_ratio", 0) or 0)
    except ValueError:
        overlap = 0.0
    if overlap > OVERLAP_THRESHOLD:
        reasons.append(f"청크 베낌 의심 (겹침 {overlap:.0%})")

    if row.get("hard_issue"):
        reasons.append(f"난이도 위반 ({row['hard_issue'][:24]})")

    if row.get("found_rank") == "0":
        reasons.append("근거가 20위 밖 (질문이 잘못됐을 수 있음)")

    try:
        n_rel = int(row.get("n_relevant", 1) or 1)
    except ValueError:
        n_rel = 1
    if n_rel >= MANY_RELEVANT:
        reasons.append(f"정답 {n_rel}개 (판정이 관대했을 수 있음)")

    return reasons


# ------------------------------------------------------------------
# 화면 출력
# ------------------------------------------------------------------


def wrap(text: str, width: int = 76, indent: str = "      ") -> str:
    """긴 텍스트를 줄바꿈한다. 터미널에서 읽기 쉽게 한다."""
    text = text.replace("\n", " ")
    lines = []
    while text:
        lines.append(indent + text[:width])
        text = text[width:]
    return "\n".join(lines)


def show_item(idx: int, total: int, row: dict, chunks: dict, reasons: list[str]) -> None:
    print("\n" + "=" * 84)
    print(f"[{idx}/{total}]  {row['query_id']}  |  난이도 {row['difficulty']}  |  {row['doc_type']}")
    print("=" * 84)

    print("\n  [질문]")
    print(wrap(row["question"]))

    print(f"\n  [출처] {row['doc_display_name'][:64]}")

    src = chunks.get(row["source_chunk_id"], {})
    if src:
        print("\n  [근거 청크]")
        print(wrap(src.get("text", "")[:520]))

    print(f"\n  [답 요약] {row.get('answer_hint', '')}")

    rank = row.get("found_rank", "0")
    rank_txt = "20위 밖" if rank == "0" else f"{rank}위"
    print(f"  [현재 검색 순위] {rank_txt}")

    if row.get("judged_added"):
        print("\n  [LLM이 추가한 정답]")
        for item in row["judged_added"].split(" | "):
            cid = item.split("(")[0]
            cand = chunks.get(cid, {})
            print(f"    - {item[:70]}")
            if cand:
                print(f"      출처: {cand.get('doc_display_name', '')[:56]}")
                print(f"      내용: {cand.get('text', '')[:110].replace(chr(10), ' ')}...")

    print("\n  [검수 필요 사유]")
    for r in reasons:
        print(f"    ! {r}")


# ------------------------------------------------------------------
# 요약
# ------------------------------------------------------------------


def print_summary(rows: list[dict]) -> None:
    kept = [r for r in rows if r.get("keep", "Y").upper() != "N"]
    dropped = [r for r in rows if r.get("keep", "Y").upper() == "N"]

    print("\n" + "=" * 70)
    print("평가셋 현황")
    print("=" * 70)
    print(f"  전체     : {len(rows)}건")
    print(f"  유지     : {len(kept)}건")
    print(f"  제외     : {len(dropped)}건")

    print("\n  [유지 항목 난이도별]")
    by_diff = Counter(r["difficulty"] for r in kept)
    for level in ("easy", "medium", "hard"):
        n = by_diff.get(level, 0)
        print(f"    {level:<8} {n:>3}건")

    print("\n  [유지 항목 문서유형별]")
    for dt, n in Counter(r["doc_type"] for r in kept).most_common():
        print(f"    {dt:<8} {n:>3}건")

    # 검수 후 예상 Recall@5
    print("\n  [예비 Recall@5] (e5-small 기준)")
    for level in ("easy", "medium", "hard"):
        subset = [r for r in kept if r["difficulty"] == level]
        if not subset:
            continue
        hits = 0
        for r in subset:
            top5 = r["top20_chunk_ids"].split(" | ")[:5]
            rel = set(r["relevant_chunk_ids"].split(" | "))
            if rel & set(top5):
                hits += 1
        print(f"    {level:<8} {hits}/{len(subset)} = {hits/len(subset):.2f}")

    total_hits = 0
    for r in kept:
        top5 = r["top20_chunk_ids"].split(" | ")[:5]
        rel = set(r["relevant_chunk_ids"].split(" | "))
        if rel & set(top5):
            total_hits += 1
    if kept:
        print(f"    {'전체':<8} {total_hits}/{len(kept)} = {total_hits/len(kept):.2f}")

    if dropped:
        print("\n  [제외된 항목]")
        for r in dropped[:15]:
            note = f" — {r['review_note']}" if r.get("review_note") else ""
            print(f"    {r['query_id']} {r['question'][:46]}{note}")
        if len(dropped) > 15:
            print(f"    ... 외 {len(dropped) - 15}건")
    print("=" * 70)


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="전체 항목 검수")
    ap.add_argument("--summary", action="store_true", help="현황만 출력")
    args = ap.parse_args()

    rows = load_rows()

    if args.summary:
        print_summary(rows)
        return

    chunks = load_chunks()

    # 검수 대상 선별
    targets: list[tuple[int, dict, list[str]]] = []
    for i, row in enumerate(rows):
        reasons = flag_reasons(row)
        if args.all or reasons:
            targets.append((i, row, reasons or ["전체 검수 모드"]))

    if not targets:
        print("검수가 필요한 항목이 없습니다.")
        print_summary(rows)
        return

    print("=" * 84)
    print("평가셋 검수")
    print("=" * 84)
    print(f"  전체 {len(rows)}건 중 {len(targets)}건이 검수 대상입니다.")
    print()
    print("  판단 기준")
    print("    유지(y) : 실제 직원이 물을 법하고, 근거 문서로 답할 수 있다")
    print("    제외(n) : 청크를 베꼈거나, 문서 없이도 답할 수 있거나, 질문이 어색하다")
    print()
    print("  입력키")
    print("    y = 유지   n = 제외   s = 건너뛰기(나중에)   q = 중단하고 저장")
    print("=" * 84)

    changed = 0
    for order, (idx, row, reasons) in enumerate(targets, 1):
        show_item(order, len(targets), row, chunks, reasons)

        while True:
            try:
                ans = input("\n  유지할까요? [y/n/s/q] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "q"

            if ans in ("y", "n", "s", "q", ""):
                break
            print("  y, n, s, q 중 하나를 입력하세요.")

        if ans == "q":
            print("\n  검수를 중단합니다. 지금까지의 결과를 저장합니다.")
            break

        if ans == "s":
            continue

        if ans == "n":
            rows[idx]["keep"] = "N"
            try:
                note = input("  제외 사유 (엔터로 생략) > ").strip()
            except (EOFError, KeyboardInterrupt):
                note = ""
            rows[idx]["review_note"] = note or "검수 중 제외"
            changed += 1
            print("  -> 제외 처리")
        else:
            rows[idx]["keep"] = "Y"
            print("  -> 유지")

    save_rows(rows)
    print(f"\n저장 완료 ({changed}건 변경) → {DRAFT_CSV.relative_to(PROJECT_ROOT)}")

    print_summary(rows)

    print("\n다음: python scripts/16_finalize_eval.py")


if __name__ == "__main__":
    main()
