"""
37_review_mismatch.py

상품 불일치 20건을 눈으로 분류한다.

왜 필요한가
---------
96건 평가에서 상품불일치율 0.290 이 나왔다. 그런데 이 수치에는 성격이
다른 두 가지가 섞여 있다.

    진짜 실패    코퍼스에 없는 상품을 물었는데 유사 문서로 답했다.
                 또는 서로 다른 상품의 규정을 섞었다.
    평가기 한계  근거는 올바른데 판정이 틀렸다. 약관은 문서명에 상품명이
                 들어가지 않으므로 이름 대조가 성립하지 않는다.

문서유형별 분해에서 약관 근거의 불일치율이 0.562 로 유독 높다. 약관은
"입출금이 자유로운 예금약관" 처럼 상품군 이름만 담기 때문에 "저축예금
통장" 질문과 이름이 맞지 않는다. 실제로는 그 약관이 정확한 근거다.

두 가지를 섞어 하나의 수치로 보고하면 시스템이 실제보다 나빠 보이고,
개선 방향도 잘못 잡는다. 사람이 20건만 분류하면 정확한 값이 나온다.

분류 기준
--------
    A  코퍼스에 없는 상품    질문한 상품의 문서가 아예 없는데 답변함
    B  다른 상품 혼합        서로 다른 상품의 규정을 한 답변에 섞음
    C  평가기 한계 (약관)    근거는 올바르나 약관이라 이름이 안 맞음
    D  평가기 한계 (추출)    질문에서 뽑은 상품명 자체가 잘못됨

A 와 B 는 진짜 실패다. C 와 D 는 평가기가 잘못 잡은 것이므로 보정
지표에서 뺀다.

사용법
-----
    python scripts/37_review_mismatch.py            # 검수 파일 생성
    python scripts/37_review_mismatch.py --tally    # 채운 파일로 보정 지표 계산
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
REPORT_DIR = PROJECT_ROOT / "reports"
ANSWER_CACHE = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
REVIEW_MD = EVAL_DIR / "mismatch_review.md"

LABELS = {
    "A": "코퍼스에 없는 상품 — 진짜 실패",
    "B": "다른 상품 혼합 — 진짜 실패",
    "C": "평가기 한계 (약관이라 이름 불일치)",
    "D": "평가기 한계 (상품명 추출 오류)",
}
REAL = {"A", "B"}

ITEM = re.compile(r"<!--\s*MM\s*(\d+)\s*-->")
CLASS = re.compile(r"^-\s*분류:\s*([A-Da-d?])", re.M)
MEMO = re.compile(r"^-\s*메모:\s*(.*)$", re.M)


def latest(pattern: str) -> Path | None:
    files = sorted(REPORT_DIR.glob(pattern))
    return files[-1] if files else None


def load_rows() -> list[dict]:
    path = latest("g_eval_full_*.csv")
    if not path:
        sys.exit("reports/g_eval_full_*.csv 가 없습니다. 36번을 먼저 실행하세요.")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"  전체 평가 결과  {path.name} — {len(rows)}건")
    return rows


def build(rows: list[dict]) -> None:
    answers = json.loads(ANSWER_CACHE.read_text(encoding="utf-8")) \
        if ANSWER_CACHE.exists() else {}

    targets = [r for r in rows if r.get("상품일치") == "fail"]
    print(f"  상품 불일치  {len(targets)}건")

    L = ["# 상품 불일치 20건 분류", "",
         f"생성일: {date.today().isoformat()} / {len(targets)}건", "",
         "## 목적", "",
         "96건 평가에서 상품불일치율 0.290 이 나왔으나, 여기에는 진짜 실패와",
         "평가기 한계가 섞여 있다. 특히 약관 근거는 문서명에 상품명이 들어가지",
         "않아 이름 대조가 성립하지 않는다(약관 근거 불일치율 0.562).",
         "", "## 분류 기준", ""]
    for k, v in LABELS.items():
        L.append(f"- **{k}** — {v}")
    L += ["",
          "`- 분류: ?` 뒤를 A, B, C, D 중 하나로 바꿔 주세요.",
          "판단이 어려우면 `?` 로 두시면 집계에서 제외됩니다.",
          "", "판단 요령", "",
          "- 근거 문서가 질문에 답하기에 **적절한 문서인가**를 먼저 보세요.",
          "- 적절한데 이름만 안 맞으면 C 또는 D 입니다.",
          "- 그 상품 문서가 코퍼스에 아예 없으면 A 입니다.",
          "- 여러 상품 규정이 한 답변에 섞였으면 B 입니다.",
          "", "---", ""]

    for n, r in enumerate(targets, 1):
        q = r["question"]
        item = answers.get(q, {})
        L.append(f"## [{n:02d}] {q}")
        L.append("")
        L.append(f"`난이도 {r.get('difficulty', '?')} / "
                 f"{r.get('doc_type', '?')} / "
                 f"추출 상품: {r.get('extracted_product') or '(없음)'}`")
        L.append("")
        L.append(f"**평가기 사유**  {r.get('why_상품일치', '')}")
        L.append("")

        if item.get("answer"):
            L.append("**답변**")
            L.append("")
            for line in item["answer"].split("\n"):
                L.append(f"> {line}")
            L.append("")

        evs = item.get("evidences", [])
        if evs:
            L.append("**근거 문서**")
            L.append("")
            for e in evs:
                L.append(f"- [{e['rank']}] `{e.get('doc_type', '?')}` "
                         f"{e.get('citation', '')}")
            L.append("")
            L.append("<details><summary>근거 원문 (펼치기)</summary>")
            L.append("")
            for e in evs:
                L.append(f"**[{e['rank']}]**")
                L.append("")
                L.append("```")
                L.append(e.get("text", "")[:900])
                L.append("```")
                L.append("")
            L.append("</details>")
            L.append("")

        L.append(f"<!-- MM {n:02d} -->")
        L.append("- 분류: ?")
        L.append("- 메모: ")
        L.append("")
        L.append("---")
        L.append("")

    REVIEW_MD.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"\n저장 → {REVIEW_MD.relative_to(PROJECT_ROOT)}")
    print("  이 파일의 '- 분류: ?' 를 채운 뒤 --tally 로 다시 실행하세요.")


def tally(rows: list[dict]) -> None:
    if not REVIEW_MD.exists():
        sys.exit(f"{REVIEW_MD} 없음. 먼저 검수 파일을 생성하세요.")

    text = REVIEW_MD.read_text(encoding="utf-8")
    blocks = ITEM.split(text)
    labels: dict[str, str] = {}
    memos: dict[str, str] = {}
    for i in range(1, len(blocks), 2):
        iid = f"{int(blocks[i]):02d}"
        m = CLASS.search(blocks[i + 1])
        if m:
            labels[iid] = m.group(1).upper()
        mm = MEMO.search(blocks[i + 1])
        if mm:
            memos[iid] = mm.group(1).strip()

    targets = [r for r in rows if r.get("상품일치") == "fail"]
    done = {k: v for k, v in labels.items() if v in LABELS}
    print(f"  분류 완료  {len(done)}/{len(targets)}건")
    if len(done) < len(targets):
        undone = [k for k in labels if labels[k] not in LABELS]
        print(f"    미분류: {', '.join(sorted(undone))}")

    counts = Counter(done.values())
    print("\n" + "=" * 76)
    print("분류 결과")
    print("=" * 76)
    for k in "ABCD":
        c = counts.get(k, 0)
        tag = "진짜 실패" if k in REAL else "평가기 한계"
        print(f"  {k}  {LABELS[k]:<34} {c:>3}건  [{tag}]")

    real = sum(counts.get(k, 0) for k in REAL)
    limit = sum(counts.get(k, 0) for k in "CD")
    print(f"\n  진짜 실패 {real}건 / 평가기 한계 {limit}건")

    # 보정 지표
    pool = [r for r in rows if r.get("상품일치") in {"pass", "fail"}]
    n_pool = len(pool)
    raw = len(targets)
    print("\n" + "=" * 76)
    print("보정 지표")
    print("=" * 76)
    print(f"  {'구분':<20}{'건수':>8}{'비율':>10}")
    print("  " + "-" * 38)
    print(f"  {'원시 상품불일치율':<20}{raw:>4}/{n_pool:<3}{raw / n_pool:>10.3f}")
    if len(done) == len(targets):
        print(f"  {'보정 상품불일치율':<20}{real:>4}/{n_pool:<3}{real / n_pool:>10.3f}")
        print(f"\n  평가기 한계 {limit}건을 제외한 값이 실제 시스템 성능이다.")
    else:
        print(f"  보정값은 전건 분류 후 계산됩니다.")

    # 평가기 정밀도
    if done:
        prec = real / len(done)
        print(f"\n  평가기 정밀도(상품일치 fail 기준)  {real}/{len(done)} = {prec:.3f}")
        print(f"    fail 로 잡은 것 중 실제 실패의 비율. 낮으면 오탐이 많다는 뜻이다.")

    # 사유별 분해
    print("\n" + "=" * 76)
    print("평가기 사유별 분류 분포")
    print("=" * 76)
    by_reason: dict[str, Counter] = {}
    for n, r in enumerate(targets, 1):
        iid = f"{n:02d}"
        if iid not in done:
            continue
        why = r.get("why_상품일치", "")
        kind = ("문서 혼합" if "혼합" in why
                else "이름 불일치" if "일치하는 근거 문서 없음" in why
                else "기타")
        by_reason.setdefault(kind, Counter())[done[iid]] += 1
    print(f"  {'사유':<14}" + "".join(f"{k:>6}" for k in "ABCD") + f"{'정밀도':>10}")
    print("  " + "-" * 44)
    for kind, c in by_reason.items():
        tot = sum(c.values())
        r_ = sum(c.get(k, 0) for k in REAL)
        print(f"  {kind:<14}" + "".join(f"{c.get(k, 0):>6}" for k in "ABCD")
              + f"{r_ / tot if tot else 0:>10.3f}")
    print("\n  사유별 정밀도가 크게 다르면, 낮은 쪽 규칙만 손보면 된다.")

    # 저장
    out = REPORT_DIR / f"mismatch_review_{date.today().isoformat()}.csv"
    recs = []
    for n, r in enumerate(targets, 1):
        iid = f"{n:02d}"
        recs.append({
            "id": iid, "question": r["question"],
            "difficulty": r.get("difficulty", ""), "doc_type": r.get("doc_type", ""),
            "extracted_product": r.get("extracted_product", ""),
            "why": r.get("why_상품일치", ""),
            "class": done.get(iid, ""), "is_real": done.get(iid, "") in REAL,
            "memo": memos.get(iid, ""),
        })
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\n저장 → {out.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tally", action="store_true")
    args = ap.parse_args()

    print("=" * 76)
    print("상품 불일치 분류")
    print("=" * 76)
    rows = load_rows()

    if args.tally:
        tally(rows)
    else:
        build(rows)


if __name__ == "__main__":
    main()
