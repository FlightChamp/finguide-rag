"""56 — 자연 오류 셋 v0: 사람이 실제로 찾은 오류만 모은다.

주입 오류는 '우리가 상상한 오류'다. 실제 오류 분포를 대표한다는 보장이 없다.
그래서 사람이 근거와 대조해 '틀렸다'고 판정한 실제 답변을 따로 모아 같은 검증기로 돌린다.
두 셋의 탐지율 차이가 곧 '주입 셋이 현실을 얼마나 대표하는가'에 대한 증거다.

출처 (둘 다 사람 판정)
-------------------
    A. 정상 초안 검수(55번)에서 human_is_correct=N 인 초안
       — v1 자동 평가기가 환각·수치·상품 모두 pass 를 준 답변들이다.
         즉 '자동 평가를 통과했지만 사람이 틀렸다고 본' 오류이며, v2 가 잡아야 할 핵심 표적이다.
    B. v1 상품불일치 수동 재분류(data/eval/mismatch_review.md)의 B 분류
       — '다른 상품 규정이 답변 안에서 합성된' 진짜 실패. v1 답변 원문(문서체) 그대로 쓴다.

기대 코드 (gold)
--------------
claim 단위 정답은 아직 없다. 초안 단위로 '이 중 하나라도 나오면 탐지'로 본다.
A 의 기대 코드는 검수 메모를 근거로 매핑했고(아래 GOLD_A), 사람 확인 전이므로
gold_status=proposed 로 둔다. 메모와 맞지 않으면 CSV 의 expected_codes 를 고치고
gold_status 를 confirmed 로 바꾸면 된다(다음 실행 때 사람이 고친 값을 보존한다).

실행
----
    python scripts/56_build_natural_set.py

산출물
-----
    data/eval/v2/natural_errors_v0.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHEET = PROJECT_ROOT / "data" / "eval" / "v2" / "clean_drafts_v1.csv"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
MISMATCH_REVIEW = PROJECT_ROOT / "data" / "eval" / "mismatch_review.md"
OUT = PROJECT_ROOT / "data" / "eval" / "v2" / "natural_errors_v0.csv"

# 검수 메모 → 기대 사유 코드 (제안). 메모 요지를 함께 적어 매핑 근거를 남긴다.
GOLD_A = {
    "d010": ("MISSING_CONDITION|CONTRADICTED", "국민주택 요건을 1순위 전체 요건처럼 일반화 (민영주택 요건 누락)"),
    "d013": ("MISSING_CONDITION|CONTRADICTED|NO_EVIDENCE", "핵심 조건 표현이 불완전·모호"),
    "d027": ("OMITTED_REQUIRED_INFO|NO_EVIDENCE|MISSING_CONDITION", "질문 핵심(허위 신고 시 결과)에 직접 답하지 않음"),
    "d029": ("MISSING_CONDITION", "'은행 책임 없는 사유로' 조건 생략 → 의미 범위 확대"),
    "d030": ("CONTRADICTED|MISSING_CONDITION", "적용되지 않는 단서 추가, 다른 조항 내용 혼입"),
}
GOLD_B = ("PRODUCT_MISMATCH|CONTRADICTED", "다른 상품·서비스 규정이 한 답변에 합성됨 (v1 재분류 B)")

COLS = ["nat_id", "source", "base_id", "question", "doc_type", "text", "human_note",
        "expected_codes", "expected_min_verdict", "gold_note", "gold_status"]


def read_csv(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "cp949"):
        try:
            with path.open(encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    return []


def class_b_questions() -> list[str]:
    text = MISMATCH_REVIEW.read_text(encoding="utf-8") if MISMATCH_REVIEW.exists() else ""
    out, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^## \[\d+\]\s*(.+)$", line)
        if m:
            cur = m.group(1).strip()
        elif cur and re.match(r"^- 분류:\s*B", line.strip()):
            out.append(cur)
            cur = None
    return out


def main() -> int:
    sheet = read_csv(SHEET)
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    prev = {r["nat_id"]: r for r in read_csv(OUT)} if OUT.exists() else {}
    rows: list[dict] = []

    for r in sheet:
        if r.get("human_is_correct", "").strip().upper() != "N":
            continue
        codes, note = GOLD_A.get(r["draft_id"], ("MISSING_CONDITION|CONTRADICTED|NO_EVIDENCE",
                                                  "매핑 없음 — 메모 보고 확정 필요"))
        rows.append({"nat_id": f"A_{r['draft_id']}", "source": "A:clean_review_N",
                     "base_id": r["draft_id"], "question": r["question"], "doc_type": r["doc_type"],
                     "text": r["draft_text"], "human_note": re.sub(r"\s*\[review:packet\]", "", r["note"]),
                     "expected_codes": codes, "expected_min_verdict": "CHECK",
                     "gold_note": note, "gold_status": "proposed"})

    qs = class_b_questions()
    by_q = {k.strip(): v for k, v in answers.items()}
    for i, q in enumerate(qs, 1):
        a = by_q.get(q)
        if not a or a.get("decision") != "answer":
            continue
        rows.append({"nat_id": f"B_{i:02d}", "source": "B:v1_mismatch_B", "base_id": "",
                     "question": q, "doc_type": (a.get("evidences") or [{}])[0].get("doc_type", ""),
                     "text": a["answer"], "human_note": "v1 상품불일치 수동 재분류 B",
                     "expected_codes": GOLD_B[0], "expected_min_verdict": "CHECK",
                     "gold_note": GOLD_B[1], "gold_status": "confirmed"})

    for r in rows:   # 사람이 고친 gold 는 보존
        old = prev.get(r["nat_id"])
        if old and old.get("gold_status") == "confirmed":
            r["expected_codes"], r["gold_status"] = old["expected_codes"], "confirmed"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print("=" * 60)
    print("자연 오류 셋 v0")
    print("=" * 60)
    na = sum(r["source"].startswith("A") for r in rows)
    print(f"A. 정상 초안 검수 N  {na}건   (gold proposed {sum(r['gold_status'] == 'proposed' for r in rows)})")
    print(f"B. v1 상품 혼합      {len(rows) - na}건   (mismatch_review B 분류 {len(qs)}건 중 답변 있음)")
    for r in rows:
        print(f"  {r['nat_id']:<8} [{r['expected_codes']}] {r['gold_note'][:50]}")
    print(f"→ {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
