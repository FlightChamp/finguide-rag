"""49 — v2 오류 주입 평가의 '정상 초안' 풀을 만든다.

배경
----
v2 검증기 평가는 정답 답변에 통제된 오류를 주입해 탐지율을 잰다.
그 출발점인 '정상 초안'은 v1 생성 평가 96건의 답변 원문
(data/interim/g_full_answers.json)에서 고른다.

다만 96건의 라벨은 자동 평가기 결과다. 사람 검수가 끝난 것은
상품불일치 20건 재분류(data/eval/mismatch_review.md)뿐이다. 따라서 이
스크립트는 '후보'를 고를 뿐이고, 정상 여부의 최종 확정은 사람이 한다
(산출 CSV 의 human_* 열).

선정 기준
--------
    포함  decision=answer, 환각=pass, 수치정확∈{pass,na}
          상품일치∈{pass,na}  또는  fail 이지만 재분류가 C/D(평가기 한계)
    제외  환각 fail, 수치 fail, 상품일치 fail 중 재분류 A/B(진짜 실패), 거절 응답
    not_found>0 인 부분 답변도 소수 포함한다 — 정당하게 CHECK 인 claim 이 있어야
    Clean False Alarm 을 '전부 OK' 가정 없이 잴 수 있기 때문이다.

층화
----
문서 유형(설명서/FAQ/약관)별 목표 수를 두고, 부족하면 다른 유형에서 채운다.
시드 고정. 같은 입력이면 같은 30건이 나온다.

산출물
-----
    data/eval/v2/clean_pool_candidates.csv   적격 후보 전체 + 제외 사유
    data/eval/v2/clean_drafts_v0.csv         선정 30건 (사람 검수·문체 변환용 시트)
    data/eval/v2/clean_pool_manifest.json    시드, 입력 파일 해시, 분포

실행
----
    python scripts/49_build_clean_pool.py
    python scripts/49_build_clean_pool.py --n 30 --seed 2026 --partial 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
REPORT_DIR = PROJECT_ROOT / "reports"
RETRIEVAL_EVAL = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
MISMATCH_REVIEW = PROJECT_ROOT / "data" / "eval" / "mismatch_review.md"
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "v2"

# 문서 유형별 목표 비율. 약관이 적으면 다른 유형으로 채운다.
TARGET_SHARE = {"설명서": 0.45, "FAQ": 0.30, "약관": 0.25}


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def latest_g_eval() -> Path:
    files = sorted(REPORT_DIR.glob("g_eval_full_*.csv"))
    if not files:
        sys.exit("reports/g_eval_full_*.csv 가 없습니다. 36번을 먼저 실행하세요.")
    return files[-1]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_mismatch_review(path: Path) -> dict[str, str]:
    """'## [NN] 질문' 블록마다 '- 분류: X' 를 읽어 {질문: 분류}."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## \[\d+\]\s*(.+?)\s*$", line)
        if m:
            current = m.group(1)
            continue
        m = re.match(r"^- 분류:\s*([ABCD?])", line.strip())
        if m and current:
            out[current] = m.group(1)
            current = None
    return out


def doc_id_of(chunk_id: str) -> str:
    return chunk_id.rsplit("_c", 1)[0] if "_c" in chunk_id else chunk_id


def eligibility(row: dict, review: dict[str, str]) -> tuple[bool, str]:
    if row["decision"] != "answer":
        return False, "거절 응답"
    if row["환각"] != "pass":
        return False, f"환각={row['환각']}"
    if row["수치정확"] not in ("pass", "na"):
        return False, f"수치정확={row['수치정확']}"
    pm = row["상품일치"]
    if pm in ("pass", "na"):
        return True, ""
    cls = review.get(row["question"].strip())
    if cls in ("C", "D"):
        return True, f"상품일치 fail → 재분류 {cls}(평가기 한계)"
    return False, f"상품일치 fail, 재분류={cls or '없음'}"


def stratified_pick(cands: list[dict], n: int, n_partial: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    full = [c for c in cands if int(c["not_found"]) == 0]
    part = [c for c in cands if int(c["not_found"]) > 0]
    rng.shuffle(full)
    rng.shuffle(part)

    picked = part[:n_partial]
    remaining = n - len(picked)
    by_type: dict[str, list[dict]] = {}
    for c in full:
        by_type.setdefault(c["doc_type"], []).append(c)

    quota = {t: round(remaining * s) for t, s in TARGET_SHARE.items()}
    # 반올림 오차 보정
    diff = remaining - sum(quota.values())
    quota["설명서"] += diff

    leftovers: list[dict] = []
    for t, q in quota.items():
        pool = by_type.get(t, [])
        picked += pool[:q]
        leftovers += pool[q:]
    for t, pool in by_type.items():
        if t not in quota:
            leftovers += pool
    rng.shuffle(leftovers)
    picked += leftovers[: n - len(picked)]
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--partial", type=int, default=4, help="not_found>0 부분 답변 포함 수")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if not ANSWERS.exists():
        sys.exit(f"{ANSWERS.relative_to(PROJECT_ROOT)} 가 없습니다.")
    g_path = latest_g_eval()
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    g_rows = read_csv(g_path)
    review = parse_mismatch_review(MISMATCH_REVIEW)
    qid = {r["question"].strip(): r for r in read_csv(RETRIEVAL_EVAL)} if RETRIEVAL_EVAL.exists() else {}

    all_rows: list[dict] = []
    for r in g_rows:
        q = r["question"].strip()
        ans = answers.get(r["question"]) or answers.get(q)
        ok, why = eligibility(r, review)
        if ok and not ans:
            ok, why = False, "답변 원문 캐시에 없음"
        meta = qid.get(q, {})
        evid = (ans or {}).get("evidences", [])
        all_rows.append({
            "query_id": meta.get("query_id", ""),
            "question": q,
            "doc_type": r["doc_type"],
            "difficulty": r["difficulty"],
            "not_found": r["not_found"],
            "eligible": ok,
            "note": why,
            "answer_hint": meta.get("answer_hint", ""),
            "v1_answer": (ans or {}).get("answer", ""),
            "v1_not_found": json.dumps((ans or {}).get("not_found", []), ensure_ascii=False),
            "evidence_doc_ids": "|".join(dict.fromkeys(doc_id_of(e["chunk_id"]) for e in evid)),
            "evidence_chunk_ids": "|".join(e["chunk_id"] for e in evid),
        })

    cands = [r for r in all_rows if r["eligible"]]
    picked = stratified_pick(cands, args.n, args.partial, args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand_path = OUT_DIR / "clean_pool_candidates.csv"
    with cand_path.open("w", encoding="utf-8-sig", newline="") as f:
        cols = ["query_id", "question", "doc_type", "difficulty", "not_found", "eligible", "note"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    sheet_path = OUT_DIR / "clean_drafts_v0.csv"
    sheet_cols = [
        "draft_id", "query_id", "question", "doc_type", "difficulty", "not_found",
        "v1_answer", "v1_not_found", "evidence_doc_ids", "evidence_chunk_ids", "answer_hint",
        # ↓ 사람이 채우는 열
        "human_is_correct",      # Y / N  — 근거 대비 정답인가 (N 이면 풀에서 제외)
        "draft_text",            # 직원 구어체로 변환한 초안 (의미 변경 금지)
        "human_style_ok",        # Y / N  — 변환 후 의미 보존 확인
        "note",
    ]
    picked_sorted = sorted(picked, key=lambda r: (r["doc_type"], r["query_id"]))
    with sheet_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sheet_cols, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(picked_sorted, 1):
            row = dict(r)
            row.update({"draft_id": f"d{i:03d}", "human_is_correct": "",
                        "draft_text": "", "human_style_ok": "", "note": ""})
            w.writerow(row)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed, "n": args.n, "partial": args.partial,
        "inputs": {
            str(ANSWERS.relative_to(PROJECT_ROOT)): sha8(ANSWERS),
            str(g_path.relative_to(PROJECT_ROOT)): sha8(g_path),
            str(MISMATCH_REVIEW.relative_to(PROJECT_ROOT)): sha8(MISMATCH_REVIEW) if MISMATCH_REVIEW.exists() else None,
        },
        "eligible": len(cands),
        "picked_doc_type": dict(Counter(r["doc_type"] for r in picked)),
        "picked_difficulty": dict(Counter(r["difficulty"] for r in picked)),
        "picked_partial": sum(int(r["not_found"]) > 0 for r in picked),
    }
    (OUT_DIR / "clean_pool_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 콘솔 요약 ----------
    excl = Counter(r["note"].split("=")[0] if r["note"] else "" for r in all_rows if not r["eligible"])
    print("=" * 60)
    print("v2 정상 초안 풀")
    print("=" * 60)
    print(f"입력 {len(all_rows)}건  →  적격 {len(cands)}건  →  선정 {len(picked)}건")
    print(f"  (상품불일치 재분류 반영 {len(review)}건, 평가 파일 {g_path.name})")
    print("\n[제외 사유]")
    for k, v in excl.most_common():
        print(f"  {k or '(사유 없음)':<28} {v:>3}")
    print("\n[적격 후보 분포]")
    print(f"  문서유형  {dict(Counter(r['doc_type'] for r in cands))}")
    print(f"  난이도    {dict(Counter(r['difficulty'] for r in cands))}")
    print("\n[선정 30건 분포]")
    print(f"  문서유형  {manifest['picked_doc_type']}")
    print(f"  난이도    {manifest['picked_difficulty']}")
    print(f"  부분답변  {manifest['picked_partial']}건 (not_found>0)")
    print(f"\n저장 → {sheet_path.relative_to(PROJECT_ROOT)}")
    print(f"       {cand_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
