"""
수집된 하나은행 FAQ 데이터 검수 스크립트
==========================================
목적: 01_crawl_hana_faq.py 로 수집한 faq_hana.jsonl 을 자동 점검하여
      사람이 손봐야 할 항목만 리포트로 뽑아준다.

특징:
- 파이썬 표준 라이브러리만 사용 (requests/bs4 등 추가 설치 불필요).
- 검사 결과를 콘솔 요약 + 플래그된 항목 CSV(review_flags.csv)로 동시 출력.

실행:
    python 02_validate_faq.py
    (파일 경로를 직접 지정하려면)  python 02_validate_faq.py C:\\Users\\sky-56\\data\\raw\\hana\\faq\\faq_hana.jsonl
"""

import csv
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median

# ── 설정 ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "raw" / "hana" / "faq" / "faq_hana.jsonl"
FLAG_OUT = ROOT / "data" / "raw" / "hana" / "faq" / "review_flags.csv"

# 카테고리별 목표 개수 (크롤러와 동일)
EXPECTED_COUNTS = {
    "deposit_trust": 32, "loan": 32, "certificate": 22, "internet_banking": 20,
    "fx": 16, "auth_otp_security_card": 16, "login_signup": 16,
    "hana_oneq_mobile_banking": 16, "fund": 12, "cd_atm": 8,
    "bill_payment": 4, "overseas_branch_others": 4, "phone_banking": 2,
}

REQUIRED_FIELDS = ["faq_id", "bank_code", "bank_name", "category",
                   "question", "answer", "source_url", "collected_at", "status"]

MIN_Q_LEN = 8       # 질문 최소 길이(자)
MIN_A_LEN = 20      # 답변 최소 길이(자)
LONG_A_LEN = 1500   # 청킹 필요할 만큼 긴 답변 기준(자) — 오류 아님, 참고용
LABEL_RE = re.compile(r"^\s*(Q\.|A\.|질문|답변)")      # 라벨 찌꺼기
BROKEN_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")  # 깨진/제어문자
SIM_THRESHOLD = 0.85  # 유사 중복 질문 판정 임계값


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def load(path: Path):
    rows, errors = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append((i, str(e)))
    return rows, errors


def check(rows):
    """각 검사를 수행하고 (요약 dict, 플래그 리스트) 반환."""
    flags = []  # (faq_id, issue, detail)

    def flag(rid, issue, detail=""):
        flags.append((rid, issue, detail))

    # 1) 카테고리별 건수
    cat_counts = Counter(r.get("category", "?") for r in rows)

    # 2) faq_id 중복
    id_counts = Counter(r.get("faq_id", "") for r in rows)
    for rid, c in id_counts.items():
        if c > 1:
            flag(rid, "faq_id 중복", f"{c}회 등장")

    # 3) 필드/내용 단위 검사
    q_lens, a_lens = [], []
    for r in rows:
        rid = r.get("faq_id", "?")
        # 필수 필드 누락
        for field in REQUIRED_FIELDS:
            if field not in r or r.get(field) in (None, ""):
                flag(rid, "빈/누락 필드", field)
        q, a = r.get("question", ""), r.get("answer", "")
        q_lens.append(len(q))
        a_lens.append(len(a))
        # 길이 이상
        if len(q) < MIN_Q_LEN:
            flag(rid, "질문 너무 짧음", f"{len(q)}자: {q[:30]}")
        if len(a) < MIN_A_LEN:
            flag(rid, "답변 너무 짧음", f"{len(a)}자: {a[:40]}")
        if len(a) > LONG_A_LEN:
            flag(rid, "답변 김(청킹 대상)", f"{len(a)}자")
        # 라벨 찌꺼기
        if LABEL_RE.match(q) or LABEL_RE.match(a):
            flag(rid, "라벨 찌꺼기(Q./A.)", f"q={q[:15]} | a={a[:15]}")
        # 깨진/제어 문자
        if BROKEN_RE.search(q) or BROKEN_RE.search(a):
            flag(rid, "깨진/제어문자", "")
        # 카테고리 유효성
        if r.get("category") not in EXPECTED_COUNTS:
            flag(rid, "미정의 카테고리", str(r.get("category")))

    # 4) 유사 중복 질문 (완전 중복은 크롤러가 이미 제거했지만, 표현만 다른 near-dup 탐지)
    seen_pairs = []
    for i in range(len(rows)):
        qi = norm(rows[i].get("question", ""))
        for j in range(i + 1, len(rows)):
            qj = norm(rows[j].get("question", ""))
            if not qi or not qj:
                continue
            ratio = SequenceMatcher(None, qi, qj).ratio()
            if ratio >= SIM_THRESHOLD:
                rid_i, rid_j = rows[i].get("faq_id"), rows[j].get("faq_id")
                flag(rid_i, "유사 질문 중복 의심", f"{rid_j} (유사도 {ratio:.2f})")

    summary = {
        "total": len(rows),
        "cat_counts": cat_counts,
        "q_len": (min(q_lens), int(mean(q_lens)), int(median(q_lens)), max(q_lens)) if q_lens else (0, 0, 0, 0),
        "a_len": (min(a_lens), int(mean(a_lens)), int(median(a_lens)), max(a_lens)) if a_lens else (0, 0, 0, 0),
    }
    return summary, flags


def report(summary, flags, json_errors):
    print("=" * 60)
    print(f"총 수집 건수: {summary['total']}건")
    print("-" * 60)

    print("[카테고리별 건수 (실제 / 목표)]")
    for cat, target in EXPECTED_COUNTS.items():
        actual = summary["cat_counts"].get(cat, 0)
        mark = "OK" if actual == target else "!!"
        print(f"  {mark}  {cat:26s} {actual:>3} / {target}")
    # 목표에 없는 카테고리가 섞였는지
    extra = set(summary["cat_counts"]) - set(EXPECTED_COUNTS)
    for cat in extra:
        print(f"  !!  {cat:26s} {summary['cat_counts'][cat]:>3} / (정의안됨)")

    print("-" * 60)
    qn, an = summary["q_len"], summary["a_len"]
    print(f"질문 길이(자)  min {qn[0]} / 평균 {qn[1]} / 중앙 {qn[2]} / max {qn[3]}")
    print(f"답변 길이(자)  min {an[0]} / 평균 {an[1]} / 중앙 {an[2]} / max {an[3]}")

    print("-" * 60)
    if json_errors:
        print(f"[JSON 파싱 실패 라인] {len(json_errors)}건")
        for ln, msg in json_errors[:5]:
            print(f"  line {ln}: {msg}")

    if not flags:
        print("플래그된 항목 없음 — 데이터 품질 양호.")
    else:
        by_issue = Counter(issue for _, issue, _ in flags)
        print(f"[플래그 요약] 총 {len(flags)}건")
        for issue, c in by_issue.most_common():
            print(f"  - {issue}: {c}건")
        print(f"\n상세 목록은 CSV로 저장됨 → {FLAG_OUT}")
    print("=" * 60)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"[오류] 파일을 찾을 수 없음: {path}")
        print("경로를 인자로 직접 넘겨보세요:  python 02_validate_faq.py <파일경로>")
        return

    rows, json_errors = load(path)
    summary, flags = check(rows)
    report(summary, flags, json_errors)

    # 플래그 상세를 CSV로 저장
    if flags:
        with open(FLAG_OUT, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["faq_id", "issue", "detail"])
            w.writerows(flags)


if __name__ == "__main__":
    main()
