"""61 — 개발 셋 N 20건(+자연 오류 B 3건)에 사유 분류를 붙인다 (제안 → 사람 확인).

사람이 남긴 검수 메모를 근거로 Claude 가 사유 분류를 '제안'한 것이다 (status=proposed).
개발 셋이므로 공식 수치에는 쓰지 않고, 59번 개발 평가에서 사유별 탐지를 보는 데만 쓴다.
메모와 맞지 않는 분류가 있으면 CSV 의 reasons 를 고치고 status 를 confirmed 로 바꾸면
다음 실행 때 보존된다.

분류 체계는 60번(held-out v1)과 같다:
    WRONG · UNSUPPORTED · MISSING_CONDITION · OVERSTATED · OMITTED ·
    MIXED_SCOPE · SCOPE_MISMATCH · OUTDATED · OTHER

실행
----
    python scripts/61_dev_reason_labels.py
산출물
-----
    data/eval/v2/dev_reason_labels_v0.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "eval" / "v2" / "dev_reason_labels_v0.csv"

# (id, 출처, 사유들, 근거가 된 메모 요지)
PROPOSED = [
    ("d010", "정상초안30", "MISSING_CONDITION|SCOPE_MISMATCH", "국민주택 요건을 1순위 전체 요건처럼 일반화, 민영주택 요건 누락"),
    ("d013", "정상초안30", "UNSUPPORTED", "핵심 조건 표현이 불완전·모호해 근거와 같은 뜻인지 확인 불가"),
    ("d027", "정상초안30", "OMITTED", "질문 핵심(허위 신고 시 결과)에 직접 답하지 않음"),
    ("d029", "정상초안30", "MISSING_CONDITION", "'은행 책임 없는 사유로' 조건 생략 → 범위 확대"),
    ("d030", "정상초안30", "WRONG|MIXED_SCOPE", "적용되지 않는 단서 추가, 다른 조항 내용 혼입"),
    ("q012", "개발43", "UNSUPPORTED", "CD수익률 산출 주체 특정 — 근거에서 확인 불가"),
    ("q014", "개발43", "MIXED_SCOPE", "디폴트옵션 적용 흐름과 구성상품 매매 제한·옵트아웃을 하나로 묶음"),
    ("q015", "개발43", "UNSUPPORTED", "포트폴리오 전체 원금보장 표현 불명확, 구성상품별 구분 필요"),
    ("q018", "개발43", "UNSUPPORTED", "'한국금융기관 등' 기관명 근거 부족, 근거 이상 구체화"),
    ("q021", "개발43", "WRONG", "질문은 환율 하락 위험인데 상승 기준으로 방향이 바뀜"),
    ("q025", "개발43", "MIXED_SCOPE", "가입 서류와 비과세 신청 서류를 합쳐 안내"),
    ("q043", "개발43", "MIXED_SCOPE", "국가간 ATM 서비스(EXK)와 앱 기반 해외 ATM 출금 서비스를 하나로 설명"),
    ("q046", "개발43", "MIXED_SCOPE", "일반 약관 변경 통지와 긴급 변경 절차를 같은 규칙처럼 처리"),
    ("q052", "개발43", "OVERSTATED", "'제한될 수 있습니다'를 확정 표현으로 바꿈"),
    ("q055", "개발43", "SCOPE_MISMATCH", "일반 외화송금 질문에 easy-one 서비스 규정으로 답함"),
    ("q059", "개발43", "WRONG", "중도상환수수료 공식이 아닌 산식을 제시"),
    ("q064", "개발43", "OUTDATED", "환율에 따라 변하는 원화 환산값을 고정 한도로 제시"),
    ("q068", "개발43", "MIXED_SCOPE", "저축예금 약관과 다른 상품설명서 내용을 합쳐 일반 규정처럼 제시"),
    ("q086", "개발43", "WRONG|OVERSTATED", "실수요자 요건의 한 요소를 결격 사유로 확대"),
    ("q109", "개발43", "UNSUPPORTED", "위험등급·손실 수치를 근거로 확인할 수 없음"),
    ("B_01", "자연8", "MIXED_SCOPE|SCOPE_MISMATCH", "v1 재분류 B — 일반 예금거래 질문에 특정 통장 규정 혼합"),
    ("B_02", "자연8", "MIXED_SCOPE", "v1 재분류 B — 다른 상품 규정 합성"),
    ("B_03", "자연8", "MIXED_SCOPE", "v1 재분류 B — 기업대출·보금자리론 규정 합성"),
]


def main() -> int:
    prev = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8-sig", newline="") as f:
            prev = {r["id"]: r for r in csv.DictReader(f)}
    rows = []
    for i, src, reasons, basis in PROPOSED:
        old = prev.get(i)
        if old and old.get("status") == "confirmed":
            rows.append(old)
        else:
            rows.append({"id": i, "source": src, "reasons": reasons, "basis": basis, "status": "proposed"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "reasons", "basis", "status"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter(x for r in rows for x in r["reasons"].split("|"))
    print(f"개발 셋 사유 분류 {len(rows)}건 → {OUT.relative_to(PROJECT_ROOT)}")
    print("  " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    print(f"  confirmed {sum(r['status'] == 'confirmed' for r in rows)} / proposed {sum(r['status'] == 'proposed' for r in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
