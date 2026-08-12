"""
13_audit_titles.py

목적
----
제목이 잘못 추출된 문서를 찾아내고, 본문에서 대체 후보를 제시해
수동 매핑 템플릿을 만든다.

왜 필요한가
---------
검색 결과에서 출처가 이렇게 표시되는 사례가 확인됐다.

    "제2조 실명거래 - ① 거래처는 실명으로 거래하여야 한다."
    "3-06-1030(6-1) (2024.10 개정)"
    "이 계약서는 법령 및 내부통제기준에 따른 절차를 거쳐 제공됩니다."

각각 실제로는 예금거래기본약관, 은행여신거래기본약관(가계용) 등이다.
표지 제목이 이미지 폰트로 되어 있거나, 문서관리번호가 더 큰 폰트로
인쇄된 경우에 발생한다.

이것이 문제인 이유는 두 가지다.
1. B2E 시스템에서 직원이 출처를 알아볼 수 없으면 근거로서 무용하다.
2. 제목은 indexable_text의 접두어로 들어가므로 검색 품질에도 악영향을 준다.

출력
----
data/registry/title_overrides.csv
    corrected_title 열이 비어 있는 템플릿. 직접 채운 뒤
    파서/청킹 단계에서 읽어 적용한다.

사용법
-----
    python scripts/13_audit_titles.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = PROJECT_ROOT / "data" / "interim" / "parsed" / "documents.jsonl"
OUT_CSV = PROJECT_ROOT / "data" / "registry" / "title_overrides.csv"


# ------------------------------------------------------------------
# 이상 제목 판정 규칙
# ------------------------------------------------------------------

# 각 규칙은 (사유, 판정함수) 쌍이다.
# 하나라도 걸리면 검토 대상으로 분류한다.

RE_ARTICLE_START = re.compile(r"^제\s*\d+\s*조")
RE_CIRCLED = re.compile(r"[\u2460-\u246f]")
RE_DOC_NUMBER = re.compile(r"^\d[\d\-()]{4,}")          # 3-06-1030(6-1)
RE_CID = re.compile(r"\(cid:")                            # 폰트 미해석
RE_SENTENCE_END = re.compile(r"(합니다|입니다|한다|됩니다|바랍니다)[.\s]*$")

# 문서 종류를 나타내는 키워드. 제목이라면 보통 이 중 하나를 포함한다.
DOC_KEYWORDS = [
    "상품설명서", "설명서", "약관", "약정서", "계약서",
    "동의서", "신청서", "확인서", "권유문서", "포트폴리오",
]


def diagnose(title: str, subtitle: str) -> list[str]:
    """제목의 문제를 진단해 사유 목록을 반환한다. 비어 있으면 정상."""
    reasons: list[str] = []
    t = title.strip()

    if not t:
        reasons.append("제목 없음")
        return reasons

    if RE_CID.search(t):
        reasons.append("폰트 미해석(cid)")
    if RE_ARTICLE_START.match(t):
        reasons.append("조항으로 시작")
    if RE_CIRCLED.search(t):
        reasons.append("항 기호 포함")
    if RE_DOC_NUMBER.match(t):
        reasons.append("문서관리번호")
    if RE_SENTENCE_END.search(t):
        reasons.append("서술형 문장")
    if len(t) > 45:
        reasons.append(f"과도하게 김({len(t)}자)")
    if not any(kw in t for kw in DOC_KEYWORDS):
        reasons.append("문서종류 키워드 없음")

    # 부제가 본문 문장을 잡은 경우도 표시한다
    s = subtitle.strip()
    if s and (RE_SENTENCE_END.search(s) or len(s) > 45):
        reasons.append("부제 이상")

    return reasons


# ------------------------------------------------------------------
# 대체 후보 추출
# ------------------------------------------------------------------


def suggest_titles(text: str, limit: int = 4) -> list[str]:
    """본문 앞부분에서 제목 후보를 뽑는다.

    문서 종류 키워드를 포함하면서 서술형이 아닌 짧은 줄을 찾는다.
    파서의 폰트 기반 추출과 달리 텍스트만 보므로, 폰트가 깨진
    문서에서도 후보를 찾을 수 있다.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    # 앞 1,200자 안에 제목이 있을 가능성이 높다
    for raw in text[:1200].splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not (4 <= len(line) <= 45):
            continue
        if RE_CID.search(line) or RE_CIRCLED.search(line):
            continue
        if "준법감시인" in line or "보존년한" in line:
            continue
        if not any(kw in line for kw in DOC_KEYWORDS):
            continue
        if RE_SENTENCE_END.search(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        candidates.append(line)
        if len(candidates) >= limit:
            break

    return candidates


def romanized_hint(filename: str) -> str:
    """파일명의 로마자 부분을 힌트로 남긴다.

    본문에서 후보를 못 찾았을 때 사람이 판단할 근거가 된다.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    # hana_{유형}_{카테고리}_{상품명...}_{날짜}
    body = [p for p in parts[2:] if not re.fullmatch(r"\d{8}", p)]
    return " ".join(body)[:70]


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------


def main() -> None:
    if not DOCS_PATH.exists():
        sys.exit(f"{DOCS_PATH} 없음. 먼저 06_build_documents.py 를 실행하세요.")

    rows = [json.loads(line) for line in DOCS_PATH.open(encoding="utf-8") if line.strip()]
    pdfs = [r for r in rows if r["doc_type"] != "FAQ" and r["is_parsable"]]

    print("=" * 72)
    print("제목 추출 감사")
    print("=" * 72)
    print(f"  대상 문서 {len(pdfs)}건 (FAQ 제외)\n")

    flagged: list[dict] = []

    for r in pdfs:
        reasons = diagnose(r.get("title", ""), r.get("subtitle", ""))
        # 신뢰도가 high가 아니면 사유가 없어도 검토 대상에 포함한다
        if not reasons and r.get("title_confidence") == "high":
            continue
        if not reasons:
            reasons = ["신뢰도 medium"]

        flagged.append({
            "row": r,
            "reasons": reasons,
            "candidates": suggest_titles(r.get("text", "")),
        })

    # 사유가 많은 순으로 정렬해 심각한 것부터 보여준다
    flagged.sort(key=lambda f: -len(f["reasons"]))

    print(f"  검토 필요 {len(flagged)}건\n")

    for i, f in enumerate(flagged, 1):
        r = f["row"]
        print("-" * 72)
        print(f"  [{i}] {r['doc_id']}  ({r['doc_type']}/{r['category']})")
        print(f"      파일   : {r['orig_filename'][:62]}")
        print(f"      현재제목: {r.get('title', '')[:62]}")
        if r.get("subtitle"):
            print(f"      현재부제: {r['subtitle'][:62]}")
        print(f"      사유   : {', '.join(f['reasons'])}")
        print(f"      파일명힌트: {romanized_hint(r['orig_filename'])}")

        if f["candidates"]:
            print("      본문 후보:")
            for j, c in enumerate(f["candidates"], 1):
                print(f"        {j}) {c}")
        else:
            print("      본문 후보: 없음 — 원본 PDF를 직접 확인해야 합니다")
        print()

    # --- 템플릿 저장 ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "doc_id", "orig_filename", "current_title", "current_subtitle",
            "reasons", "candidate_1", "candidate_2", "candidate_3",
            "corrected_title", "corrected_subtitle",
        ])
        for f in flagged:
            r = f["row"]
            cands = f["candidates"] + ["", "", ""]
            w.writerow([
                r["doc_id"],
                r["orig_filename"],
                r.get("title", ""),
                r.get("subtitle", ""),
                "; ".join(f["reasons"]),
                cands[0], cands[1], cands[2],
                "",  # 직접 채운다
                "",
            ])

    print("=" * 72)
    print("작성 방법")
    print("=" * 72)
    print(f"  1. {OUT_CSV.relative_to(PROJECT_ROOT)} 를 Excel로 연다")
    print("  2. corrected_title 열에 올바른 제목을 입력한다")
    print("     - candidate 열에 맞는 것이 있으면 그대로 복사")
    print("     - 없으면 원본 PDF를 열어 표지 제목을 확인")
    print("  3. 부제가 필요하면 corrected_subtitle 에도 입력한다")
    print("  4. 비워 두면 기존 제목을 유지한다")
    print()
    print("  원본 PDF 열기 예시:")
    if flagged:
        sample = flagged[0]["row"]
        sub = "terms" if sample["doc_type"] == "약관" else "desc"
        print(f'    start "data\\raw\\hana\\{sub}\\{sample["orig_filename"]}"')
    print("=" * 72)


if __name__ == "__main__":
    main()
