"""
16_finalize_eval.py

목적
----
검수를 마친 초안(retrieval_eval_draft.csv)에서 유지 항목만 추려
최종 평가셋을 확정한다.

초안과 최종본을 분리하는 이유
--------------------------
초안에는 생성·검수 과정의 흔적(겹침 비율, LLM 판정 사유, 검수 메모,
당시 검색 결과)이 들어 있다. 이것들은 평가셋의 신뢰도를 설명하는
근거이므로 보존해야 하지만, 평가 실행에는 불필요하다.

특히 top20_chunk_ids 는 e5-small 인덱스로 검색한 당시 결과인데,
이것이 최종본에 남아 있으면 나중에 다른 모델로 평가할 때 혼동을 준다.
평가는 실행 시점에 새로 검색해야 하므로 최종본에서는 제거한다.

출력
----
data/eval/retrieval_eval.csv   평가 실행용 (열 최소화)
data/eval/eval_manifest.json   평가셋 구성 내역 (README 작성 근거)

사용법
-----
    python scripts/16_finalize_eval.py
    python scripts/16_finalize_eval.py --check   # 저장하지 않고 검증만
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_draft.csv"
FINAL_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
MANIFEST = PROJECT_ROOT / "data" / "eval" / "eval_manifest.json"
REPORT_MD = PROJECT_ROOT / "reports" / "eval_set_summary.md"
CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"

# 최종 평가셋에 남길 열.
# 평가 실행에 필요한 것과, 결과를 난이도·유형별로 나눠 볼 때 필요한 것만 남긴다.
FINAL_COLUMNS = [
    "query_id",
    "question",
    "relevant_chunk_ids",   # 파이프(|)로 구분된 정답 청크 ID
    "difficulty",
    "doc_type",
    "category",
    "answer_hint",          # 결과를 눈으로 확인할 때 참고용
]


def load_draft() -> list[dict]:
    if not DRAFT_CSV.exists():
        sys.exit(f"{DRAFT_CSV} 없음. 먼저 14/15번 스크립트를 실행하세요.")
    with DRAFT_CSV.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_chunk_ids() -> set[str]:
    """현재 인덱스에 존재하는 청크 ID 집합."""
    if not CHUNKS_PATH.exists():
        return set()
    ids = set()
    for line in CHUNKS_PATH.open(encoding="utf-8"):
        if line.strip():
            ids.add(json.loads(line)["chunk_id"])
    return ids


def validate(rows: list[dict], valid_ids: set[str]) -> list[str]:
    """최종 평가셋의 무결성을 검사한다.

    평가셋에 존재하지 않는 청크 ID가 들어 있으면 그 항목은 영원히
    정답을 맞힐 수 없다. 청킹 설정을 바꾸면 청크 ID가 달라지므로
    실제로 발생할 수 있는 문제다.
    """
    errors: list[str] = []

    seen_ids: set[str] = set()
    for r in rows:
        qid = r["query_id"]

        if qid in seen_ids:
            errors.append(f"{qid}: query_id 중복")
        seen_ids.add(qid)

        if not r.get("question", "").strip():
            errors.append(f"{qid}: 질문이 비어 있음")

        rel = [c.strip() for c in r["relevant_chunk_ids"].split("|") if c.strip()]
        if not rel:
            errors.append(f"{qid}: 정답 청크가 없음")

        if valid_ids:
            missing = [c for c in rel if c not in valid_ids]
            if missing:
                errors.append(f"{qid}: 존재하지 않는 청크 {missing[:2]}")

        if len(set(rel)) != len(rel):
            errors.append(f"{qid}: 정답 청크 중복")

    return errors


def compute_baseline(kept: list[dict]) -> dict:
    """확정 시점의 검색 성능을 스냅샷으로 남긴다.

    초안에 기록된 top20_chunk_ids(e5-small 인덱스 기준)로 계산한다.
    정식 지표는 평가 모듈에서 실행 시점에 다시 산출하며, 여기 값은
    "평가셋을 확정할 때 성능이 이랬다"는 기록이다. 나중에 개선 실험을
    할 때 출발점을 확인하는 근거가 된다.
    """
    out: dict = {"model": "multilingual-e5-small", "by_difficulty": {}}

    total_hit = total_n = 0
    for level in ("easy", "medium", "hard"):
        subset = [r for r in kept if r["difficulty"] == level]
        if not subset:
            continue
        hit = 0
        for r in subset:
            top5 = [c.strip() for c in r.get("top20_chunk_ids", "").split("|")][:5]
            rel = {c.strip() for c in r["relevant_chunk_ids"].split("|") if c.strip()}
            if rel & set(top5):
                hit += 1
        out["by_difficulty"][level] = {
            "n": len(subset), "hit": hit, "recall_at_5": round(hit / len(subset), 4)
        }
        total_hit += hit
        total_n += len(subset)

    out["overall_recall_at_5"] = round(total_hit / total_n, 4) if total_n else 0.0

    # 실패 원인 분류. 개선 수단을 정하는 근거가 된다.
    low_rank = sum(1 for r in kept if r.get("found_rank", "0").isdigit()
                   and 5 < int(r["found_rank"]) <= 20)
    outside = sum(1 for r in kept if r.get("found_rank") == "0")
    out["failure_analysis"] = {
        "rank_6_to_20": low_rank,
        "beyond_20": outside,
        "note": "6~20위는 재순위화로 개선 여지가 있고, 20위 밖은 질의-문서 표현 격차 문제다.",
    }
    return out


def build_manifest(draft: list[dict], kept: list[dict]) -> dict:
    """평가셋 구성 내역을 기록한다.

    "이 평가셋을 어떻게 만들었는가"에 답하는 자료다. 지표만 제시하고
    평가셋의 출처와 한계를 밝히지 않으면 그 숫자를 신뢰할 근거가 없다.
    """
    dropped = [r for r in draft if r.get("keep", "Y").upper() == "N"]

    # 제외 사유를 유형별로 묶는다
    reasons: dict[str, int] = defaultdict(int)
    for r in dropped:
        note = r.get("review_note", "")
        if "베낌" in note or "재사용" in note or "그대로" in note:
            reasons["표현 베낌"] += 1
        elif "불명확" in note or "특정되지" in note or "일반적" in note:
            reasons["대상 불명확"] += 1
        elif "불일치" in note or "어긋" in note or "다른 개념" in note or "부정확" in note:
            reasons["질문-근거 불일치"] += 1
        else:
            reasons["기타"] += 1

    n_relevant = Counter()
    for r in kept:
        n = len([c for c in r["relevant_chunk_ids"].split("|") if c.strip()])
        n_relevant[n] += 1

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_generated": len(draft),
        "final_count": len(kept),
        "excluded_count": len(dropped),
        "generation": {
            "method": "LLM 생성 (gpt-4.1-mini) 후 사람 검수",
            "source": "청크에서 계층 표본 추출 (문서유형별 비중 고정)",
            "difficulty_levels": {
                "easy": "상품명 + 정식 용어 사용",
                "medium": "상품명 언급, 구어체 표현",
                "hard": "상품명·전문용어 배제, 상황만 서술",
            },
            "answer_labeling": (
                "생성 근거 청크를 정답으로 하되, 상위 후보 4개를 LLM으로 "
                "판정해 실제로 답이 있는 청크를 추가. 근거 청크가 20위 밖인 "
                "경우에는 판정을 생략(검색 실패를 성공으로 왜곡하지 않기 위함)"
            ),
        },
        "quality_control": {
            "automatic": [
                "어절 겹침률 계산 (도메인 용어 제외, 0.6 초과 시 표시)",
                "hard 난이도 용어 회피 준수 자동 검사",
                "3개 이상 문서에 반복되는 정형 문구 청크 사전 제외",
            ],
            "manual_review": "의심 항목 전수 검수",
            "exclusion_reasons": dict(reasons),
        },
        "composition": {
            "by_difficulty": dict(Counter(r["difficulty"] for r in kept)),
            "by_doc_type": dict(Counter(r["doc_type"] for r in kept)),
            "by_category": dict(Counter(r["category"] for r in kept)),
            "relevant_per_query": {str(k): v for k, v in sorted(n_relevant.items())},
        },
        "limitations": [
            "질문이 LLM으로 생성되어 실제 직원의 질의 분포와 다를 수 있음",
            "정답 라벨의 1차 판정을 LLM이 수행했으므로 누락 가능성이 있음",
            "정답이 여러 문서에 흩어진 경우 라벨이 완전하지 않을 수 있음",
            "스캔본 4건은 인덱싱에서 제외되어 평가 대상이 아님",
        ],
    }


def write_report(manifest: dict, kept: list[dict]) -> None:
    """README에 인용할 수 있는 요약 문서를 만든다.

    manifest는 기계가 읽는 형식이라 사람이 훑기 불편하다. 포트폴리오에서는
    "이 평가셋을 어떻게 만들었고 왜 신뢰할 수 있는가"를 설명해야 하므로
    읽기 좋은 형태로도 남긴다.
    """
    b = manifest["baseline_snapshot"]
    lines = [
        "# 검색 평가셋 구성 요약",
        "",
        f"작성일: {manifest['created_at'][:10]}",
        "",
        "## 구축 절차",
        "",
        "| 단계 | 내용 |",
        "|---|---|",
        "| 표본 추출 | 문서 유형별 비중 고정(설명서 50% / 약관 30% / FAQ 20%) 후 카테고리 균등 배분 |",
        "| 정형 문구 제외 | 3개 이상 문서에 반복되는 청크(면책 조항 등)를 자동 탐지해 제외 |",
        "| 질문 생성 | gpt-4.1-mini로 난이도 3단계 생성 |",
        "| 자동 검증 | 어절 겹침률, hard 난이도의 용어 회피 준수 여부 검사 |",
        "| 정답 확장 | 상위 후보를 LLM으로 판정. 근거가 20위 밖이면 판정 생략 |",
        "| 사람 검수 | 자동 검사에 걸린 항목을 직접 확인 |",
        "",
        f"생성 {manifest['total_generated']}건 → 확정 **{manifest['final_count']}건** "
        f"(제외 {manifest['excluded_count']}건)",
        "",
        "## 난이도 정의",
        "",
        "| 난이도 | 건수 | 정의 |",
        "|---|---|---|",
    ]

    defs = {
        "easy": "상품명 + 문서의 정식 용어 사용",
        "medium": "상품명은 언급하되 구어체 표현",
        "hard": "상품명·전문용어 배제, 상황만 서술",
    }
    counts = Counter(r["difficulty"] for r in kept)
    for level in ("easy", "medium", "hard"):
        lines.append(f"| {level} | {counts.get(level, 0)} | {defs[level]} |")

    lines += [
        "",
        "hard 난이도가 중요하다. 실제 창구에서는 고객이 정식 용어를 쓰지 않으므로,",
        "이 구간의 성능이 실사용 가능성을 좌우한다.",
        "",
        "## 베이스라인 (multilingual-e5-small, dense 검색 단독)",
        "",
        "| 난이도 | Recall@5 |",
        "|---|---|",
    ]
    for level in ("easy", "medium", "hard"):
        v = b["by_difficulty"].get(level)
        if v:
            lines.append(f"| {level} | {v['recall_at_5']:.2f} ({v['hit']}/{v['n']}) |")
    lines.append(f"| **전체** | **{b['overall_recall_at_5']:.2f}** |")

    fa = b["failure_analysis"]
    lines += [
        "",
        "## 실패 원인 분석",
        "",
        f"- 근거 문서가 6~20위: **{fa['rank_6_to_20']}건** — 재순위화(reranking)로 개선 여지",
        f"- 근거 문서가 20위 밖: **{fa['beyond_20']}건** — 질의와 문서의 표현 격차",
        "",
        "이 분포가 개선 실험의 우선순위를 정한다.",
        "",
        "## 제외 사유 분류",
        "",
        "| 유형 | 건수 |",
        "|---|---|",
    ]
    reasons = manifest.get("quality_control", {}).get("exclusion_reasons", {})
    for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {n} |")

    lines += [
        "",
        "## 한계",
        "",
        "- 질문이 LLM으로 생성되어 실제 직원의 질의 분포와 다를 수 있다.",
        "- 정답 라벨을 LLM이 1차 판정했으므로 누락된 정답이 있을 수 있다.",
        "- 하나은행 공개 문서만 대상이며, 행내 시스템의 실제 문서와는 다르다.",
        "",
    ]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="저장하지 않고 검증만 수행")
    args = ap.parse_args()

    print("=" * 68)
    print("평가셋 확정")
    print("=" * 68)

    draft = load_draft()
    kept = [r for r in draft if r.get("keep", "Y").upper() != "N"]

    print(f"  초안 {len(draft)}건 중 {len(kept)}건 유지")

    # 미검수 항목 경고
    # 검수 이력(reviewed 열)이 있는 경우에만 미검수 항목을 경고한다.
    # 초기 버전 검수 도구에는 이 열이 없어, 실제로 검수했더라도 기록이
    # 남지 않는다. 열이 없으면 판단할 수 없으므로 경고하지 않는다.
    has_review_log = "reviewed" in kept[0]
    if has_review_log:
        unreviewed = [
            r for r in kept
            if r.get("reviewed", "").upper() != "Y"
            and (float(r.get("overlap_ratio", 0) or 0) > 0.6
                 or r.get("hard_issue")
                 or r.get("found_rank") == "0")
        ]
        if unreviewed:
            print(f"  참고: 검수 기록이 없는 의심 항목 {len(unreviewed)}건이 포함됩니다.")
            print("        15_review_eval.py 로 추가 검수할 수 있습니다.")
    else:
        flagged = [
            r for r in kept
            if float(r.get("overlap_ratio", 0) or 0) > 0.6
            or r.get("hard_issue")
            or r.get("found_rank") == "0"
        ]
        print(f"  참고: 자동 검사에 걸렸던 항목 {len(flagged)}건이 유지되었습니다.")
        print("        검수 결과 문제없다고 판단된 것으로 간주합니다.")

    # --- 무결성 검증 ---
    valid_ids = load_chunk_ids()
    if valid_ids:
        print(f"  현재 청크 {len(valid_ids):,}개와 대조합니다.")

    errors = validate(kept, valid_ids)
    if errors:
        print(f"\n  [오류 {len(errors)}건]")
        for e in errors[:10]:
            print(f"    - {e}")
        if len(errors) > 10:
            print(f"    ... 외 {len(errors) - 10}건")
        sys.exit("\n오류를 수정한 뒤 다시 실행하세요.")

    print("  무결성 검증 통과")

    # --- 요약 ---
    print("\n  [구성]")
    for level in ("easy", "medium", "hard"):
        n = sum(1 for r in kept if r["difficulty"] == level)
        print(f"    {level:<8} {n:>3}건")

    print()
    for dt, n in Counter(r["doc_type"] for r in kept).most_common():
        print(f"    {dt:<8} {n:>3}건")

    total_rel = sum(
        len([c for c in r["relevant_chunk_ids"].split("|") if c.strip()])
        for r in kept
    )
    print(f"\n    정답 청크 총 {total_rel}개 (질의당 평균 {total_rel/len(kept):.1f}개)")

    if args.check:
        print("\n  --check 이므로 저장하지 않았습니다.")
        return

    # --- 저장 ---
    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FINAL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in kept:
            # 정답 청크 표기를 정규화한다. 검수 중 공백이 섞였을 수 있다.
            rel = [c.strip() for c in r["relevant_chunk_ids"].split("|") if c.strip()]
            row = dict(r)
            row["relevant_chunk_ids"] = "|".join(rel)
            w.writerow(row)

    manifest = build_manifest(draft, kept)
    manifest["baseline_snapshot"] = compute_baseline(kept)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n저장 → {FINAL_CSV.relative_to(PROJECT_ROOT)}")
    write_report(manifest, kept)
    print(f"저장 → {MANIFEST.relative_to(PROJECT_ROOT)}")
    print(f"저장 → {REPORT_MD.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 68)
    print("평가셋 확정 완료")
    print("=" * 68)
    print("  retrieval_eval.csv   평가 실행용")
    print("  eval_manifest.json   구성 내역 (README 작성 근거)")
    print()
    print("  초안(retrieval_eval_draft.csv)은 생성·검수 이력이므로 함께 보관합니다.")
    print("  두 파일 모두 git으로 관리해야 평가 재현이 가능합니다.")
    print()
    print("  다음: 평가 모듈 구현 (Recall@k, MRR, NDCG)")
    print("=" * 68)


if __name__ == "__main__":
    main()
