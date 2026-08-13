"""
17_evaluate.py

목적
----
확정된 평가셋으로 검색 성능을 측정하고 리포트를 만든다.

측정 항목
--------
- Recall@1 / @3 / @5 / @10
- MRR, NDCG@10
- 난이도별 (easy / medium / hard)
- 문서유형별 (FAQ / 설명서 / 약관)
- 실패 케이스 목록

문서유형별로 나눠 보는 이유
------------------------
FAQ는 질문-답변 쌍이라 질의와 형태가 유사해 검색이 쉽다. 반면 약관은
"제7조 ① 채무자는..." 같은 문어체라 구어체 질의와 거리가 멀다.
전체 평균 하나만 보면 이 격차가 묻히고, 개선 노력을 어디에 쏟을지
판단할 수 없다.

출력
----
reports/eval_{model}_{date}.md         사람이 읽는 리포트
reports/eval_{model}_{date}.json       기계가 읽는 결과 (실험 비교용)
reports/failures_{model}_{date}.csv    실패 케이스 상세

사용법
-----
    python scripts/17_evaluate.py
    python scripts/17_evaluate.py --model bge-m3
    python scripts/17_evaluate.py --tag baseline    # 리포트에 이름표
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.embedding import MODELS, Embedder, FaissStore  # noqa: E402
from finguide_rag.evaluation.metrics import (  # noqa: E402
    QueryResult,
    aggregate,
    failure_analysis,
    group_by,
)

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
INDEX_ROOT = PROJECT_ROOT / "data" / "indexes" / "faiss"
REPORT_DIR = PROJECT_ROOT / "reports"

TOP_K = 20   # 실패 분석을 위해 20위까지 가져온다
KS = (1, 3, 5, 10)


# ------------------------------------------------------------------
# 실행
# ------------------------------------------------------------------


def load_eval_set() -> list[dict]:
    if not EVAL_CSV.exists():
        sys.exit(f"{EVAL_CSV} 없음. 먼저 16_finalize_eval.py 를 실행하세요.")
    with EVAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run_search(rows: list[dict], model_key: str) -> list[QueryResult]:
    index_dir = INDEX_ROOT / model_key
    if not (index_dir / "config.json").exists():
        sys.exit(
            f"{index_dir} 에 인덱스가 없습니다.\n"
            f"먼저 실행하세요: python scripts/11_build_index.py --model {model_key}"
        )

    store = FaissStore.load(index_dir)
    embedder = Embedder(model_key)
    print(f"  인덱스 {store.size:,}개 로드 ({model_key})")

    # 질의를 한 번에 인코딩하면 배치 처리로 훨씬 빠르다
    questions = [r["question"] for r in rows]
    print(f"  질의 {len(questions)}건 인코딩 중...")
    vectors = embedder.encode_queries(questions, show_progress=True)

    results: list[QueryResult] = []
    for row, vec in zip(rows, vectors):
        hits = store.search(vec, top_k=TOP_K)
        results.append(QueryResult(
            query_id=row["query_id"],
            question=row["question"],
            difficulty=row["difficulty"],
            doc_type=row["doc_type"],
            category=row["category"],
            relevant={c.strip() for c in row["relevant_chunk_ids"].split("|") if c.strip()},
            retrieved=[h.chunk_id for h in hits],
            scores=[h.score for h in hits],
        ))
    return results


# ------------------------------------------------------------------
# 리포트
# ------------------------------------------------------------------


def fmt_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def build_report(results: list[QueryResult], model_key: str, tag: str) -> str:
    overall = aggregate(results, KS)
    by_diff = group_by(results, "difficulty", KS)
    by_type = group_by(results, "doc_type", KS)
    by_cat = group_by(results, "category", KS)
    fails = failure_analysis(results)

    lines = [
        f"# 검색 성능 평가 — {tag or model_key}",
        "",
        f"- 실행일: {date.today().isoformat()}",
        f"- 모델: {MODELS[model_key].model_id} ({MODELS[model_key].dim}차원)",
        f"- 검색: dense 단독 (FAISS IndexFlatIP, 코사인 유사도)",
        f"- 평가셋: {len(results)}건",
        "",
        "## 전체 성능",
        "",
    ]
    lines += fmt_table(
        ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "NDCG@10"],
        [[f"{overall['recall@1']:.3f}", f"{overall['recall@3']:.3f}",
          f"{overall['recall@5']:.3f}", f"{overall['recall@10']:.3f}",
          f"{overall['mrr']:.3f}", f"{overall['ndcg@10']:.3f}"]],
    )

    # Recall 곡선의 기울기는 실패 유형을 시사한다.
    slope = overall["recall@10"] - overall["recall@1"]
    lines += [
        "",
        f"Recall@1에서 @10까지 {slope:+.3f} 상승했다. ",
    ]
    if slope > 0.25:
        lines.append("상승폭이 크므로 정답을 후보에는 넣지만 상위로 올리지 못하는 경우가 많다. "
                     "재순위화(reranking)의 효과가 클 것으로 보인다.")
    else:
        lines.append("상승폭이 작으므로 정답이 후보에 아예 들어오지 못하는 경우가 많다. "
                     "재순위화보다 후보 확보(sparse 검색 병행, 질의 확장)가 우선이다.")

    # --- 난이도별 ---
    lines += ["", "## 난이도별", ""]
    lines += fmt_table(
        ["난이도", "건수", "R@1", "R@5", "R@10", "MRR", "NDCG@10"],
        [[lvl, by_diff[lvl]["n"],
          f"{by_diff[lvl]['recall@1']:.3f}", f"{by_diff[lvl]['recall@5']:.3f}",
          f"{by_diff[lvl]['recall@10']:.3f}", f"{by_diff[lvl]['mrr']:.3f}",
          f"{by_diff[lvl]['ndcg@10']:.3f}"]
         for lvl in ("easy", "medium", "hard") if lvl in by_diff],
    )

    # --- 문서유형별 ---
    lines += [
        "", "## 문서 유형별", "",
        "FAQ는 질문-답변 쌍이라 질의와 형태가 유사하고, 약관은 문어체라 "
        "구어체 질의와 거리가 멀다. 이 격차가 개선 우선순위를 정한다.",
        "",
    ]
    lines += fmt_table(
        ["문서 유형", "건수", "R@1", "R@5", "R@10", "MRR", "NDCG@10"],
        [[dt, m["n"], f"{m['recall@1']:.3f}", f"{m['recall@5']:.3f}",
          f"{m['recall@10']:.3f}", f"{m['mrr']:.3f}", f"{m['ndcg@10']:.3f}"]
         for dt, m in sorted(by_type.items(), key=lambda x: -x[1]["recall@5"])],
    )

    # --- 카테고리별 (표본이 적은 것은 참고용) ---
    big_cats = {k: v for k, v in by_cat.items() if v["n"] >= 4}
    if big_cats:
        lines += ["", "## 카테고리별 (표본 4건 이상)", ""]
        lines += fmt_table(
            ["카테고리", "건수", "R@5", "MRR"],
            [[c, m["n"], f"{m['recall@5']:.3f}", f"{m['mrr']:.3f}"]
             for c, m in sorted(big_cats.items(), key=lambda x: -x[1]["recall@5"])],
        )

    # --- 실패 분석 ---
    total = len(results)
    lines += [
        "", "## 실패 분석", "",
        "정답 문서가 몇 위로 검색되는지에 따라 처방이 다르다.",
        "",
    ]
    lines += fmt_table(
        ["정답 순위", "건수", "비율", "처방"],
        [
            ["1~5위", fails["top5"], f"{fails['top5']/total:.0%}", "성공"],
            ["6~20위", fails["rank_6_20"], f"{fails['rank_6_20']/total:.0%}",
             "재순위화 (리랭커)"],
            ["20위 밖", fails["beyond_20"], f"{fails['beyond_20']/total:.0%}",
             "후보 확보 (BM25, 질의 확장)"],
        ],
    )

    # --- 점수 분포 (거절 로직 설계용) ---
    hits = [r for r in results if r.hit_rank in (1, 2, 3, 4, 5)]
    misses = [r for r in results if r.hit_rank == 0]
    if hits and misses:
        lines += [
            "", "## 점수 분포 (거절 로직 설계 참고)", "",
            "절대 점수로 거절 여부를 판단할 수 있는지 확인한다.",
            "",
        ]
        lines += fmt_table(
            ["구분", "건수", "top1 점수 평균", "1위-2위 격차 평균"],
            [
                ["성공 (5위 내)", len(hits),
                 f"{sum(r.top1_score for r in hits)/len(hits):.4f}",
                 f"{sum(r.score_gap for r in hits)/len(hits):.4f}"],
                ["실패 (미발견)", len(misses),
                 f"{sum(r.top1_score for r in misses)/len(misses):.4f}",
                 f"{sum(r.score_gap for r in misses)/len(misses):.4f}"],
            ],
        )
        gap = (sum(r.top1_score for r in hits) / len(hits)
               - sum(r.top1_score for r in misses) / len(misses))
        lines.append("")
        if abs(gap) < 0.02:
            lines.append(f"성공/실패의 top1 점수 차이가 {gap:+.4f}로 미미하다. "
                         "절대 점수 임계값만으로는 거절 판정이 어려우므로, "
                         "순위 간 격차나 재순위화 점수 같은 상대적 신호가 필요하다.")
        else:
            lines.append(f"성공/실패의 top1 점수 차이가 {gap:+.4f}다. "
                         "임계값 기반 거절 판정을 검토할 수 있다.")

    return "\n".join(lines) + "\n"


def save_failures(results: list[QueryResult], path: Path) -> None:
    """실패 케이스를 CSV로 남긴다.

    지표만 보면 무엇이 안 되는지 알 수 없다. 실제 질문과 검색 결과를
    나란히 봐야 개선 방향이 보인다.
    """
    rows = []
    for r in results:
        if r.hit_rank in (1, 2, 3, 4, 5):
            continue
        rows.append({
            "query_id": r.query_id,
            "question": r.question,
            "difficulty": r.difficulty,
            "doc_type": r.doc_type,
            "category": r.category,
            "hit_rank": r.hit_rank or "미발견",
            "failure_type": "20위 밖" if r.hit_rank == 0 else "순위 낮음",
            "relevant_chunk_ids": "|".join(sorted(r.relevant)),
            "top5_retrieved": "|".join(r.retrieved[:5]),
            "top1_score": round(r.top1_score, 4),
            "score_gap": round(r.score_gap, 4),
        })

    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small", choices=list(MODELS))
    ap.add_argument("--tag", default="", help="리포트 제목에 붙일 이름표")
    args = ap.parse_args()

    print("=" * 68)
    print("검색 성능 평가")
    print("=" * 68)

    rows = load_eval_set()
    print(f"  평가셋 {len(rows)}건")

    results = run_search(rows, args.model)

    overall = aggregate(results, KS)
    by_diff = group_by(results, "difficulty", KS)
    by_type = group_by(results, "doc_type", KS)
    fails = failure_analysis(results)

    # --- 터미널 요약 ---
    print("\n" + "=" * 68)
    print("결과")
    print("=" * 68)
    print(f"  Recall@1  {overall['recall@1']:.3f}")
    print(f"  Recall@3  {overall['recall@3']:.3f}")
    print(f"  Recall@5  {overall['recall@5']:.3f}")
    print(f"  Recall@10 {overall['recall@10']:.3f}")
    print(f"  MRR       {overall['mrr']:.3f}")
    print(f"  NDCG@10   {overall['ndcg@10']:.3f}")

    print("\n  [난이도별]  R@5    MRR")
    for lvl in ("easy", "medium", "hard"):
        if lvl in by_diff:
            m = by_diff[lvl]
            print(f"    {lvl:<8} {m['recall@5']:.3f}  {m['mrr']:.3f}  (n={m['n']})")

    print("\n  [문서유형별]  R@5    MRR")
    for dt, m in sorted(by_type.items(), key=lambda x: -x[1]["recall@5"]):
        print(f"    {dt:<10} {m['recall@5']:.3f}  {m['mrr']:.3f}  (n={m['n']})")

    print("\n  [실패 분석]")
    total = len(results)
    print(f"    1~5위    {fails['top5']:>3}건 ({fails['top5']/total:.0%})")
    print(f"    6~20위   {fails['rank_6_20']:>3}건 ({fails['rank_6_20']/total:.0%})  <- 리랭커 대상")
    print(f"    20위 밖  {fails['beyond_20']:>3}건 ({fails['beyond_20']/total:.0%})  <- BM25 대상")

    # --- 저장 ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    name = args.tag or args.model

    report_md = REPORT_DIR / f"eval_{name}_{stamp}.md"
    report_json = REPORT_DIR / f"eval_{name}_{stamp}.json"
    failures_csv = REPORT_DIR / f"failures_{name}_{stamp}.csv"

    report_md.write_text(build_report(results, args.model, args.tag), encoding="utf-8")

    report_json.write_text(json.dumps({
        "date": stamp,
        "tag": args.tag or args.model,
        "model": args.model,
        "retrieval": "dense",
        "n_queries": len(results),
        "overall": overall,
        "by_difficulty": by_diff,
        "by_doc_type": by_type,
        "by_category": group_by(results, "category", KS),
        "failure_analysis": fails,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    save_failures(results, failures_csv)

    print(f"\n저장 → {report_md.relative_to(PROJECT_ROOT)}")
    print(f"저장 → {report_json.relative_to(PROJECT_ROOT)}")
    if failures_csv.exists():
        print(f"저장 → {failures_csv.relative_to(PROJECT_ROOT)}")
    print("=" * 68)


if __name__ == "__main__":
    main()
