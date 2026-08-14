"""
20_tune_hybrid.py

목적
----
하이브리드 검색의 결합 방식과 가중치를 실험으로 정한다.

무엇을 비교하는가
---------------
- dense 단독 (베이스라인)
- sparse 단독 (BM25만)
- 가중합 alpha = 0.3 / 0.5 / 0.7
- RRF

alpha는 dense 가중치다. 낮을수록 어휘 매칭(BM25) 비중이 커진다.

왜 실험이 필요한가
----------------
최적값은 데이터가 정한다. 소규모 시험에서 정답 청크가 alpha 0.3일 때
1위, 0.7일 때 3위로 나온 사례가 있었다. 임의로 0.5를 택하면 성능을
놓친다.

또한 문서 유형별로 최적값이 다를 수 있다. 약관은 어휘 매칭이 유리하고
FAQ는 의미 유사도가 유리할 가능성이 있으므로, 유형별 결과도 함께 본다.

출력
----
reports/hybrid_tuning_{date}.md     비교표와 해석
reports/hybrid_tuning_{date}.json   기계용 결과

사용법
-----
    python scripts/20_tune_hybrid.py
    python scripts/20_tune_hybrid.py --alphas 0.2 0.4 0.6 0.8
    python scripts/20_tune_hybrid.py --quick    # alpha 0.5 와 RRF 만
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.embedding import MODELS, Embedder, FaissStore  # noqa: E402
from finguide_rag.evaluation import (  # noqa: E402
    QueryResult,
    aggregate,
    failure_analysis,
    group_by,
)
from finguide_rag.retrieval import (  # noqa: E402
    BM25Store,
    FusionMethod,
    HybridRetriever,
    NormalizeMethod,
)

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
FAISS_ROOT = PROJECT_ROOT / "data" / "indexes" / "faiss"
BM25_ROOT = PROJECT_ROOT / "data" / "indexes" / "bm25"
REPORT_DIR = PROJECT_ROOT / "reports"

TOP_K = 20
KS = (1, 3, 5, 10)


# ------------------------------------------------------------------


def load_eval_set(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} 없음. 먼저 16_finalize_eval.py 를 실행하세요.")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_doc_id(chunk_id: str) -> str:
    """청크 ID 에서 문서 ID 를 뽑는다. {doc_id}_c{번호} 규칙."""
    return chunk_id.rsplit("_c", 1)[0] if "_c" in chunk_id else chunk_id


def evaluate_config(
    rows: list[dict],
    retriever: HybridRetriever,
    label: str,
    level: str = "chunk",
) -> tuple[dict, list[QueryResult]]:
    """설정 하나로 전체 평가셋을 검색하고 지표를 계산한다."""
    results: list[QueryResult] = []

    t0 = time.perf_counter()
    for row in rows:
        hits = retriever.search(row["question"], top_k=TOP_K)
        relevant = {c.strip() for c in row["relevant_chunk_ids"].split("|") if c.strip()}
        retrieved = [h.chunk_id for h in hits]
        scores = [h.score for h in hits]

        if level == "doc":
            # 청킹 전략이 달라도 문서 ID 는 불변이므로 공정 비교가 가능하다
            relevant = {to_doc_id(c) for c in relevant}
            seen: set[str] = set()
            dd, ds = [], []
            for cid, sc in zip(retrieved, scores):
                did = to_doc_id(cid)
                if did not in seen:
                    seen.add(did)
                    dd.append(did)
                    ds.append(sc)
            retrieved, scores = dd, ds

        results.append(QueryResult(
            query_id=row["query_id"],
            question=row["question"],
            difficulty=row["difficulty"],
            doc_type=row["doc_type"],
            category=row["category"],
            relevant=relevant,
            retrieved=retrieved,
            scores=scores,
        ))
    elapsed = time.perf_counter() - t0

    metrics = aggregate(results, KS)
    metrics["latency_ms"] = round(elapsed / len(rows) * 1000, 1)

    return metrics, results


def build_configs(args) -> list[tuple[str, FusionMethod, float, NormalizeMethod]]:
    """실험할 설정 목록을 만든다.

    가중합은 정규화 방식에 민감하다. 초기 실험에서 min-max 정규화가
    dense 점수 분포를 훼손해 베이스라인보다 나쁜 결과가 나왔으므로,
    수정된 min-max 와 z-score 를 모두 시험한다.

    RRF 도 가중치를 줄 수 있다. 순위 역수에 alpha 를 곱하는 방식이라
    스케일 문제 없이 검색기 비중을 조절할 수 있다.
    """
    mm, zs = NormalizeMethod.MINMAX, NormalizeMethod.ZSCORE

    configs: list[tuple[str, FusionMethod, float, NormalizeMethod]] = [
        ("dense 단독", FusionMethod.DENSE_ONLY, 1.0, mm),
        ("sparse 단독", FusionMethod.SPARSE_ONLY, 0.0, mm),
    ]

    if args.quick:
        configs += [
            ("weighted α=0.5", FusionMethod.WEIGHTED, 0.5, mm),
            ("RRF α=0.5", FusionMethod.RRF, 0.5, mm),
        ]
        return configs

    for a in args.alphas:
        configs.append((f"weighted α={a}", FusionMethod.WEIGHTED, a, mm))

    if not args.no_zscore:
        for a in args.alphas:
            configs.append((f"weighted-z α={a}", FusionMethod.WEIGHTED, a, zs))

    for a in args.rrf_alphas:
        configs.append((f"RRF α={a}", FusionMethod.RRF, a, mm))

    return configs


# ------------------------------------------------------------------
# 리포트
# ------------------------------------------------------------------


def fmt_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def build_report(all_results: dict, model_key: str) -> str:
    baseline = all_results["dense 단독"]["overall"]
    best_label = max(
        all_results,
        key=lambda k: all_results[k]["overall"]["recall@5"],
    )
    best = all_results[best_label]["overall"]

    lines = [
        "# 하이브리드 검색 파라미터 실험",
        "",
        f"- 실행일: {date.today().isoformat()}",
        f"- 임베딩: {MODELS[model_key].model_id}",
        f"- 평가셋: {all_results['dense 단독']['overall']['n']}건",
        "",
        "## 배경",
        "",
        "베이스라인(dense 단독) 실패 분석에서, 실패 23건 중 20건(87%)이 정답 청크에",
        "질의 어휘를 25% 이상 포함하고 있었다. 어휘 기반 검색으로 잡을 여지가 크다고",
        "판단해 BM25를 결합했다.",
        "",
        "특히 약관은 실패 10건 중 9건이 20위 밖이었다. 순위가 낮은 것이 아니라",
        "후보에 아예 들지 못한 것이므로, 재순위화가 아니라 후보 확보가 필요했다.",
        "",
        "## 전체 비교",
        "",
    ]

    rows = []
    for label, data in all_results.items():
        m = data["overall"]
        delta = m["recall@5"] - baseline["recall@5"]
        mark = "**" if label == best_label else ""
        rows.append([
            f"{mark}{label}{mark}",
            f"{m['recall@1']:.3f}",
            f"{m['recall@5']:.3f}",
            f"{m['recall@10']:.3f}",
            f"{m['mrr']:.3f}",
            f"{m['ndcg@10']:.3f}",
            f"{delta:+.3f}" if label != "dense 단독" else "기준",
            f"{m['latency_ms']:.0f}ms",
        ])
    lines += fmt_table(
        ["설정", "R@1", "R@5", "R@10", "MRR", "NDCG@10", "R@5 변화", "지연"],
        rows,
    )

    gain = best["recall@5"] - baseline["recall@5"]
    lines += [
        "",
        f"최고 성능은 **{best_label}** 로 Recall@5 {best['recall@5']:.3f}",
        f"(베이스라인 대비 {gain:+.3f}, {gain / baseline['recall@5'] * 100:+.1f}%).",
        "",
        "## 문서 유형별",
        "",
        "유형에 따라 최적 결합 비율이 다를 수 있다. 약관은 문어체라 어휘 매칭이",
        "유리하고, FAQ는 질의와 형태가 비슷해 의미 유사도만으로도 잘 찾는다.",
        "",
    ]

    doc_types = sorted(all_results["dense 단독"]["by_doc_type"].keys())
    header = ["설정"] + [f"{dt} R@5" for dt in doc_types]
    rows = []
    for label, data in all_results.items():
        row = [label]
        for dt in doc_types:
            m = data["by_doc_type"].get(dt, {})
            row.append(f"{m.get('recall@5', 0):.3f}")
        rows.append(row)
    lines += fmt_table(header, rows)

    # 유형별 최적 설정
    lines += ["", "유형별 최고 성능 설정:", ""]
    for dt in doc_types:
        best_dt = max(
            all_results,
            key=lambda k: all_results[k]["by_doc_type"].get(dt, {}).get("recall@5", 0),
        )
        v = all_results[best_dt]["by_doc_type"][dt]["recall@5"]
        base_v = all_results["dense 단독"]["by_doc_type"][dt]["recall@5"]
        lines.append(f"- **{dt}**: {best_dt} ({v:.3f}, 베이스라인 {base_v:.3f} 대비 {v - base_v:+.3f})")

    # --- 난이도별 ---
    lines += ["", "## 난이도별", ""]
    levels = ["easy", "medium", "hard"]
    header = ["설정"] + [f"{lv} R@5" for lv in levels]
    rows = []
    for label, data in all_results.items():
        row = [label]
        for lv in levels:
            m = data["by_difficulty"].get(lv, {})
            row.append(f"{m.get('recall@5', 0):.3f}")
        rows.append(row)
    lines += fmt_table(header, rows)

    # --- 실패 분포 변화 ---
    lines += [
        "", "## 실패 분포 변화", "",
        "20위 밖 건수가 줄었다면 후보 확보가 개선된 것이고,",
        "6~20위가 줄고 상위로 이동했다면 순위 결정이 개선된 것이다.",
        "",
    ]
    rows = []
    for label, data in all_results.items():
        f = data["failure_analysis"]
        rows.append([label, f["top5"], f["rank_6_20"], f["beyond_20"]])
    lines += fmt_table(["설정", "1~5위", "6~20위", "20위 밖"], rows)

    # --- 권고 ---
    lines += ["", "## 결론", ""]
    if gain > 0.03:
        lines.append(f"하이브리드 결합이 유효하다. {best_label} 설정을 채택한다.")
    elif gain > 0:
        lines.append(f"개선폭이 {gain:+.3f}로 작다. 결합 자체보다 다른 요인"
                     "(청킹 전략, 임베딩 모델)의 영향이 클 수 있다.")
    else:
        lines.append("하이브리드가 베이스라인을 넘지 못했다. BM25 토큰화나 "
                     "정규화 방식을 재검토해야 한다.")

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small", choices=list(MODELS))
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                    help="가중합에서 시험할 dense 가중치")
    ap.add_argument("--rrf-alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                    help="RRF에서 시험할 dense 가중치")
    ap.add_argument("--no-zscore", action="store_true", help="z-score 정규화 실험 생략")
    ap.add_argument("--quick", action="store_true", help="설정을 최소로 줄여 빠르게 확인")
    ap.add_argument("--candidate-k", type=int, default=100,
                    help="각 검색기에서 가져올 후보 수")
    ap.add_argument("--index", default=None, help="dense 인덱스 이름 (기본: 모델명)")
    ap.add_argument("--bm25", default="default", help="BM25 인덱스 이름")
    ap.add_argument("--eval", default=None, help="평가셋 CSV 경로")
    ap.add_argument("--tag", default="", help="리포트 파일명에 붙일 이름표")
    ap.add_argument("--level", default="chunk", choices=["chunk", "doc"],
                    help="평가 단위. doc 은 청킹 전략 간 공정 비교에 쓴다")
    args = ap.parse_args()

    print("=" * 72)
    print("하이브리드 검색 파라미터 실험")
    print("=" * 72)

    eval_path = Path(args.eval) if args.eval else EVAL_CSV
    rows = load_eval_set(eval_path)
    print(f"  평가셋 {len(rows)}건 ({eval_path.name})")

    # --- 인덱스 로드 ---
    faiss_dir = FAISS_ROOT / (args.index or args.model)
    if not (faiss_dir / "config.json").exists():
        sys.exit(f"{faiss_dir} 에 인덱스가 없습니다. 11_build_index.py 를 먼저 실행하세요.")
    bm25_dir = BM25_ROOT / args.bm25
    if not (bm25_dir / "bm25.pkl").exists():
        sys.exit(f"{bm25_dir} 에 BM25 인덱스가 없습니다. 19_build_bm25.py 를 먼저 실행하세요.")

    faiss_store = FaissStore.load(faiss_dir)
    bm25_store = BM25Store.load(bm25_dir)
    embedder = Embedder(args.model)

    print(f"  dense  인덱스 {faiss_store.size:,}개")
    print(f"  sparse 인덱스 {bm25_store.size:,}개")

    if faiss_store.size != bm25_store.size:
        print("  !! 두 인덱스의 청크 수가 다릅니다. 한쪽이 오래된 것일 수 있습니다.")

    if args.level == "doc":
        print("  평가 단위: 문서 (청킹 전략 간 공정 비교용)")

    configs = build_configs(args)
    print(f"  설정 {len(configs)}가지 실험\n")

    all_results: dict[str, dict] = {}

    for label, method, alpha, norm in configs:
        retriever = HybridRetriever(
            embedder, faiss_store, bm25_store,
            method=method, alpha=alpha, candidate_k=args.candidate_k,
            normalize=norm,
        )
        print(f"  [{label}] 실행 중...", end=" ", flush=True)
        metrics, results = evaluate_config(rows, retriever, label, args.level)
        print(f"R@5={metrics['recall@5']:.3f}  MRR={metrics['mrr']:.3f}  "
              f"({metrics['latency_ms']:.0f}ms/건)")

        all_results[label] = {
            "method": method.value,
            "alpha": alpha,
            "normalize": norm.value,
            "overall": metrics,
            "by_difficulty": group_by(results, "difficulty", KS),
            "by_doc_type": group_by(results, "doc_type", KS),
            "failure_analysis": failure_analysis(results),
        }

    # --- 요약 ---
    baseline = all_results["dense 단독"]["overall"]
    best_label = max(all_results, key=lambda k: all_results[k]["overall"]["recall@5"])
    best = all_results[best_label]["overall"]

    print("\n" + "=" * 72)
    print("결과")
    print("=" * 72)
    print(f"  {'설정':<18} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'변화':>8}")
    for label, data in all_results.items():
        m = data["overall"]
        delta = m["recall@5"] - baseline["recall@5"]
        mark = "  <-- 최고" if label == best_label else ""
        d_txt = "기준" if label == "dense 단독" else f"{delta:+.3f}"
        print(f"  {label:<18} {m['recall@1']:>6.3f} {m['recall@5']:>6.3f} "
              f"{m['mrr']:>6.3f} {d_txt:>8}{mark}")

    print("\n  [문서 유형별 R@5]")
    doc_types = sorted(all_results["dense 단독"]["by_doc_type"].keys())
    print(f"  {'설정':<18}" + "".join(f"{dt:>10}" for dt in doc_types))
    for label, data in all_results.items():
        vals = "".join(
            f"{data['by_doc_type'].get(dt, {}).get('recall@5', 0):>10.3f}"
            for dt in doc_types
        )
        print(f"  {label:<18}{vals}")

    print("\n  [실패 분포]")
    print(f"  {'설정':<18} {'1~5위':>7} {'6~20위':>8} {'20위밖':>8}")
    for label, data in all_results.items():
        f = data["failure_analysis"]
        print(f"  {label:<18} {f['top5']:>7} {f['rank_6_20']:>8} {f['beyond_20']:>8}")

    # --- 저장 ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()

    suffix = f"_{args.tag}" if args.tag else ""
    if args.level == "doc":
        suffix += "_doclevel"
    report_md = REPORT_DIR / f"hybrid_tuning{suffix}_{stamp}.md"
    report_json = REPORT_DIR / f"hybrid_tuning{suffix}_{stamp}.json"

    report_md.write_text(build_report(all_results, args.model), encoding="utf-8")
    report_json.write_text(
        json.dumps({
            "date": stamp,
            "model": args.model,
            "index": args.index or args.model,
            "bm25_index": args.bm25,
            "eval_set": eval_path.name,
            "level": args.level,
            "candidate_k": args.candidate_k,
            "results": all_results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gain = best["recall@5"] - baseline["recall@5"]
    print(f"\n  최고: {best_label}  R@5 {best['recall@5']:.3f} ({gain:+.3f})")
    print(f"\n저장 → {report_md.relative_to(PROJECT_ROOT)}")
    print(f"저장 → {report_json.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
