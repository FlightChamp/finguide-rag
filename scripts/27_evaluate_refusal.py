"""
27_evaluate_refusal.py

목적
----
거절 판정 로직을 평가셋으로 측정하고 임계값을 탐색한다.

지표
----
Refusal Accuracy   거절해야 할 때 거절한 비율
False Answer Rate  거절해야 하는데 답한 비율   <- 가장 위험
Over-refusal Rate  답할 수 있는데 거절한 비율
Balanced Accuracy  두 집단 정확도의 평균

왜 정확도만 보면 안 되는가
-----------------------
거절 33건 / 대조군 20건이므로 무조건 거절하면 거절 정확도가 1.0 이 된다.
그러나 그런 시스템은 쓸모가 없다. Balanced Accuracy 로 두 오류를 함께
관리한다.

단계별 기여도
-----------
0단계(패턴), 1단계(검색 신호), 2단계(LLM)를 하나씩 켜가며 각 단계가
무엇을 개선하는지 분리해서 본다. LLM 호출 비용이 정당한지 판단할
근거가 된다.

사용법
-----
    python scripts/27_evaluate_refusal.py            # 전체 (LLM 포함)
    python scripts/27_evaluate_refusal.py --no-llm   # 규칙만 (무료)
    python scripts/27_evaluate_refusal.py --tune     # 임계값 탐색 (무료)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.embedding import MODELS, Embedder, FaissStore  # noqa: E402
from finguide_rag.generation import (  # noqa: E402
    Decision,
    RefusalJudge,
    check_question_pattern,
    check_retrieval_signals,
    compute_signals,
)
from finguide_rag.generation import refusal as refusal_mod  # noqa: E402
from finguide_rag.retrieval import (  # noqa: E402
    BM25Store,
    FusionMethod,
    HybridRetriever,
    NormalizeMethod,
)

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "refusal_eval.csv"
CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"
FAISS_ROOT = PROJECT_ROOT / "data" / "indexes" / "faiss"
BM25_ROOT = PROJECT_ROOT / "data" / "indexes" / "bm25"
REPORT_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / "data" / "interim"

TOP_K = 10


# ==================================================================
# LLM 판정 캐시
# ==================================================================
#
# 왜 필요한가
# ---------
# temperature=0 이어도 OpenAI API 는 동일 입력에 동일 출력을 보장하지
# 않는다. 실제로 같은 코드를 두 번 돌렸을 때 과잉거절이 0.100 과 0.200
# 으로 갈렸다. 4건이 뒤집힌 것이다.
#
# 이 상태에서는 어떤 개선도 검증할 수 없다. 수치가 바뀌었을 때 그것이
# 수정 때문인지 호출 편차인지 구별되지 않기 때문이다. 실제로 거절 규칙을
# 고친 뒤 과잉거절이 0.100 에서 0.200 으로 오른 것을 보고 규칙 탓으로
# 오해할 뻔했다.
#
# 무엇을 해결하고 무엇을 해결하지 못하는가
# ----------------------------------
# 캐싱은 측정의 재현성을 보장한다. 같은 (질문, 근거) 에는 항상 같은
# 판정이 나오므로, 규칙을 고쳐 재측정했을 때 변화는 전부 규칙 때문이다.
#
# 다만 운영 환경의 비결정성 자체는 없애지 못한다. 실제 서비스에서는
# 같은 질문에 다른 판정이 나올 수 있다. 이것은 별도로 다뤄야 할 문제이며,
# 여기서는 평가가 흔들리지 않게 하는 것까지만 한다.
#
# 캐시 키에 프롬프트를 넣는다
# ------------------------
# VERIFY_SYSTEM 을 고치면 키가 달라져 자동으로 다시 호출된다. 프롬프트를
# 바꿔놓고 옛 판정을 재사용하는 사고를 막는다.


def cache_path(model: str, alpha: float, llm_model: str) -> Path:
    tag = f"{model}_a{alpha}_{llm_model}".replace(".", "")
    return CACHE_DIR / f"refusal_verify_cache_{tag}.json"


class VerifyCache:
    """verify_with_llm 호출을 가로채 디스크에 캐시한다."""

    def __init__(self, path: Path, refresh: bool = False):
        self.path = path
        self.store: dict = {}
        if path.exists() and not refresh:
            try:
                self.store = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.store = {}
        self.hits = 0
        self.calls = 0
        self._original = None

    def install(self) -> None:
        """모듈의 verify_with_llm 을 캐시 버전으로 교체한다."""
        self._original = refusal_mod.verify_with_llm
        original = self._original
        store, outer = self.store, self

        def cached_verify(client, model, question, evidence, prior):
            key = hashlib.sha256(
                f"{model}|{refusal_mod.VERIFY_SYSTEM}|{question}|{evidence}"
                .encode("utf-8")
            ).hexdigest()[:24]

            if key in store:
                outer.hits += 1
                d = store[key]
            else:
                r = original(client, model, question, evidence, prior)
                d = {
                    "decision": r.decision.value,
                    "stage": r.stage,
                    "llm_reason": (r.signals or {}).get("llm_reason", ""),
                    "tokens": (r.signals or {}).get("llm_tokens", 0),
                }
                store[key] = d
                outer.calls += 1

            # prior 에 담긴 사유는 캐시하지 않는다. 임계값에 따라 달라지므로
            # 매번 현재 값으로 다시 만든다.
            if d["decision"] == Decision.ANSWER.value:
                return refusal_mod.RefusalResult(
                    decision=Decision.ANSWER, stage=d["stage"], confidence=0.80,
                    signals={**(prior.signals or {}),
                             "llm_reason": d["llm_reason"],
                             "llm_tokens": d["tokens"]},
                )

            reason = prior.reason or refusal_mod.RefusalReason.NO_EVIDENCE
            if reason == refusal_mod.RefusalReason.BLANK_VALUE:
                message = ("문서에 해당 항목은 있으나 구체적인 값이 기재되어 있지 "
                           "않습니다. 계약 조건에 따라 결정되므로 담당 부서에 확인해 "
                           "주시기 바랍니다.")
            else:
                message = "문서에서 이 질문에 답할 근거를 찾지 못했습니다."

            return refusal_mod.RefusalResult(
                decision=Decision.REFUSE, reason=reason, stage=d["stage"],
                confidence=0.80, message=message,
                signals={**(prior.signals or {}),
                         "llm_reason": d["llm_reason"],
                         "llm_tokens": d["tokens"]},
            )

        refusal_mod.verify_with_llm = cached_verify

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.store, ensure_ascii=False, indent=2), encoding="utf-8")


# ==================================================================
# 지표
# ==================================================================


def score(records: list[dict]) -> dict:
    """혼동행렬과 지표. 양성(positive)은 '거절해야 함'이다."""
    tp = sum(1 for r in records if r["expected"] == "refuse" and r["predicted"] == "refuse")
    fn = sum(1 for r in records if r["expected"] == "refuse" and r["predicted"] == "answer")
    fp = sum(1 for r in records if r["expected"] == "answer" and r["predicted"] == "refuse")
    tn = sum(1 for r in records if r["expected"] == "answer" and r["predicted"] == "answer")

    n_refuse, n_answer = tp + fn, fp + tn
    refusal_acc = tp / n_refuse if n_refuse else 0.0
    answer_acc = tn / n_answer if n_answer else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * refusal_acc / (precision + refusal_acc)
          if (precision + refusal_acc) else 0.0)

    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "refusal_accuracy": round(refusal_acc, 4),
        "false_answer_rate": round(fn / n_refuse if n_refuse else 0.0, 4),
        "over_refusal_rate": round(fp / n_answer if n_answer else 0.0, 4),
        "answer_accuracy": round(answer_acc, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round((refusal_acc + answer_acc) / 2, 4),
        "n": len(records),
    }


# ==================================================================
# 실행
# ==================================================================


def load_eval() -> list[dict]:
    if not EVAL_CSV.exists():
        sys.exit(f"{EVAL_CSV} 없음. 먼저 26_analyze_refusal_signals.py 를 실행하세요.")
    with EVAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_chunk_texts() -> dict[str, str]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} 없음.")
    out = {}
    for line in CHUNKS_PATH.open(encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            out[c["chunk_id"]] = c.get("text", "")
    return out


def retrieve_all(rows: list[dict], retriever, chunk_texts: dict) -> list[dict]:
    """검색을 한 번만 수행하고 결과를 재사용한다.

    임계값 탐색에서 수십 조합을 시험하므로 매번 검색하면 시간이 크게 는다.
    """
    cached = []
    for i, row in enumerate(rows, 1):
        hits = retriever.search(row["question"], top_k=TOP_K)
        cached.append({
            "row": row,
            "hits": hits,
            "signals": compute_signals(hits, chunk_texts),
        })
        if i % 20 == 0:
            print(f"    {i}/{len(rows)}")
    return cached


def run_stages(cached: list[dict], use_pattern: bool, use_signals: bool,
               judge: RefusalJudge | None, chunk_texts: dict) -> tuple[list[dict], int]:
    """지정한 단계만 켜고 판정한다."""
    records = []
    llm_calls = 0

    for item in cached:
        row, hits, signals = item["row"], item["hits"], item["signals"]
        decision = stage = reason = None

        if use_pattern:
            r = check_question_pattern(row["question"])
            if r is not None:
                decision, stage = "refuse", "pattern"
                reason = r.reason.value if r.reason else ""

        if decision is None and use_signals:
            r = check_retrieval_signals(signals, row["question"])
            if r.decision == Decision.REFUSE:
                decision, stage = "refuse", "retrieval"
                reason = r.reason.value if r.reason else ""

        if decision is None:
            if judge is not None:
                r = judge.judge(row["question"], hits, chunk_texts)
                decision = "refuse" if r.should_refuse else "answer"
                stage = r.stage
                reason = r.reason.value if r.reason else ""
                if r.stage in ("llm", "llm_error"):
                    llm_calls += 1
            else:
                # 판정 수단이 없으면 답변으로 둔다(단계 기여도 측정용)
                decision, stage, reason = "answer", "default", ""

        records.append({
            "query_id": row["query_id"],
            "question": row["question"],
            "expected": row["expected"],
            "refusal_type": row["refusal_type"],
            "predicted": decision,
            "stage": stage,
            "reason": reason or "",
            "score_top1": round(signals["score_top1"], 4),
            "gap_1_5": round(signals["gap_1_5"], 4),
            "blank_ratio": round(signals["blank_ratio"], 2),
            "correct": decision == row["expected"],
        })

    return records, llm_calls


# ==================================================================
# 임계값 탐색
# ==================================================================


def tune(cached: list[dict], chunk_texts: dict) -> list[dict]:
    """1단계 임계값을 탐색한다.

    현재 1단계는 두 가지만 판정한다.
      - 검색 실패 (HARD_REFUSE_TOP1, HARD_REFUSE_GAP15)
      - 공란 (BLANK_RATIO_THRESHOLD)

    LLM 없이 규칙만으로 평가하므로 비용이 들지 않는다. 다만 이 탐색은
    "LLM 이 없을 때의 최선"을 찾는 것이므로, 최종 성능은 LLM 을 얹어
    다시 측정해야 한다.
    """
    saved = (refusal_mod.HARD_REFUSE_TOP1,
             refusal_mod.HARD_REFUSE_GAP15,
             refusal_mod.BLANK_RATIO_THRESHOLD)
    results = []

    combos = list(itertools.product(
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],   # top1
        [0.03, 0.05, 0.08, 0.12],               # gap_1_5
        [0.20, 0.35, 0.50, 0.60],               # blank_ratio
    ))
    print(f"  조합 {len(combos)}가지 탐색 중...")

    for t1, g15, br in combos:
        refusal_mod.HARD_REFUSE_TOP1 = t1
        refusal_mod.HARD_REFUSE_GAP15 = g15
        refusal_mod.BLANK_RATIO_THRESHOLD = br

        records, _ = run_stages(cached, True, True, None, chunk_texts)
        m = score(records)
        results.append({
            "hard_top1": t1, "hard_gap15": g15, "blank_ratio": br,
            "to_llm": sum(1 for r in records if r["stage"] == "default"),
            **m,
        })

    (refusal_mod.HARD_REFUSE_TOP1,
     refusal_mod.HARD_REFUSE_GAP15,
     refusal_mod.BLANK_RATIO_THRESHOLD) = saved
    return results


# ==================================================================
# 메인
# ==================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small", choices=list(MODELS))
    ap.add_argument("--index", default=None)
    ap.add_argument("--bm25", default="default")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--llm-model", default="gpt-4.1-mini")
    ap.add_argument("--no-llm", action="store_true", help="LLM 단계 생략")
    ap.add_argument("--tune", action="store_true", help="임계값 탐색")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="LLM 판정 캐시를 버리고 다시 호출")
    ap.add_argument("--no-cache", action="store_true",
                    help="캐시를 쓰지 않는다(호출 편차 확인용)")
    args = ap.parse_args()

    print("=" * 76)
    print("거절 로직 평가")
    print("=" * 76)

    rows = load_eval()
    chunk_texts = load_chunk_texts()
    n_refuse = sum(1 for r in rows if r["expected"] == "refuse")
    n_answer = len(rows) - n_refuse
    print(f"  평가셋 {len(rows)}건 (거절 {n_refuse} / 대조군 {n_answer})")

    faiss_dir = FAISS_ROOT / (args.index or args.model)
    bm25_dir = BM25_ROOT / args.bm25
    if not (faiss_dir / "config.json").exists():
        sys.exit(f"{faiss_dir} 에 인덱스가 없습니다.")

    retriever = HybridRetriever(
        Embedder(args.model),
        FaissStore.load(faiss_dir),
        BM25Store.load(bm25_dir),
        method=FusionMethod.WEIGHTED,
        alpha=args.alpha,
        normalize=NormalizeMethod.MINMAX,
    )

    print(f"\n  검색 중... (weighted α={args.alpha})")
    cached = retrieve_all(rows, retriever, chunk_texts)

    # --- 임계값 탐색 ---
    if args.tune:
        print("\n" + "=" * 76)
        print("임계값 탐색 (규칙만, LLM 미사용)")
        print("=" * 76)
        tuned = tune(cached, chunk_texts)

        # 금융 도메인의 비용 비대칭을 반영한다.
        # 근거 없는 답변이 과잉 거절보다 비싸므로 가중치를 2배로 둔다.
        for t in tuned:
            t["cost"] = t["false_answer_rate"] * 2.0 + t["over_refusal_rate"]
        tuned.sort(key=lambda t: (t["cost"], -t["balanced_accuracy"]))

        print(f"\n  {'top1':>6}{'gap15':>7}{'blank':>7}"
              f"{'FAR':>8}{'과잉거절':>9}{'BalAcc':>8}{'LLM행':>7}")
        print("  " + "-" * 52)
        for t in tuned[:12]:
            print(f"  {t['hard_top1']:>6.2f}{t['hard_gap15']:>7.2f}{t['blank_ratio']:>7.2f}"
                  f"{t['false_answer_rate']:>8.3f}{t['over_refusal_rate']:>9.3f}"
                  f"{t['balanced_accuracy']:>8.3f}{t['to_llm']:>7}")

        best = tuned[0]
        print(f"\n  권장 임계값")
        print(f"    HARD_REFUSE_TOP1      = {best['hard_top1']}")
        print(f"    HARD_REFUSE_GAP15     = {best['hard_gap15']}")
        print(f"    BLANK_RATIO_THRESHOLD = {best['blank_ratio']}")
        print(f"    -> src/finguide_rag/generation/refusal.py 에서 수정")
        print(f"\n  참고: 이 값은 LLM 없이 최선인 조합이다.")
        print(f"        LLM 을 얹으면 to_llm({best['to_llm']}건)이 다시 판정된다.")

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / f"refusal_tuning_{date.today().isoformat()}.csv"
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tuned[0].keys()))
            w.writeheader()
            w.writerows(tuned)
        print(f"\n저장 → {out.relative_to(PROJECT_ROOT)}")
        return

    # --- 단계별 기여도 ---
    client = None
    verify_cache = None
    if not args.no_llm:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI()

        if not args.no_cache:
            verify_cache = VerifyCache(
                cache_path(args.model, args.alpha, args.llm_model),
                args.refresh_cache)
            verify_cache.install()
            print(f"\n  LLM 판정 캐시 {len(verify_cache.store)}건 로드")

    stages = [
        ("0단계만 (질문 패턴)", True, False, None),
        ("0+1단계 (패턴+검색)", True, True, None),
    ]
    if client is not None:
        stages.append(("0+1+2단계 (LLM 포함)", True, True,
                       RefusalJudge(client, args.llm_model)))

    print("\n" + "=" * 76)
    print("단계별 기여도")
    print("=" * 76)
    print(f"  {'구성':<24}{'거절정확도':>10}{'FAR':>8}{'과잉거절':>9}{'BalAcc':>8}{'LLM':>6}")
    print("  " + "-" * 65)

    all_records: dict[str, list[dict]] = {}
    all_metrics: dict[str, dict] = {}

    for label, use_pat, use_sig, judge in stages:
        records, calls = run_stages(cached, use_pat, use_sig, judge, chunk_texts)
        m = score(records)
        all_records[label] = records
        all_metrics[label] = {**m, "llm_calls": calls}
        print(f"  {label:<24}{m['refusal_accuracy']:>10.3f}"
              f"{m['false_answer_rate']:>8.3f}{m['over_refusal_rate']:>9.3f}"
              f"{m['balanced_accuracy']:>8.3f}{calls:>6}")

    if verify_cache is not None:
        verify_cache.flush()
        print(f"\n  LLM 판정 캐시 적중 {verify_cache.hits}회 / "
              f"신규 호출 {verify_cache.calls}회")
        print(f"    → {cache_path(args.model, args.alpha, args.llm_model).name}")
        if verify_cache.calls == 0:
            print(f"    전건 캐시 재사용. 이 실행은 완전히 재현 가능하다.")

    final_label = stages[-1][0]
    final = all_records[final_label]
    m = all_metrics[final_label]

    print("\n" + "=" * 76)
    print(f"최종 결과 — {final_label}")
    print("=" * 76)
    print("               예측: 거절    예측: 답변")
    print(f"  실제 거절 {m['tp']:>10}{m['fn']:>14}   <- fn 이 False Answer")
    print(f"  실제 답변 {m['fp']:>10}{m['tn']:>14}   <- fp 가 과잉 거절")

    print(f"\n  Refusal Accuracy  {m['refusal_accuracy']:.3f}")
    print(f"  False Answer Rate {m['false_answer_rate']:.3f}   <- 가장 위험")
    print(f"  Over-refusal Rate {m['over_refusal_rate']:.3f}")
    print(f"  Balanced Accuracy {m['balanced_accuracy']:.3f}")
    print(f"  F1                {m['f1']:.3f}")

    print("\n  [거절 유형별 정확도]")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in final:
        if r["expected"] == "refuse":
            by_type[r["refusal_type"]].append(r)
    for t in sorted(by_type):
        g = by_type[t]
        hit = sum(1 for r in g if r["predicted"] == "refuse")
        print(f"    {t:<16} {hit}/{len(g)} = {hit / len(g):.3f}")

    print("\n  [판정 단계 분포]")
    for stage, cnt in Counter(r["stage"] for r in final).most_common():
        print(f"    {stage:<20} {cnt:>3}건 ({cnt / len(final):.0%})")

    fa = [r for r in final if r["expected"] == "refuse" and r["predicted"] == "answer"]
    orf = [r for r in final if r["expected"] == "answer" and r["predicted"] == "refuse"]

    if fa:
        print(f"\n  [False Answer {len(fa)}건]")
        for r in fa[:8]:
            print(f"    [{r['refusal_type']:<13}] {r['question'][:46]}")
            print(f"       top1={r['score_top1']:.3f} gap15={r['gap_1_5']:.3f} "
                  f"blank={r['blank_ratio']:.2f} stage={r['stage']}")
    if orf:
        print(f"\n  [과잉 거절 {len(orf)}건]")
        for r in orf[:8]:
            print(f"    {r['question'][:50]}")
            print(f"       stage={r['stage']} reason={r['reason']} "
                  f"top1={r['score_top1']:.3f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    csv_path = REPORT_DIR / f"refusal_eval_{stamp}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final[0].keys()))
        w.writeheader()
        w.writerows(final)

    json_path = REPORT_DIR / f"refusal_eval_{stamp}.json"
    json_path.write_text(json.dumps({
        "date": stamp,
        "eval_set": {"total": len(rows), "refuse": n_refuse, "answer": n_answer},
        "retrieval": {"model": args.model, "alpha": args.alpha},
        "thresholds": {
            "hard_refuse_top1": refusal_mod.HARD_REFUSE_TOP1,
            "hard_refuse_gap15": refusal_mod.HARD_REFUSE_GAP15,
            "blank_ratio": refusal_mod.BLANK_RATIO_THRESHOLD,
        },
        "stages": all_metrics,
        "by_type": {t: {"n": len(g),
                        "hit": sum(1 for r in g if r["predicted"] == "refuse")}
                    for t, g in by_type.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n저장 → {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"저장 → {json_path.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()

