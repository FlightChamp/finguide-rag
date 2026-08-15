"""
28_tune_refusal_v2.py

목적
----
거절 판정 임계값을 "전체 파이프라인(0+1+2단계) 기준"으로 탐색한다.

27번과의 차이
------------
27번의 --tune 은 LLM 없이 규칙만으로 임계값을 골랐다. 그러나 실제
시스템은 규칙이 넘긴 사례를 LLM 이 다시 판정한다. 규칙 단계만 최적화한
값이 전체 최적이라는 보장이 없다.

전체 파이프라인으로 수백 조합을 재려면 조합마다 LLM 을 호출해야 하므로
비용이 감당되지 않는 것처럼 보인다. 그러나 그렇지 않다.

    LLM 판정은 (질문, 근거) 만으로 결정된다.
    근거는 검색 결과에서 나오고, 검색 결과는 임계값과 무관하다.
    따라서 같은 질문의 LLM 판정은 어떤 임계값에서도 동일하다.

즉 질문당 한 번만 호출해 캐시해두면(53건 = 53콜), 이후 모든 조합을
비용 0 으로 전체 파이프라인 평가할 수 있다. 캐시는 파일로 저장하므로
재실행 시에는 호출이 아예 없다.

추가로 다루는 것
--------------
1. 그리드 확장   1차 탐색에서 top1 이 상단 경계(0.75), blank 가 하단
                 경계(0.20)에 걸렸다. 경계에서 멈춘 최적값은 신뢰할 수
                 없으므로 범위를 넓혀 다시 본다.
2. 가중치 민감도  27번은 cost = 2*FAR + 과잉거절 로 골랐다. 이 "2"는
                 임의값이다. 1.0~5.0 에서 최적 조합이 바뀌는지 본다.
3. LLM 호출 비용  기존 cost 에는 호출량 항이 없었다. 성능이 같다면
                 호출이 적은 쪽이 낫다.
4. 동률 평탄면    53건에서는 1건이 FAR 0.03 을 움직인다. 같은 성능을
                 내는 조합이 몇 개나 되는지 세어, 임계값 선택이 실제로
                 의미 있는 결정인지 확인한다.
5. 파레토 경계    FAR 과 과잉거절 양쪽에서 지배당하지 않는 조합만 추린다.

사용법
-----
    python scripts/28_tune_refusal_v2.py
        전체 탐색. 최초 1회만 LLM 53콜, 이후 캐시 재사용.

    python scripts/28_tune_refusal_v2.py --rules-only
        LLM 완전 미사용(무료). 27번 --tune 과 비교용.

    python scripts/28_tune_refusal_v2.py --apply 0.75,0.08,0.20
        refusal.py 의 임계값 3개를 교체한다. 원본은 .bak 로 백업된다.

    python scripts/28_tune_refusal_v2.py --refresh-cache
        LLM 판정 캐시를 버리고 다시 호출한다(프롬프트를 고쳤을 때).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.embedding import MODELS, Embedder, FaissStore  # noqa: E402
from finguide_rag.generation import refusal as refusal_mod  # noqa: E402
from finguide_rag.retrieval import (  # noqa: E402
    BM25Store,
    FusionMethod,
    HybridRetriever,
    NormalizeMethod,
)

Decision = refusal_mod.Decision

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "refusal_eval.csv"
CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"
FAISS_ROOT = PROJECT_ROOT / "data" / "indexes" / "faiss"
BM25_ROOT = PROJECT_ROOT / "data" / "indexes" / "bm25"
REPORT_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / "data" / "interim"
REFUSAL_PY = PROJECT_ROOT / "src" / "finguide_rag" / "generation" / "refusal.py"

TOP_K = 10

# 확장 그리드.
# 1차 탐색(6 x 4 x 4 = 96)에서 top1 이 최댓값 0.75, blank 가 최솟값 0.20
# 으로 선택됐다. 둘 다 탐색 범위의 끝이므로 진짜 최적점이 밖에 있을 수
# 있다. 양방향으로 넓힌다.
GRID_TOP1 = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
GRID_GAP15 = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20]
GRID_BLANK = [0.05, 0.10, 0.15, 0.20, 0.35, 0.50, 0.60, 0.80]

# FAR 가중치 민감도를 볼 지점.
# 27번이 쓰던 2.0 이 특별한 근거가 있는 값인지 확인한다.
WEIGHTS = [1.0, 1.5, 2.0, 3.0, 5.0]

# LLM 호출 비용 계수. cost 에 lambda * (호출건수/전체) 로 더한다.
# 성능이 같으면 호출이 적은 쪽을 고르게 하는 정도의 작은 값.
LAMBDA_LLM = 0.10


# ==================================================================
# 지표 (27번과 동일 정의)
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
# 로딩
# ==================================================================


def load_eval() -> list[dict]:
    if not EVAL_CSV.exists():
        sys.exit(f"{EVAL_CSV} 없음.")
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
    """검색은 임계값과 무관하므로 한 번만 수행한다."""
    cached = []
    for i, row in enumerate(rows, 1):
        hits = retriever.search(row["question"], top_k=TOP_K)
        cached.append({
            "row": row,
            "hits": hits,
            "signals": refusal_mod.compute_signals(hits, chunk_texts),
        })
        if i % 20 == 0:
            print(f"    {i}/{len(rows)}")
    return cached


# ==================================================================
# LLM 판정 캐시
# ==================================================================


def cache_path(model: str, alpha: float, llm_model: str) -> Path:
    tag = f"{model}_a{alpha}_{llm_model}".replace(".", "")
    return CACHE_DIR / f"refusal_llm_verdicts_{tag}.json"


def precompute_verdicts(cached: list[dict], chunk_texts: dict,
                        client, llm_model: str, path: Path,
                        refresh: bool) -> dict[str, dict]:
    """질문별 LLM 판정을 한 번만 계산해 캐시한다.

    LLM 이 보는 것은 (질문, 상위 3청크 근거) 뿐이고 둘 다 임계값과
    무관하므로, 판정 결과도 임계값과 무관하다. 따라서 캐시가 성립한다.
    """
    store: dict[str, dict] = {}
    if path.exists() and not refresh:
        store = json.loads(path.read_text(encoding="utf-8"))

    todo = [it for it in cached if it["row"]["query_id"] not in store]
    if todo:
        print(f"  LLM 판정 {len(todo)}건 호출 중 (캐시 {len(store)}건 재사용)...")
    else:
        print(f"  LLM 판정 전건 캐시 재사용 ({len(store)}건). 호출 없음.")

    for i, item in enumerate(todo, 1):
        row, hits, signals = item["row"], item["hits"], item["signals"]
        evidence = refusal_mod.build_evidence(hits, chunk_texts)
        prior = refusal_mod.RefusalResult(decision=Decision.UNCERTAIN, signals=signals)
        r = refusal_mod.verify_with_llm(client, llm_model, row["question"], evidence, prior)
        store[row["query_id"]] = {
            "verdict": "ANSWERABLE" if r.decision == Decision.ANSWER else "REFUSE",
            "stage": r.stage,
            "llm_reason": (r.signals or {}).get("llm_reason", ""),
            "tokens": (r.signals or {}).get("llm_tokens", 0),
        }
        if i % 10 == 0:
            print(f"    {i}/{len(todo)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")

    n_err = sum(1 for v in store.values() if v["stage"] == "llm_error")
    if n_err:
        print(f"  주의: LLM 호출 오류 {n_err}건. 해당 건은 거절로 처리됨.")
    tok = sum(v.get("tokens", 0) for v in store.values())
    print(f"  캐시 저장 → {path.relative_to(PROJECT_ROOT)} (누적 토큰 {tok:,})")
    return store


# ==================================================================
# 판정 실행
# ==================================================================


def run_pipeline(cached: list[dict], pattern_hits: list,
                 verdicts: dict[str, dict] | None) -> tuple[list[dict], int]:
    """현재 모듈 전역 임계값으로 0+1(+2)단계를 실행한다.

    verdicts 가 None 이면 LLM 단계를 끄고, 미판정 건은 답변으로 둔다
    (27번 --tune 과 동일한 규칙만 평가).
    """
    records = []
    n_llm = 0

    for item, pat in zip(cached, pattern_hits):
        row, signals = item["row"], item["signals"]
        decision = stage = reason = None

        # 0단계 — 질문 패턴 (임계값과 무관하므로 미리 계산해 재사용)
        if pat is not None:
            decision, stage = "refuse", "pattern"
            reason = pat.reason.value if pat.reason else ""

        # 1단계 — 검색 신호 (임계값 의존)
        if decision is None:
            r = refusal_mod.check_retrieval_signals(signals, row["question"])
            if r.decision == Decision.REFUSE:
                decision, stage = "refuse", "retrieval"
                reason = r.reason.value if r.reason else ""

        # 2단계 — LLM (캐시된 판정 사용)
        if decision is None:
            if verdicts is not None:
                v = verdicts.get(row["query_id"], {"verdict": "REFUSE"})
                decision = "answer" if v["verdict"] == "ANSWERABLE" else "refuse"
                stage = "llm"
                reason = "" if decision == "answer" else "no_evidence"
                n_llm += 1
            else:
                decision, stage, reason = "answer", "default", ""
                n_llm += 1   # 규칙만 모드에서도 'LLM 으로 넘어갔을 건수'로 센다

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

    return records, n_llm


def sweep(cached: list[dict], pattern_hits: list,
          verdicts: dict[str, dict] | None) -> list[dict]:
    """확장 그리드 전체를 평가한다."""
    saved = (refusal_mod.HARD_REFUSE_TOP1,
             refusal_mod.HARD_REFUSE_GAP15,
             refusal_mod.BLANK_RATIO_THRESHOLD)

    combos = list(itertools.product(GRID_TOP1, GRID_GAP15, GRID_BLANK))
    print(f"  조합 {len(combos)}가지 평가 중 (LLM 호출 없음)...")

    results = []
    for n, (t1, g15, br) in enumerate(combos, 1):
        refusal_mod.HARD_REFUSE_TOP1 = t1
        refusal_mod.HARD_REFUSE_GAP15 = g15
        refusal_mod.BLANK_RATIO_THRESHOLD = br

        records, n_llm = run_pipeline(cached, pattern_hits, verdicts)
        m = score(records)
        results.append({
            "hard_top1": t1, "hard_gap15": g15, "blank_ratio": br,
            "to_llm": n_llm,
            **m,
        })
        if n % 100 == 0:
            print(f"    {n}/{len(combos)}")

    (refusal_mod.HARD_REFUSE_TOP1,
     refusal_mod.HARD_REFUSE_GAP15,
     refusal_mod.BLANK_RATIO_THRESHOLD) = saved
    return results


# ==================================================================
# 분석
# ==================================================================


def add_cost(rows: list[dict], w: float, lam: float) -> None:
    for r in rows:
        r["cost"] = round(
            r["false_answer_rate"] * w
            + r["over_refusal_rate"]
            + lam * (r["to_llm"] / r["n"]),
            5,
        )


def pareto(rows: list[dict]) -> list[dict]:
    """FAR 과 과잉거절 양쪽에서 지배당하지 않는 조합만 남긴다."""
    front = []
    for a in rows:
        dominated = any(
            b is not a
            and b["false_answer_rate"] <= a["false_answer_rate"]
            and b["over_refusal_rate"] <= a["over_refusal_rate"]
            and (b["false_answer_rate"] < a["false_answer_rate"]
                 or b["over_refusal_rate"] < a["over_refusal_rate"])
            for b in rows
        )
        if not dominated:
            front.append(a)
    front.sort(key=lambda r: (r["false_answer_rate"], r["over_refusal_rate"]))
    return front


def boundary_warning(best: dict) -> list[str]:
    """최적값이 그리드 끝에 걸렸는지 확인한다."""
    msgs = []
    for key, grid, name in [
        ("hard_top1", GRID_TOP1, "HARD_REFUSE_TOP1"),
        ("hard_gap15", GRID_GAP15, "HARD_REFUSE_GAP15"),
        ("blank_ratio", GRID_BLANK, "BLANK_RATIO_THRESHOLD"),
    ]:
        if best[key] == grid[0]:
            msgs.append(f"{name}={best[key]} 는 탐색 범위의 최솟값입니다. 더 낮춰볼 것.")
        elif best[key] == grid[-1]:
            msgs.append(f"{name}={best[key]} 는 탐색 범위의 최댓값입니다. 더 높여볼 것.")
    return msgs


def plateau(rows: list[dict], best: dict) -> list[dict]:
    """최적 조합과 혼동행렬이 완전히 동일한 조합들을 찾는다."""
    key = (best["fn"], best["fp"], best["to_llm"])
    return [r for r in rows
            if (r["fn"], r["fp"], r["to_llm"]) == key]


# ==================================================================
# 임계값 적용
# ==================================================================


def apply_thresholds(spec: str) -> None:
    """refusal.py 의 상수 3개를 교체한다. 원본은 .bak 로 남긴다."""
    try:
        t1, g15, br = [float(x) for x in spec.split(",")]
    except ValueError:
        sys.exit("--apply 형식: --apply 0.75,0.08,0.20")

    if not REFUSAL_PY.exists():
        sys.exit(f"{REFUSAL_PY} 없음.")

    text = REFUSAL_PY.read_text(encoding="utf-8")
    backup = REFUSAL_PY.with_suffix(".py.bak")
    shutil.copy2(REFUSAL_PY, backup)

    pairs = [
        ("HARD_REFUSE_TOP1", t1),
        ("HARD_REFUSE_GAP15", g15),
        ("BLANK_RATIO_THRESHOLD", br),
    ]
    for name, val in pairs:
        text, n = re.subn(
            rf"^{name}\s*=\s*[0-9.]+",
            f"{name} = {val}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n != 1:
            sys.exit(f"{name} 를 찾지 못했습니다. 백업({backup.name})은 그대로입니다.")
        print(f"  {name} = {val}")

    REFUSAL_PY.write_text(text, encoding="utf-8")
    print(f"\n  적용 완료. 백업 → {backup.relative_to(PROJECT_ROOT)}")
    print("  확인: python scripts/27_evaluate_refusal.py")


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
    ap.add_argument("--rules-only", action="store_true",
                    help="LLM 완전 미사용 (27번 --tune 과 동일 조건)")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="LLM 판정 캐시를 버리고 다시 호출")
    ap.add_argument("--apply", default=None, metavar="T1,G15,BR",
                    help="refusal.py 임계값 교체 (예: --apply 0.75,0.08,0.20)")
    args = ap.parse_args()

    if args.apply:
        print("=" * 76)
        print("임계값 적용")
        print("=" * 76)
        apply_thresholds(args.apply)
        return

    print("=" * 76)
    print("거절 임계값 탐색 v2 — 전체 파이프라인 기준")
    print("=" * 76)

    rows = load_eval()
    chunk_texts = load_chunk_texts()
    n_refuse = sum(1 for r in rows if r["expected"] == "refuse")
    print(f"  평가셋 {len(rows)}건 (거절 {n_refuse} / 대조군 {len(rows) - n_refuse})")
    print(f"  현재 refusal.py 값: top1={refusal_mod.HARD_REFUSE_TOP1} "
          f"gap15={refusal_mod.HARD_REFUSE_GAP15} "
          f"blank={refusal_mod.BLANK_RATIO_THRESHOLD}")

    faiss_dir = FAISS_ROOT / (args.index or args.model)
    if not (faiss_dir / "config.json").exists():
        sys.exit(f"{faiss_dir} 에 인덱스가 없습니다.")

    retriever = HybridRetriever(
        Embedder(args.model),
        FaissStore.load(faiss_dir),
        BM25Store.load(BM25_ROOT / args.bm25),
        method=FusionMethod.WEIGHTED,
        alpha=args.alpha,
        normalize=NormalizeMethod.MINMAX,
    )

    print(f"\n  검색 중... (weighted α={args.alpha})")
    cached = retrieve_all(rows, retriever, chunk_texts)

    # 0단계는 임계값과 무관하므로 미리 한 번만 계산한다.
    pattern_hits = [refusal_mod.check_question_pattern(it["row"]["question"])
                    for it in cached]

    # --- LLM 판정 캐시 ---
    verdicts = None
    if not args.rules_only:
        print("\n" + "=" * 76)
        print("LLM 근거 검증 (질문당 1회, 이후 전 조합 재사용)")
        print("=" * 76)
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI()
        verdicts = precompute_verdicts(
            cached, chunk_texts, client, args.llm_model,
            cache_path(args.model, args.alpha, args.llm_model),
            args.refresh_cache,
        )

    mode = "규칙만" if args.rules_only else "전체 파이프라인(0+1+2)"

    # --- 그리드 탐색 ---
    print("\n" + "=" * 76)
    print(f"그리드 탐색 — {mode}")
    print("=" * 76)
    results = sweep(cached, pattern_hits, verdicts)

    # --- 가중치 민감도 ---
    print("\n" + "=" * 76)
    print("가중치 민감도 — FAR 가중치를 바꾸면 최적 조합이 바뀌는가")
    print("=" * 76)
    print(f"  cost = w*FAR + 과잉거절 + {LAMBDA_LLM}*(LLM호출/전체)")
    print(f"\n  {'w':>5}  {'top1':>6}{'gap15':>7}{'blank':>7}"
          f"{'FAR':>8}{'과잉거절':>9}{'BalAcc':>8}{'LLM':>6}{'동률':>6}")
    print("  " + "-" * 62)

    best_by_w = {}
    for w in WEIGHTS:
        add_cost(results, w, LAMBDA_LLM)
        ranked = sorted(results, key=lambda r: (r["cost"], -r["balanced_accuracy"]))
        b = ranked[0]
        best_by_w[w] = b
        tie = len(plateau(results, b))
        mark = " <-" if w == 2.0 else ""
        print(f"  {w:>5.1f}  {b['hard_top1']:>6.2f}{b['hard_gap15']:>7.2f}"
              f"{b['blank_ratio']:>7.2f}{b['false_answer_rate']:>8.3f}"
              f"{b['over_refusal_rate']:>9.3f}{b['balanced_accuracy']:>8.3f}"
              f"{b['to_llm']:>6}{tie:>6}{mark}")

    distinct = {(b["fn"], b["fp"]) for b in best_by_w.values()}
    if len(distinct) == 1:
        print("\n  -> 가중치를 바꿔도 최적 조합의 혼동행렬이 같다. 선택이 안정적이다.")
    else:
        print(f"\n  -> 가중치에 따라 최적 조합이 달라진다({len(distinct)}가지).")
        print("     가중치 2.0 은 임의 선택이므로 README 에 근거를 명시해야 한다.")

    # --- 기준 가중치(2.0)로 상위 목록 ---
    add_cost(results, 2.0, LAMBDA_LLM)
    ranked = sorted(results, key=lambda r: (r["cost"], -r["balanced_accuracy"]))
    best = ranked[0]

    print("\n" + "=" * 76)
    print("상위 15조합 (w=2.0)")
    print("=" * 76)
    print(f"  {'top1':>6}{'gap15':>7}{'blank':>7}{'FAR':>8}{'과잉거절':>9}"
          f"{'BalAcc':>8}{'LLM':>6}{'cost':>8}")
    print("  " + "-" * 60)
    for r in ranked[:15]:
        print(f"  {r['hard_top1']:>6.2f}{r['hard_gap15']:>7.2f}{r['blank_ratio']:>7.2f}"
              f"{r['false_answer_rate']:>8.3f}{r['over_refusal_rate']:>9.3f}"
              f"{r['balanced_accuracy']:>8.3f}{r['to_llm']:>6}{r['cost']:>8.3f}")

    # --- 동률 평탄면 ---
    flat = plateau(results, best)
    print("\n" + "=" * 76)
    print("동률 조합 — 최적값과 혼동행렬이 완전히 같은 조합")
    print("=" * 76)
    print(f"  {len(flat)}개 / 전체 {len(results)}개")
    if flat:
        print(f"    top1  {min(r['hard_top1'] for r in flat):.2f} ~ "
              f"{max(r['hard_top1'] for r in flat):.2f}")
        print(f"    gap15 {min(r['hard_gap15'] for r in flat):.2f} ~ "
              f"{max(r['hard_gap15'] for r in flat):.2f}")
        print(f"    blank {min(r['blank_ratio'] for r in flat):.2f} ~ "
              f"{max(r['blank_ratio'] for r in flat):.2f}")
    if len(flat) > 1:
        print("\n  -> 이 범위 안에서는 어떤 값을 골라도 결과가 같다.")
        print("     단일 최적값이 아니라 '평탄 구간의 중앙'을 고르는 것이 안전하다.")

    # --- 파레토 ---
    front = pareto(results)
    print("\n" + "=" * 76)
    print("파레토 경계 (FAR / 과잉거절 트레이드오프)")
    print("=" * 76)
    print(f"  {'top1':>6}{'gap15':>7}{'blank':>7}{'FAR':>8}{'과잉거절':>9}"
          f"{'BalAcc':>8}{'LLM':>6}")
    print("  " + "-" * 52)
    seen = set()
    for r in front:
        k = (r["fn"], r["fp"])
        if k in seen:
            continue
        seen.add(k)
        print(f"  {r['hard_top1']:>6.2f}{r['hard_gap15']:>7.2f}{r['blank_ratio']:>7.2f}"
              f"{r['false_answer_rate']:>8.3f}{r['over_refusal_rate']:>9.3f}"
              f"{r['balanced_accuracy']:>8.3f}{r['to_llm']:>6}")
    print(f"\n  주의: 53건 기준이므로 1건이 FAR 0.030 / 과잉거절 0.050 을 움직인다.")
    print(f"        표에서 이보다 작은 차이는 통계적으로 구별되지 않는다.")

    # --- 경계 확인 ---
    warns = boundary_warning(best)
    print("\n" + "=" * 76)
    print("그리드 경계 확인")
    print("=" * 76)
    if warns:
        for w in warns:
            print(f"  [경고] {w}")
    else:
        print("  최적 조합이 모든 축에서 그리드 내부에 있다. 범위는 충분하다.")

    # --- 권장값 ---
    if flat:
        mid = sorted(flat, key=lambda r: (r["hard_top1"], r["hard_gap15"], r["blank_ratio"]))
        rec = mid[len(mid) // 2]
    else:
        rec = best

    print("\n" + "=" * 76)
    print("권장 임계값")
    print("=" * 76)
    print(f"  HARD_REFUSE_TOP1      = {rec['hard_top1']}")
    print(f"  HARD_REFUSE_GAP15     = {rec['hard_gap15']}")
    print(f"  BLANK_RATIO_THRESHOLD = {rec['blank_ratio']}")
    print(f"\n  예상 성능 ({mode})")
    print(f"    FAR              {rec['false_answer_rate']:.3f}  ({rec['fn']}/{rec['fn'] + rec['tp']})")
    print(f"    과잉거절          {rec['over_refusal_rate']:.3f}  ({rec['fp']}/{rec['fp'] + rec['tn']})")
    print(f"    Balanced Accuracy {rec['balanced_accuracy']:.3f}")
    print(f"    LLM 호출          {rec['to_llm']}건 ({rec['to_llm'] / rec['n']:.0%})")
    print(f"\n  적용:")
    print(f"    python scripts/28_tune_refusal_v2.py --apply "
          f"{rec['hard_top1']},{rec['hard_gap15']},{rec['blank_ratio']}")

    # --- 권장값에서의 오답 목록 ---
    refusal_mod.HARD_REFUSE_TOP1 = rec["hard_top1"]
    refusal_mod.HARD_REFUSE_GAP15 = rec["hard_gap15"]
    refusal_mod.BLANK_RATIO_THRESHOLD = rec["blank_ratio"]
    final, _ = run_pipeline(cached, pattern_hits, verdicts)

    fa = [r for r in final if r["expected"] == "refuse" and r["predicted"] == "answer"]
    orf = [r for r in final if r["expected"] == "answer" and r["predicted"] == "refuse"]

    if fa:
        print(f"\n  [False Answer {len(fa)}건]")
        for r in fa:
            print(f"    [{r['refusal_type']:<13}] {r['question'][:44]}")
            print(f"       top1={r['score_top1']:.3f} gap15={r['gap_1_5']:.3f} "
                  f"blank={r['blank_ratio']:.2f} stage={r['stage']}")
    if orf:
        print(f"\n  [과잉 거절 {len(orf)}건]")
        for r in orf:
            print(f"    {r['question'][:48]}")
            print(f"       stage={r['stage']} reason={r['reason']} "
                  f"top1={r['score_top1']:.3f}")

    # --- 유형별 ---
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in final:
        if r["expected"] == "refuse":
            by_type[r["refusal_type"]].append(r)
    print("\n  [거절 유형별 정확도]")
    for t in sorted(by_type):
        g = by_type[t]
        hit = sum(1 for r in g if r["predicted"] == "refuse")
        print(f"    {t:<16} {hit}/{len(g)} = {hit / len(g):.3f}")

    # --- 저장 ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "rules" if args.rules_only else "full"
    out = REPORT_DIR / f"refusal_tuning_v2_{tag}_{date.today().isoformat()}.csv"
    add_cost(results, 2.0, LAMBDA_LLM)
    results.sort(key=lambda r: (r["cost"], -r["balanced_accuracy"]))
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n저장 → {out.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
