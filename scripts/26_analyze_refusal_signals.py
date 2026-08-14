"""
26_analyze_refusal_signals.py

목적
----
거절 평가셋을 확정하고, 어떤 신호로 거절 여부를 판정할 수 있는지 분석한다.

핵심 질문
--------
"거절해야 할 질문과 답할 수 있는 질문이 검색 결과만으로 구별되는가?"

검색 평가에서 이미 절대 점수로는 어렵다는 것을 확인했다.

    top-1 점수: 성공 0.9053  vs  실패 0.8976  (차이 0.0077)
    1위-2위 격차: 성공 0.0120  vs  실패 0.0017  (7배)

거절 유형별로 신호가 다를 것으로 예상한다.

    out_of_scope   문서에 없으니 검색 점수가 낮을 것
    blank_value    문서를 정확히 찾아오므로 점수가 높을 것 -> 점수로는 못 잡음
    personalized   애매
    time_variant   애매

blank_value 가 확인되면 "단일 임계값으로는 불가능하고 다중 신호가 필요하다"는
근거가 된다. 이것이 이 분석의 목적이다.

측정하는 신호
-----------
score_top1      1위 유사도. 절대 신뢰도
gap_1_2         1위와 2위의 점수 차. 확신도
gap_1_5         1위와 5위의 점수 차. 결과 집중도
std_top5        상위 5개 점수의 표준편차
blank_ratio     상위 5개 청크 중 공란이 있는 비율
doc_concentration 상위 5개가 몇 개 문서에서 나왔는지

출력
----
data/eval/refusal_eval.csv          확정된 평가셋
reports/refusal_signals_{date}.md   신호 분석 리포트
reports/refusal_signals_{date}.csv  질의별 신호 원본

사용법
-----
    python scripts/26_analyze_refusal_signals.py
    python scripts/26_analyze_refusal_signals.py --alpha 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.embedding import MODELS, Embedder, FaissStore  # noqa: E402
from finguide_rag.retrieval import (  # noqa: E402
    BM25Store,
    FusionMethod,
    HybridRetriever,
    NormalizeMethod,
)

DRAFT_CSV = PROJECT_ROOT / "data" / "eval" / "refusal_eval_draft.csv"
FINAL_CSV = PROJECT_ROOT / "data" / "eval" / "refusal_eval.csv"
CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"
FAISS_ROOT = PROJECT_ROOT / "data" / "indexes" / "faiss"
BM25_ROOT = PROJECT_ROOT / "data" / "indexes" / "bm25"
REPORT_DIR = PROJECT_ROOT / "reports"

TOP_K = 10

FINAL_COLUMNS = [
    "query_id", "question", "expected", "refusal_type",
    "why_unanswerable", "source_chunk_id", "doc_display_name",
    "doc_type", "category",
]

# 계약 시 채워질 공란 패턴. 생성 스크립트와 동일하게 유지한다.
RE_BLANK = re.compile(r"\(\s*\)|\(\s*%\s*\)|（\s*）|_{3,}|\.{5,}|☐|□")


# ==================================================================
# 준비
# ==================================================================


def load_and_finalize() -> list[dict]:
    """검수를 마친 초안에서 keep=Y 만 추려 확정본을 만든다."""
    if not DRAFT_CSV.exists():
        sys.exit(f"{DRAFT_CSV} 없음. 먼저 24/25번 스크립트를 실행하세요.")

    with DRAFT_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    kept = [r for r in rows if r.get("keep", "Y").upper() != "N"]

    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FINAL_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FINAL_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(kept)
    except PermissionError:
        sys.exit(f"저장 실패: {FINAL_CSV.name} 이 열려 있습니다.")

    return kept


def load_chunks() -> dict[str, dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} 없음.")
    out = {}
    for line in CHUNKS_PATH.open(encoding="utf-8"):
        if line.strip():
            c = json.loads(line)
            out[c["chunk_id"]] = c
    return out


# ==================================================================
# 신호 계산
# ==================================================================


def compute_signals(hits, chunks: dict) -> dict:
    """검색 결과에서 거절 판정에 쓸 신호들을 뽑는다."""
    if not hits:
        return {
            "score_top1": 0.0, "gap_1_2": 0.0, "gap_1_5": 0.0,
            "std_top5": 0.0, "blank_ratio": 0.0, "doc_concentration": 0,
            "top1_doc": "", "top1_chunk": "",
        }

    scores = [h.score for h in hits]
    top5 = scores[:5]

    # 상위 5개 청크 중 공란이 있는 비율.
    # blank_value 유형은 검색이 성공하므로 점수로는 못 잡는다.
    # 대신 "찾아온 문서에 값이 비어 있다"는 것 자체가 신호가 될 수 있다.
    blank_count = 0
    for h in hits[:5]:
        text = chunks.get(h.chunk_id, {}).get("text", "")
        if RE_BLANK.search(text):
            blank_count += 1

    # 상위 5개가 몇 개 문서에서 나왔는지.
    # 한 문서에 몰리면 확신도가 높고, 흩어지면 검색이 헤맸다는 뜻이다.
    docs = {h.chunk_id.rsplit("_c", 1)[0] for h in hits[:5]}

    return {
        "score_top1": round(scores[0], 4),
        "gap_1_2": round(scores[0] - scores[1], 4) if len(scores) > 1 else 0.0,
        "gap_1_5": round(scores[0] - scores[4], 4) if len(scores) > 4 else 0.0,
        "std_top5": round(statistics.pstdev(top5), 4) if len(top5) > 1 else 0.0,
        "blank_ratio": round(blank_count / min(5, len(hits)), 2),
        "doc_concentration": len(docs),
        "top1_doc": hits[0].meta.get("doc_display_name", "")[:40],
        "top1_chunk": hits[0].chunk_id,
    }


def separability(a: list[float], b: list[float]) -> float:
    """두 집단이 얼마나 잘 나뉘는지. AUC 로 계산한다.

    0.5 는 전혀 구별되지 않음, 1.0 은 완벽히 구별됨을 뜻한다.
    a 의 값이 b 보다 클수록 1.0 에 가까워진다.

    표본이 적어 정규성 가정이 위험하므로, 모든 쌍을 비교하는
    비모수 방식(Mann-Whitney U 와 동치)을 쓴다.
    """
    if not a or not b:
        return 0.5
    wins = ties = 0
    for x in a:
        for y in b:
            if x > y:
                wins += 1
            elif x == y:
                ties += 1
    return (wins + 0.5 * ties) / (len(a) * len(b))


# ==================================================================
# 메인
# ==================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="e5-small", choices=list(MODELS))
    ap.add_argument("--index", default=None)
    ap.add_argument("--bm25", default="default")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    print("=" * 74)
    print("거절 평가셋 확정 및 신호 분석")
    print("=" * 74)

    rows = load_and_finalize()
    chunks = load_chunks()

    n_refuse = sum(1 for r in rows if r["expected"] == "refuse")
    n_answer = sum(1 for r in rows if r["expected"] == "answer")
    print(f"  확정 {len(rows)}건  (거절 {n_refuse} / 대조군 {n_answer})")
    print(f"  저장 → {FINAL_CSV.relative_to(PROJECT_ROOT)}")

    # --- 검색기 준비 ---
    faiss_dir = FAISS_ROOT / (args.index or args.model)
    bm25_dir = BM25_ROOT / args.bm25
    if not (faiss_dir / "config.json").exists():
        sys.exit(f"{faiss_dir} 에 인덱스가 없습니다.")
    if not (bm25_dir / "bm25.pkl").exists():
        sys.exit(f"{bm25_dir} 에 BM25 인덱스가 없습니다.")

    retriever = HybridRetriever(
        Embedder(args.model),
        FaissStore.load(faiss_dir),
        BM25Store.load(bm25_dir),
        method=FusionMethod.WEIGHTED,
        alpha=args.alpha,
        normalize=NormalizeMethod.MINMAX,
    )
    print(f"  검색 설정: weighted α={args.alpha}")

    # --- 검색 및 신호 수집 ---
    print(f"\n  {len(rows)}건 검색 중...")
    records: list[dict] = []
    for i, row in enumerate(rows, 1):
        hits = retriever.search(row["question"], top_k=TOP_K)
        sig = compute_signals(hits, chunks)
        records.append({
            "query_id": row["query_id"],
            "question": row["question"],
            "expected": row["expected"],
            "refusal_type": row["refusal_type"],
            **sig,
        })
        if i % 20 == 0:
            print(f"    {i}/{len(rows)}")

    # --- 집단별 비교 ---
    answerable = [r for r in records if r["expected"] == "answer"]
    refusable = [r for r in records if r["expected"] == "refuse"]

    signals = ["score_top1", "gap_1_2", "gap_1_5", "std_top5",
               "blank_ratio", "doc_concentration"]

    print("\n" + "=" * 74)
    print("신호별 분리도")
    print("=" * 74)
    print("  AUC 0.5 = 구별 불가, 1.0 = 완벽히 구별")
    print("  답변 가능한 질문이 거절 질문보다 값이 클수록 1.0 에 가깝다.")
    print()
    print(f"  {'신호':<20}{'답변가능':>10}{'거절':>10}{'차이':>10}{'AUC':>8}")
    print("  " + "-" * 58)

    auc_table: dict[str, float] = {}
    for sig in signals:
        a = [r[sig] for r in answerable]
        b = [r[sig] for r in refusable]
        ma = statistics.mean(a) if a else 0
        mb = statistics.mean(b) if b else 0
        auc = separability(a, b)
        auc_table[sig] = auc
        mark = ""
        if auc >= 0.70 or auc <= 0.30:
            mark = "  <-- 유효"
        print(f"  {sig:<20}{ma:>10.4f}{mb:>10.4f}{ma - mb:>+10.4f}{auc:>8.3f}{mark}")

    # --- 거절 유형별 ---
    print("\n" + "=" * 74)
    print("거절 유형별 신호")
    print("=" * 74)
    print("  유형마다 검색 결과의 성격이 다르므로 신호도 다르게 나타난다.")
    print()

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in refusable:
        by_type[r["refusal_type"]].append(r)

    print(f"  {'유형':<16}{'n':>4}{'top1':>9}{'gap1-2':>9}{'blank':>8}{'AUC*':>8}")
    print("  " + "-" * 54)
    a_top1 = [r["score_top1"] for r in answerable]
    for t in sorted(by_type):
        group = by_type[t]
        n = len(group)
        m_top1 = statistics.mean(r["score_top1"] for r in group)
        m_gap = statistics.mean(r["gap_1_2"] for r in group)
        m_blank = statistics.mean(r["blank_ratio"] for r in group)
        auc = separability(a_top1, [r["score_top1"] for r in group])
        print(f"  {t:<16}{n:>4}{m_top1:>9.4f}{m_gap:>9.4f}{m_blank:>8.2f}{auc:>8.3f}")
    print()
    print("  * AUC 는 top1 점수 기준. 0.5 에 가까울수록 점수로 구별할 수 없다는 뜻.")

    # --- 해석 ---
    print("\n" + "=" * 74)
    print("해석")
    print("=" * 74)

    best_sig = max(auc_table, key=lambda k: abs(auc_table[k] - 0.5))
    best_auc = auc_table[best_sig]
    print(f"  가장 변별력 있는 신호: {best_sig} (AUC {best_auc:.3f})")

    if abs(best_auc - 0.5) < 0.15:
        print("  단일 신호로는 거절 판정이 어렵다. 여러 신호를 조합하거나")
        print("  LLM 판단 같은 별도 장치가 필요하다.")
    else:
        print("  이 신호를 1차 필터로 쓸 수 있다.")

    blank_group = by_type.get("blank_value", [])
    if blank_group:
        b_auc = separability(a_top1, [r["score_top1"] for r in blank_group])
        print()
        print(f"  blank_value 유형의 top1 점수 AUC: {b_auc:.3f}")
        if abs(b_auc - 0.5) < 0.2:
            print("  예상대로 검색 점수로는 구별되지 않는다. 문서를 정확히 찾아오지만")
            print("  값이 비어 있는 경우이므로, 점수가 아니라 내용을 봐야 한다.")
            m = statistics.mean(r["blank_ratio"] for r in blank_group)
            m_ans = statistics.mean(r["blank_ratio"] for r in answerable)
            print(f"  공란 비율은 {m_ans:.2f}(답변가능) vs {m:.2f}(blank_value) 로")
            if m > m_ans + 0.15:
                print("  구별 가능성이 보인다. 공란 탐지를 별도 신호로 쓸 수 있다.")
            else:
                print("  차이가 크지 않다. 공란 탐지만으로는 부족하다.")

    # --- 저장 ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    csv_path = REPORT_DIR / f"refusal_signals_{stamp}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    md_path = REPORT_DIR / f"refusal_signals_{stamp}.md"
    lines = [
        "# 거절 신호 분석",
        "",
        f"- 실행일: {stamp}",
        f"- 평가셋: 거절 {n_refuse}건 / 대조군 {n_answer}건",
        f"- 검색: weighted α={args.alpha}",
        "",
        "## 배경",
        "",
        "근거 없는 답변은 금융 도메인에서 불완전판매로 직결된다. 검색 성능만",
        "높이면 관련 없는 문서를 자신 있게 근거로 제시하는 시스템이 된다.",
        "",
        "거절 여부를 무엇으로 판정할지 정하기 위해, 거절해야 할 질문과 답할 수",
        "있는 질문의 검색 결과가 어떻게 다른지 측정했다.",
        "",
        "## 신호별 분리도",
        "",
        "| 신호 | 답변 가능 | 거절 | 차이 | AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for sig in signals:
        a = [r[sig] for r in answerable]
        b = [r[sig] for r in refusable]
        ma, mb = statistics.mean(a), statistics.mean(b)
        lines.append(
            f"| `{sig}` | {ma:.4f} | {mb:.4f} | {ma - mb:+.4f} | {auc_table[sig]:.3f} |"
        )

    lines += [
        "",
        "AUC 0.5는 두 집단이 전혀 구별되지 않음을, 1.0은 완벽히 구별됨을 뜻한다.",
        "",
        "## 거절 유형별",
        "",
        "| 유형 | 건수 | top1 점수 | 1-2위 격차 | 공란 비율 | top1 AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t in sorted(by_type):
        g = by_type[t]
        lines.append(
            f"| `{t}` | {len(g)} | "
            f"{statistics.mean(r['score_top1'] for r in g):.4f} | "
            f"{statistics.mean(r['gap_1_2'] for r in g):.4f} | "
            f"{statistics.mean(r['blank_ratio'] for r in g):.2f} | "
            f"{separability(a_top1, [r['score_top1'] for r in g]):.3f} |"
        )
    lines.append(
        f"| *(대조군)* | {len(answerable)} | "
        f"{statistics.mean(a_top1):.4f} | "
        f"{statistics.mean(r['gap_1_2'] for r in answerable):.4f} | "
        f"{statistics.mean(r['blank_ratio'] for r in answerable):.2f} | – |"
    )

    lines += ["", "## 설계 방향", ""]
    if abs(best_auc - 0.5) < 0.15:
        lines.append("단일 신호로는 거절 판정이 어렵다. 여러 신호를 조합하거나 "
                     "LLM 판단을 병행해야 한다.")
    else:
        lines.append(f"`{best_sig}` (AUC {best_auc:.3f})를 1차 필터로 쓸 수 있다. "
                     "다만 유형별로 신호가 다르므로 단일 임계값으로는 부족하다.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n저장 → {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"저장 → {md_path.relative_to(PROJECT_ROOT)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
