"""
22_remap_eval.py

목적
----
청킹 전략을 바꾼 뒤 평가셋의 정답 청크 ID를 새 청크에 다시 매핑한다.

왜 필요한가
---------
평가셋의 relevant_chunk_ids 는 특정 시점의 청크 ID를 가리킨다.

    q001, "기업 인터넷뱅킹에서...", hana_desc_digital_banking_001_c007

청킹 방식을 바꾸면 c007 이 다른 내용이 되거나 사라진다. 그대로 평가하면
Recall 이 0 에 가깝게 나오는데, 이것이 청킹 실패 때문인지 라벨이 어긋난
탓인지 구별할 수 없다. 개선 실험 자체가 불가능해진다.

매핑 방법
--------
기존 정답 청크의 본문을 새 청크들과 대조해, 텍스트가 겹치는 청크를 새
정답으로 삼는다. 같은 문서 안에서만 후보를 찾으므로 오매핑 위험이 낮다.

하나가 아니라 여럿을 인정하는 이유
-------------------------------
청킹 방식이 바뀌면 청크 크기가 달라진다. 기존 500자 청크가 새 청크
두 개에 걸치는 일이 흔하다. 가장 겹치는 것 하나만 정답으로 두면
검색이 나머지를 찾아왔을 때 오답으로 집계되어, 새 청킹 전략이
부당하게 낮게 평가된다.

따라서 임계값을 넘는 청크를 모두 인정하되, 최고 유사도의 절반에
못 미치는 것은 부분 겹침으로 보고 제외한다.

겹침 판정은 문자 3-gram 자카드 유사도를 쓴다. 형태소 분석보다 단순하지만
"같은 원문을 다르게 자른 것"을 찾는 데는 충분하고, 경계가 조금 달라져도
안정적으로 매칭된다.

사용법
-----
    # 새 청크 파일을 만든 뒤
    python scripts/22_remap_eval.py --new data/interim/chunks_article.jsonl

    # 결과 확인만
    python scripts/22_remap_eval.py --new ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
OLD_CHUNKS = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"

# 이 값 미만이면 매핑 실패로 본다.
# 낮추면 엉뚱한 청크가 정답으로 들어가고, 높이면 매핑 누락이 늘어난다.
MIN_SIMILARITY = 0.25

NGRAM = 3


def normalize(text: str) -> str:
    """비교용 정규화. 공백과 기호를 제거해 서식 차이를 흡수한다."""
    return re.sub(r"[^가-힣A-Za-z0-9]", "", text)


def ngrams(text: str, n: int = NGRAM) -> set[str]:
    t = normalize(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def similarity(a: set[str], b: set[str]) -> float:
    """자카드 유사도가 아니라 포함률을 쓴다.

    새 청크가 기존 청크보다 크거나 작을 수 있으므로, 교집합을 '더 작은
    쪽' 크기로 나눈다. 기존 청크가 새 청크에 온전히 포함되면 1.0 이 된다.
    자카드를 쓰면 크기 차이만으로 점수가 크게 떨어져 매핑을 놓친다.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} 없음.")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="새 청크 JSONL 경로")
    ap.add_argument("--old", default=str(OLD_CHUNKS), help="기존 청크 JSONL 경로")
    ap.add_argument("--eval", default=str(EVAL_CSV), help="평가셋 CSV 경로")
    ap.add_argument("--out", default=None, help="출력 경로 (기본: 평가셋 덮어쓰기)")
    ap.add_argument("--min-sim", type=float, default=MIN_SIMILARITY)
    ap.add_argument("--max-targets", type=int, default=3,
                    help="기존 청크 하나가 매핑될 수 있는 새 청크의 최대 개수")
    ap.add_argument("--dry-run", action="store_true", help="저장 없이 결과만 확인")
    args = ap.parse_args()

    eval_path = Path(args.eval)
    if not eval_path.exists():
        sys.exit(f"{eval_path} 없음.")

    old_chunks = {c["chunk_id"]: c for c in load_chunks(Path(args.old))}
    new_chunks = load_chunks(Path(args.new))

    # 문서별로 새 청크를 묶는다. 같은 문서 안에서만 후보를 찾는다.
    new_by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in new_chunks:
        new_by_doc[c["doc_id"]].append(c)

    # n-gram 은 한 번만 계산해 재사용한다
    new_grams = {c["chunk_id"]: ngrams(c["text"]) for c in new_chunks}

    with eval_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    print("=" * 72)
    print("평가셋 정답 라벨 재매핑")
    print("=" * 72)
    print(f"  기존 청크 {len(old_chunks):,}개 → 새 청크 {len(new_chunks):,}개")
    print(f"  평가셋 {len(rows)}건")
    print(f"  최소 유사도 {args.min_sim}")
    print()

    stats = {"mapped": 0, "expanded": 0, "failed": 0, "unchanged": 0}
    failures: list[tuple[str, str, float]] = []
    detail: list[dict] = []

    for row in rows:
        old_ids = [c.strip() for c in row["relevant_chunk_ids"].split("|") if c.strip()]
        new_ids: list[str] = []
        sims: list[float] = []

        for old_id in old_ids:
            old = old_chunks.get(old_id)
            if old is None:
                failures.append((row["query_id"], old_id, -1.0))
                stats["failed"] += 1
                continue

            # ID 가 그대로 살아 있고 내용도 같으면 유지
            same = next(
                (c for c in new_by_doc[old["doc_id"]] if c["chunk_id"] == old_id),
                None,
            )
            if same and normalize(same["text"]) == normalize(old["text"]):
                new_ids.append(old_id)
                sims.append(1.0)
                stats["unchanged"] += 1
                continue

            # 같은 문서의 새 청크 중 겹치는 것을 모두 찾는다.
            #
            # 하나만 고르면 안 되는 이유
            # -------------------------
            # 청킹 방식이 바뀌면 청크 크기가 달라진다. 기존 500자 청크가
            # 새 청크 두 개에 걸치는 일이 흔하다.
            #
            #   기존 c007 (500자) = 제6조 후반 + 제7조 전반
            #   신규 c009(제6조), c010(제7조)
            #
            # 가장 겹치는 것 하나만 정답으로 두면, 검색이 나머지를 찾아왔을 때
            # 오답으로 집계된다. 새 청킹 전략이 부당하게 낮게 평가된다.
            #
            # 따라서 임계값을 넘는 모든 청크를 정답으로 인정하되,
            # 최고 유사도 대비 지나치게 낮은 것은 제외해 오염을 막는다.
            old_g = ngrams(old["text"])
            scored: list[tuple[float, str]] = []
            for cand in new_by_doc.get(old["doc_id"], []):
                s = similarity(old_g, new_grams[cand["chunk_id"]])
                if s >= args.min_sim:
                    scored.append((s, cand["chunk_id"]))

            if not scored:
                # 최고 유사도를 실패 로그에 남긴다
                best = max(
                    (similarity(old_g, new_grams[c["chunk_id"]])
                     for c in new_by_doc.get(old["doc_id"], [])),
                    default=0.0,
                )
                failures.append((row["query_id"], old_id, best))
                stats["failed"] += 1
                continue

            scored.sort(reverse=True)
            top_sim = scored[0][0]

            # 최고 유사도의 절반에 못 미치는 것은 부분 겹침으로 보고 제외한다.
            # 이 조건이 없으면 스치듯 겹친 청크까지 정답이 되어 지표가 부풀려진다.
            accepted = [(s, cid) for s, cid in scored[:args.max_targets]
                        if s >= top_sim * 0.5]

            for s, cid in accepted:
                new_ids.append(cid)
                sims.append(s)
            stats["mapped"] += 1
            if len(accepted) > 1:
                stats["expanded"] += 1

        # 중복 제거하되 순서는 유지
        seen: set[str] = set()
        deduped = [i for i in new_ids if not (i in seen or seen.add(i))]

        detail.append({
            "query_id": row["query_id"],
            "old": old_ids,
            "new": deduped,
            "sims": sims,
        })
        row["relevant_chunk_ids"] = "|".join(deduped)

        # source_chunk_id 도 함께 갱신한다
        src = row.get("source_chunk_id", "").strip()
        if src in old_chunks:
            old_g = ngrams(old_chunks[src]["text"])
            best_id, best_sim = "", 0.0
            for cand in new_by_doc.get(old_chunks[src]["doc_id"], []):
                s = similarity(old_g, new_grams[cand["chunk_id"]])
                if s > best_sim:
                    best_id, best_sim = cand["chunk_id"], s
            if best_sim >= args.min_sim:
                row["source_chunk_id"] = best_id

    # --- 요약 ---
    total = stats["mapped"] + stats["failed"] + stats["unchanged"]
    n_labels = sum(len(d["new"]) for d in detail)
    print("  [매핑 결과]")
    print(f"    변경 없음     : {stats['unchanged']:>4}개")
    print(f"    재매핑        : {stats['mapped']:>4}개")
    print(f"      복수로 확장 : {stats['expanded']:>4}개 (기존 청크가 새 청크 여럿에 걸침)")
    print(f"    실패          : {stats['failed']:>4}개")
    print(f"    원본 라벨 수  : {total:>4}개")
    print(f"    최종 라벨 수  : {n_labels:>4}개  (질의당 평균 {n_labels / len(rows):.1f}개)")

    # 정답이 하나도 남지 않은 질의는 평가에서 무의미해진다
    empty = [d for d in detail if not d["new"]]
    if empty:
        print(f"\n  !! 정답이 모두 사라진 질의 {len(empty)}건")
        for d in empty[:8]:
            print(f"     {d['query_id']}: {' | '.join(d['old'])}")
        print("     이 질의들은 평가에서 항상 실패로 집계됩니다.")

    if failures:
        print(f"\n  [매핑 실패 상세] 상위 {min(8, len(failures))}건")
        for qid, old_id, sim in failures[:8]:
            reason = "기존 청크 없음" if sim < 0 else f"최고 유사도 {sim:.2f}"
            print(f"     {qid}  {old_id}  ({reason})")

    # 매핑 품질 표본
    changed = [d for d in detail if d["sims"] and min(d["sims"]) < 1.0]
    if changed:
        print(f"\n  [재매핑 표본] {min(5, len(changed))}건")
        for d in changed[:5]:
            pairs = ", ".join(
                f"{o} → {n} ({s:.2f})"
                for o, n, s in zip(d["old"], d["new"], d["sims"])
            )
            print(f"     {d['query_id']}: {pairs}")

    if args.dry_run:
        print("\n  --dry-run 이므로 저장하지 않았습니다.")
        return

    out_path = Path(args.out) if args.out else eval_path
    try:
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        sys.exit(f"\n저장 실패: {out_path.name} 이 열려 있습니다. 닫고 다시 실행하세요.")

    # 매핑 이력을 남긴다. 나중에 문제가 생기면 추적할 근거가 된다.
    log_path = out_path.parent / "remap_log.json"
    log_path.write_text(json.dumps({
        "old_chunks": args.old,
        "new_chunks": args.new,
        "min_similarity": args.min_sim,
        "max_targets": args.max_targets,
        "stats": stats,
        "empty_queries": [d["query_id"] for d in empty],
        "mappings": detail,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    def rel(path: Path) -> str:
        """프로젝트 루트 기준 상대경로. 밖에 있으면 원래 경로를 그대로 쓴다."""
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    print(f"\n저장 → {rel(out_path)}")
    print(f"저장 → {rel(log_path)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
