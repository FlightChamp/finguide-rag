"""64 — v2-B 역검증 개발 평가: 정상 쉬운 설명(사람 Y) + 교란 10종 + 1회 수정 경로.

개발 수치다. 공식 수치는 새 held-out(65번, 다음 단계)으로만 보고한다.

입력
----
    data/eval/v2/b_dev_base_v0.jsonl   (63번 산출물)
    data/eval/v2/b_dev_review_v0.csv   사람 라벨 (Y/N/X). 라벨이 없으면 과차단율은 '라벨 없음'

평가 대상과 지표
--------------
    정상 기반(사람 Y)        → PASS 가 아니면 과차단         Over-block Rate
    자연 오류(사람 N)        → PASS 면 놓침                 (자연) False Pass
    교란본(사람 Y 기반 × 10종) → PASS 면 놓침                 (교란) False Pass Rate  ← 핵심 안전 지표
        조건 보존 탐지율   = CONDITION_DROP 교란 중 PASS 아닌 비율
        수치 보존 탐지율   = NUM_VALUE · NUM_UNIT · NUM_COMPARATOR · BOUND_DROP
        상품 보존 탐지율   = PRODUCT_SWAP
    모든 비율은 1차 역검증(attempts=0) 기준이다. 수정 경로는 따로 센다 (--with-fix).

구성
----
    rules   결정적 규칙만 (LLM 0콜)
    full    규칙 + LLM 조건 보존 검사 (운영 경로). 교란본마다 1콜 (캐시)

실행
----
    python scripts/64_b_dev_eval.py --config rules
    python scripts/64_b_dev_eval.py --config full --dry-run   # 필요한 API 콜 수만
    python scripts/64_b_dev_eval.py --config full
    python scripts/64_b_dev_eval.py --config full --with-fix  # 정상·자연 기반 중 NEEDS_FIX 를 1회 수정해 재검증

산출물
-----
    reports/v2_b_dev_eval_<날짜>_<config>.md / .csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.product_catalog import ProductCatalog  # noqa: E402
from finguide_rag.generation.product_match import ProductMatcher  # noqa: E402
from finguide_rag.verification.back_verify import (  # noqa: E402
    BackVerifier,
    BVContext,
    claims_from_json,
    explanation_from_json,
)
from finguide_rag.verification.bv_perturb import TYPES, perturb  # noqa: E402
from finguide_rag.verification.claim_verifier import CachedLLM  # noqa: E402
from finguide_rag.verification.schema import (  # noqa: E402
    BVStatus,
    ReasonCode,
    ReasonHit,
    Source,
    Verdict,
    Verification,
)
from finguide_rag.verification.simplify import Simplifier  # noqa: E402

V2 = PROJECT_ROOT / "data" / "eval" / "v2"
INTERIM = PROJECT_ROOT / "data" / "interim"
CATALOG = INTERIM / "product_catalog.json"
BASE = V2 / "b_dev_base_v0.jsonl"
REVIEW = V2 / "b_dev_review_v0.csv"
BV_CACHE = INTERIM / "v2_bv_cache.json"
SP_CACHE = INTERIM / "v2_simplify_cache.json"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL = "gpt-4.1-mini"
GROUPS = {"수치": ["NUM_VALUE", "NUM_UNIT", "NUM_COMPARATOR", "BOUND_DROP"], "상품": ["PRODUCT_SWAP"],
          "조건": ["CONDITION_DROP"]}


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if not n:
        return "-"
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n}={p:.2f} [{max(0, c - h):.2f}–{min(1, c + h):.2f}]"


def seed_of(*parts) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def openai_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(PROJECT_ROOT / ".env")
    return OpenAI()


def rebuild(row: dict):
    claims = claims_from_json(row["claims"])
    vers = [Verification(cid, Verdict(v), [ReasonHit(ReasonCode(c), Source.RULE) for c in row["reasons"].get(cid, [])])
            for cid, v in row["verdicts"].items()]
    ctx = BVContext.build(claims, vers, row.get("evidence_institutions") or ["하나은행"])
    return claims, vers, ctx, explanation_from_json(row["explanation"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["rules", "full"], default="rules")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--with-fix", action="store_true")
    args = ap.parse_args()
    if not BASE.exists():
        print("63번을 먼저 실행하세요 (b_dev_base_v0.jsonl 없음)")
        return 1

    rows = [json.loads(x) for x in BASE.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if not r.get("generation_error")]
    labels = {r["b_id"]: r for r in csv.DictReader(REVIEW.open(encoding="utf-8-sig"))} if REVIEW.exists() else {}
    lab = {r["b_id"]: (labels.get(r["b_id"], {}).get("human_faithful") or "").strip().upper() for r in rows}
    labeled = any(v in ("Y", "N") for v in lab.values())

    matcher = ProductMatcher.from_catalog(CATALOG)
    known = [e.canonical_name for e in ProductCatalog.load(CATALOG).all_named if e.canonical_name]
    client = None if (args.dry_run or args.config == "rules") else openai_client()
    llm = CachedLLM(client, MODEL, BV_CACHE)
    bv = BackVerifier(llm if args.config == "full" else None, matcher, known,
                      condition_check=(args.config == "full"))

    lines: list[str] = []

    def out(x=""):
        print(x)
        lines.append(x)

    detail_rows = []
    base_res: dict[str, object] = {}
    pert: dict[str, list] = defaultdict(list)
    for r in rows:
        claims, vers, ctx, expl = rebuild(r)
        res = bv.verify(expl, ctx)
        base_res[r["b_id"]] = res
        detail_rows.append({"b_id": r["b_id"], "kind": "BASE", "label": lab[r["b_id"]] or "-",
                            "status": res.status.value, "codes": "|".join(sorted({p.code.value for p in res.problems})),
                            "detail": ""})
        if labeled and lab[r["b_id"]] != "Y":
            continue                         # 교란은 사람이 '정상'이라 한 기반에만 넣는다
        products = [c.product for c in ctx.ok_claims() if c.product]
        for t in TYPES:
            p = perturb(expl, t, seed_of(r["b_id"], t), products=products, known_products=known, matcher=matcher)
            if p is None:
                continue
            pres = bv.verify(p[0], ctx)
            pert[t].append((r["b_id"], p[1], pres))
            detail_rows.append({"b_id": r["b_id"], "kind": t, "label": "PERTURBED", "status": pres.status.value,
                                "codes": "|".join(sorted({x.code.value for x in pres.problems})), "detail": p[1]})

    if args.dry_run:
        print(f"[dry-run] 조건 검사 LLM 캐시 미스(=실행 시 API 콜) 약 {sum(1 for d in detail_rows if 'BV_ERROR' in d['codes'])}건")
        return 0

    out("=" * 76)
    out(f"v2-B 역검증 — 개발 평가 (config={args.config})   ※ 개발 수치. 공식은 새 held-out")
    out("=" * 76)
    ys = [b for b in base_res if lab[b] == "Y"]
    ns = [b for b in base_res if lab[b] == "N"]
    out(f"쉬운 설명 기반 {len(base_res)}건 · 사람 라벨 Y {len(ys)} / N {len(ns)} / 미검수·X {len(base_res) - len(ys) - len(ns)}")
    if not labeled:
        out("※ 사람 라벨이 아직 없다 — 과차단율은 계산하지 않고, 교란은 모든 기반에 넣었다.")

    out("\n[기반 쉬운 설명]")
    if ys:
        ob = [b for b in ys if base_res[b].status != BVStatus.PASS]
        out(f"  과차단 (사람 Y 인데 PASS 아님)   {wilson(len(ob), len(ys))}")
        c = Counter(p.code.value for b in ob for p in base_res[b].problems)
        out("    원인: " + (", ".join(f"{k} {v}" for k, v in c.most_common()) or "-"))
    if ns:
        fp = [b for b in ns if base_res[b].status == BVStatus.PASS]
        out(f"  자연 오류 놓침 (사람 N 인데 PASS) {wilson(len(fp), len(ns))}   {fp}")
    st = Counter(base_res[b].status.value for b in base_res)
    out(f"  상태 분포 {dict(st)}")

    out("\n[교란본 — False Pass (낮을수록 좋음) / 탐지]")
    allp = [x for t in TYPES for x in pert[t]]
    fpass = [x for x in allp if x[2].status == BVStatus.PASS]
    out(f"  전체 False Pass   {wilson(len(fpass), len(allp))}")
    for t in TYPES:
        xs = pert[t]
        miss = [x for x in xs if x[2].status == BVStatus.PASS]
        out(f"  {t:<16} 놓침 {wilson(len(miss), len(xs)):<26} " + (f"예: {miss[0][0]} {miss[0][1][:60]}" if miss else ""))
    out("\n[보존 탐지율 (= 1 − False Pass)]")
    for g, ts in GROUPS.items():
        xs = [x for t in ts for x in pert[t]]
        det = [x for x in xs if x[2].status != BVStatus.PASS]
        out(f"  {g} 보존  {wilson(len(det), len(xs))}")

    miss_cond = [x for x in pert["CONDITION_DROP"] if x[2].status == BVStatus.PASS]
    if miss_cond:
        out("\n[조건 삭제 교란 놓침 — 사람 확인 필요: 조건이 다른 문구에 다른 말로 남았을 수 있음]")
        for b, d, _ in miss_cond:
            out(f"  {b}: {d[:110]}")

    if args.with_fix and args.config == "full":
        out("\n[1회 수정 경로 — 기반 중 NEEDS_FIX]")
        sp = Simplifier(CachedLLM(client, MODEL, SP_CACHE))
        fixed = Counter()
        for r in rows:
            b = r["b_id"]
            if base_res[b].status != BVStatus.NEEDS_FIX:
                continue
            claims, vers, ctx, expl = rebuild(r)
            e2, gen = sp.fix(r["question"], expl, ctx.ok_claims(), base_res[b].problems)
            r2 = bv.verify(e2, ctx, attempts=1, generation_problems=gen)
            fixed[(lab[b] or "-", r2.status.value)] += 1
            detail_rows.append({"b_id": b, "kind": "FIX", "label": lab[b] or "-", "status": r2.status.value,
                                "codes": "|".join(sorted({x.code.value for x in r2.problems})),
                                "detail": e2.customer_text()[:400]})
        for (lb, s), n in sorted(fixed.items()):
            out(f"  사람={lb}  수정 후 {s}  {n}건")
        out("  ※ 수정본이 실제로 맞는지는 사람이 봐야 한다 (CSV 의 FIX 행 detail).")
        sp.llm.flush()

    out(f"\n조건 검사 API {llm.calls}콜 / 캐시 {llm.hits} · 토큰 {llm.tokens:,}")
    if args.config == "full":
        llm.flush()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = REPORT_DIR / f"v2_b_dev_eval_{date.today().isoformat()}_{args.config}"
    stem.with_suffix(".md").write_text("# v2-B 역검증 개발 평가\n\n```\n" + "\n".join(lines) + "\n```\n",
                                       encoding="utf-8")
    with stem.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["b_id", "kind", "label", "status", "codes", "detail"])
        w.writeheader()
        w.writerows(detail_rows)
    print(f"→ {stem.relative_to(PROJECT_ROOT)}.md / .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
