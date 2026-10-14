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
        조건 보존 탐지율   = 66번 조건 교란(LLM 생성 · 사람 확인 valid=Y) 중 PASS 아닌 비율  ← 주 지표 (10-15 변경)
                            규칙 교란 CONDITION_DROP 은 라벨 잡음이 커서 참고로만 본다
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
import re
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
          "조건(규칙 교란·참고)": ["CONDITION_DROP"]}


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


FIX_REVIEW = V2 / "b_dev_fix_review_v0.csv"
FIX_FIELDS = ["b_id", "base_label", "ok_claims", "problems_before", "before", "after", "text_sha", "fix_ok", "memo"]


def write_fix_review(fix_rows: list[dict], out) -> None:
    """수정 후 PASS 된 쉬운 설명을 사람이 확인하는 시트. fix_ok: Y(뜻 그대로 · 고객에게 보여도 됨) / N.

    같은 수정본이면 라벨을 보존한다 (text_sha). 라벨이 있으면 '수정 후 PASS 중 실제로 맞은 비율'을 낸다.
    """
    prev = {}
    if FIX_REVIEW.exists():
        prev = {x["b_id"]: x for x in csv.DictReader(FIX_REVIEW.open(encoding="utf-8-sig"))}
    for r in fix_rows:
        old = prev.get(r["b_id"], {})
        same = old.get("text_sha") == r["text_sha"]
        r["fix_ok"] = (old.get("fix_ok") or "").strip().upper() if same else ""
        r["memo"] = old.get("memo", "") if same else ""
    with FIX_REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIX_FIELDS)
        w.writeheader()
        w.writerows(fix_rows)
    done = [r for r in fix_rows if r["fix_ok"] in ("Y", "N")]
    if done:
        bad = [r["b_id"] for r in done if r["fix_ok"] == "N"]
        out(f"  수정 후 PASS 의 사람 확인: 맞음 {wilson(len(done) - len(bad), len(done))}   틀림 {bad}")
        out("  ※ 틀림이 있으면 그 건은 '수정 경로의 False Pass' — 1회 수정이 오류를 숨긴 것")
    out(f"  → {FIX_REVIEW.relative_to(PROJECT_ROOT)}  (수정 후 PASS {len(fix_rows)}건, 확인 {len(done)}건 — fix_ok 에 Y/N)")


CONDPERT = V2 / "b_dev_condpert_v0.jsonl"
CONDPERT_REVIEW = V2 / "b_dev_condpert_review_v0.csv"


def cond_section(rows, lab, labeled, bv, args, out, detail_rows) -> None:
    """66번 조건 교란 (LLM 생성 → 사람 확인 valid=Y) — 조건 보존 검사의 주 지표."""
    if not CONDPERT.exists():
        out("\n[조건 교란 (66번)] 없음 — 66번을 먼저 실행")
        return
    base_by_id = {r["b_id"]: r for r in rows}
    valid = {}
    if CONDPERT_REVIEW.exists():
        valid = {x["cp_id"]: (x.get("valid") or "").strip().upper()
                 for x in csv.DictReader(CONDPERT_REVIEW.open(encoding="utf-8-sig"))}
    cps = [json.loads(x) for x in CONDPERT.read_text(encoding="utf-8").splitlines() if x.strip()]
    confirmed = any(v in ("Y", "N") for v in valid.values())
    use = [c for c in cps if c["b_id"] in base_by_id
           and (valid.get(c["cp_id"]) == "Y" if confirmed else True)
           and (not labeled or lab.get(c["b_id"]) == "Y")]
    rules_only = BackVerifier(None, bv.matcher, bv.known_products, condition_check=False)
    res = []
    for c in use:
        _, _, ctx, _ = rebuild(base_by_id[c["b_id"]])
        e = explanation_from_json(c["explanation"])
        # 66번 cp-v1 생성기가 '삭제'를 빈 문자열 대신 '""' 로 낸 경우가 있다 (d005·q027·q082·q101).
        # 그대로 두면 조각 문구 규칙이 '""' 를 잡아 '규칙만' 탐지가 부풀려지므로 삭제로 처리한다.
        e.core = [it for it in e.core if re.search(r"[가-힣A-Za-z0-9]", it.text)]
        e.conditions = [it for it in e.conditions if re.search(r"[가-힣A-Za-z0-9]", it.text)]
        r_full = bv.verify(e, ctx) if bv.condition_check else None
        r_rule = rules_only.verify(e, ctx)
        res.append((c, r_full, r_rule))
        detail_rows.append({"b_id": c["b_id"], "kind": "CONDPERT_" + c["kind"], "label": valid.get(c["cp_id"]) or "-",
                            "status": (r_full or r_rule).status.value,
                            "codes": "|".join(sorted({x.code.value for x in (r_full or r_rule).problems})),
                            "detail": c["cond_text"]})
    head = "사람 확인 valid=Y" if confirmed else "※ 사람 확인 전 — 전체 교란 (라벨 잡음 포함)"
    out(f"\n[조건 교란 (66번, LLM 생성) — 조건 보존의 주 지표]  {head}")
    for kind in ("DROP", "ALTER", None):
        xs = [x for x in res if kind is None or x[0]["kind"] == kind]
        name = kind or "전체"
        miss_rule = [x for x in xs if x[2].status == BVStatus.PASS]
        line = f"  {name:<6} 규칙만 놓침 {wilson(len(miss_rule), len(xs)):<24}"
        if args.config == "full":
            miss_full = [x for x in xs if x[1].status == BVStatus.PASS]
            llm_only = [x for x in xs if x[2].status == BVStatus.PASS and x[1].status != BVStatus.PASS]
            line += f" full 놓침 {wilson(len(miss_full), len(xs)):<24} LLM 만 잡은 것 {len(llm_only)}"
        out(line)
    if args.config == "full":
        for c, rf, _ in res:
            if rf.status == BVStatus.PASS:
                out(f"    놓침 {c['cp_id']}  '{c['cond_text'][:40]}'  LLM {rf.condition_results.get(c['cond_id'], '-')}"
                    f" 위치 {rf.condition_where.get(c['cond_id'], [])}")


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
        claim_texts = {c.claim_id: c.text for c in claims}
        for t in TYPES:
            p = perturb(expl, t, seed_of(r["b_id"], t), products=products, known_products=known, matcher=matcher,
                        claim_texts=claim_texts)
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
        for b in ob:      # 과차단 사례: 사람 확인용 (어떤 문구·사유로 막혔는가)
            for p in base_res[b].problems:
                out(f"      {b} [{p.code.value}] {p.item_id or '-'} {p.detail[:90]}")
    if ns:
        fp = [b for b in ns if base_res[b].status == BVStatus.PASS]
        out(f"  자연 오류 놓침 (사람 N 인데 PASS) {wilson(len(fp), len(ns))}   {fp}")
        for b in fp:
            lb = labels.get(b, {})
            out(f"      {b} 사람 사유 {lb.get('reasons') or '-'}  메모 {(lb.get('memo') or '-')[:60]}")
    st = Counter(base_res[b].status.value for b in base_res)
    out(f"  상태 분포 {dict(st)}")

    out("\n[교란본 — False Pass (낮을수록 좋음) / 탐지]")
    # CONDITION_DROP(규칙 교란)은 라벨 잡음이 커서 전체 집계에서 뺀다 (10-15 진단). 조건은 66번 교란으로 본다.
    allp = [x for t in TYPES if t != "CONDITION_DROP" for x in pert[t]]
    fpass = [x for x in allp if x[2].status == BVStatus.PASS]
    out(f"  전체 False Pass (규칙 교란 9종, CONDITION_DROP 제외)   {wilson(len(fpass), len(allp))}")
    for t in TYPES:
        xs = pert[t]
        miss = [x for x in xs if x[2].status == BVStatus.PASS]
        out(f"  {t:<16} 놓침 {wilson(len(miss), len(xs)):<26} " + (f"예: {miss[0][0]} {miss[0][1][:60]}" if miss else ""))
    out("\n[보존 탐지율 (= 1 − False Pass)]")
    for g, ts in GROUPS.items():
        xs = [x for t in ts for x in pert[t]]
        det = [x for x in xs if x[2].status != BVStatus.PASS]
        out(f"  {g} 보존  {wilson(len(det), len(xs))}")

    cond_section(rows, lab, labeled, bv, args, out, detail_rows)

    miss_cond = [x for x in pert["CONDITION_DROP"] if x[2].status == BVStatus.PASS]
    if miss_cond:
        out("\n[규칙 교란 CONDITION_DROP 놓침 — 참고용 (10-15 진단: 대부분 조건이 핵심 설명에 남은 라벨 잡음)]")
        for b, d, _ in miss_cond:
            out(f"  {b}: {d[:110]}")

    if args.with_fix and args.config == "full":
        out("\n[1회 수정 경로 — 기반 중 NEEDS_FIX]")
        sp = Simplifier(CachedLLM(client, MODEL, SP_CACHE))
        fixed = Counter()
        fix_ids: dict = {}
        fix_rows: list[dict] = []
        for r in rows:
            b = r["b_id"]
            if base_res[b].status != BVStatus.NEEDS_FIX:
                continue
            claims, vers, ctx, expl = rebuild(r)
            e2, gen = sp.fix(r["question"], expl, ctx.ok_claims(), base_res[b].problems)
            r2 = bv.verify(e2, ctx, attempts=1, generation_problems=gen)
            fixed[(lab[b] or "-", r2.status.value)] += 1
            fix_ids.setdefault((lab[b] or "-", r2.status.value), []).append(b)
            detail_rows.append({"b_id": b, "kind": "FIX", "label": lab[b] or "-", "status": r2.status.value,
                                "codes": "|".join(sorted({x.code.value for x in r2.problems})),
                                "detail": e2.customer_text()[:400]})
            if r2.status == BVStatus.PASS:      # 수정 후 고객에게 보일 수 있는 것만 사람이 확인한다
                after = e2.customer_text()
                fix_rows.append({
                    "b_id": b, "base_label": lab[b] or "-",
                    "ok_claims": "\n".join(f"{c.claim_id}: {c.text}" for c in ctx.ok_claims()),
                    "problems_before": "\n".join(f"[{x.code.value}] {x.detail[:80]}" for x in base_res[b].problems),
                    "before": expl.customer_text(), "after": after,
                    "text_sha": hashlib.sha256(after.encode("utf-8")).hexdigest()[:10]})
        for (lb, s), n in sorted(fixed.items()):
            out(f"  사람={lb}  수정 후 {s}  {n}건   {fix_ids.get((lb, s), [])}")
        sp.llm.flush()
        write_fix_review(fix_rows, out)

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
