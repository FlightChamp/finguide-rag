"""65 — v2-B 개발 진단: 놓친 교란의 원인 · Identity baseline · 쉬움 지표.

64번 결과(2026-10-15)에서 확인할 것 세 가지
-----------------------------------------
1. 놓친 교란 9건이 왜 놓쳤는가
   - HEDGE_DROP 3/8, CONDITION_DROP 4/7, NUM_COMPARATOR 1/11, INTENSIFIER 1/51
   - full(조건 검사 LLM 포함)과 rules 의 놓침이 똑같았다 → LLM 조건 검사가 교란에서 추가로 잡은 것이 0건.
     LLM 이 '보존'이라며 댄 문구(where)를 보여 준다. 그 문구에 정말 조건이 남아 있으면 교란 라벨 잡음,
     없으면 LLM 조건 검사의 놓침이다. → 이 판단은 사람이 한다.
2. Identity baseline (의료 평이화 연구의 원칙 1 차용)
   OK claim 원문을 그대로 '쉬운 설명'으로 넣는다. 역검증은 전부 PASS 여야 한다.
   PASS 가 아니면 평가기 자체의 문제(자기 비교 실패)다. 또 Identity 는 '안전하지만 쉽지 않은' 기준점이다.
3. 쉬움 지표 (안전과 따로 잰다)
   문장당 어절 수, 원문 복사 유사도, 길이 비율. 역검증만 보면 '원문 그대로 옮기기'가 최적이 되므로,
   Simplifier 가 실제로 쉽게 만들었는지 같이 본다. 길이 비율 상위 사례는 '내용 창작' 의심 목록으로 낸다.

역검증 판정은 바꾸지 않는다 (진단 전용). 1·2는 64번 캐시를 재사용한다.
Identity 의 조건 검사만 새 텍스트라 API 를 쓴다 (기반 수만큼, 약 51콜). --offline 이면 규칙만.

기반 쉬운 설명의 역검증 결과는 출력하지 않는다 — 사람 검수(b_dev_review_v0.csv)를 결과 없이 하기 위해서다.

실행
----
    python scripts/65_b_dev_diagnose.py --offline
    python scripts/65_b_dev_diagnose.py
산출물
-----
    reports/v2_b_dev_diagnose_<날짜>.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
import sys
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
from finguide_rag.verification.numeric import extract_numerics  # noqa: E402
from finguide_rag.verification.rules import HEDGE_GROUPS, _hedges  # noqa: E402
from finguide_rag.verification.schema import (  # noqa: E402
    BVCode,
    BVStatus,
    ReasonCode,
    ReasonHit,
    SimpleItem,
    SimplifiedExplanation,
    Source,
    Verdict,
    Verification,
)

V2 = PROJECT_ROOT / "data" / "eval" / "v2"
INTERIM = PROJECT_ROOT / "data" / "interim"
CATALOG = INTERIM / "product_catalog.json"
BASE = V2 / "b_dev_base_v0.jsonl"
BV_CACHE = INTERIM / "v2_bv_cache.json"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL = "gpt-4.1-mini"
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def seed_of(*parts) -> int:          # 64번과 같은 시드 → 같은 교란
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
    return ctx, explanation_from_json(row["explanation"])


def identity(expl: SimplifiedExplanation, ctx: BVContext) -> SimplifiedExplanation:
    """OK claim 원문 = 핵심 설명. 조건 블록은 비운다 (조건은 claim 원문 안에 이미 있다).

    10-15 첫 실행에서는 조건 원문 조각을 조건 블록에 그대로 넣었는데, 조각 문구 검사(SENTENCE_FRAGMENT)를
    추가한 뒤로는 그게 형식 오류가 된다. 원문만 넣는 것이 'Identity' 의 뜻에도 맞다.
    """
    core = [SimpleItem(f"s{i}", c.text, [c.claim_id]) for i, c in enumerate(ctx.ok_claims(), 1)]
    return SimplifiedExplanation(core=core, conditions=[], staff_check=expl.staff_check,
                                 excluded=expl.excluded, required_conditions=expl.required_conditions,
                                 prompt_hash="identity")


def bigrams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def simplicity(text: str, source: str) -> dict:
    sents = [s for s in _SENT.split(text) if s.strip() and s.strip() != "꼭 확인하세요"]
    eoj = [len(s.split()) for s in sents]
    bt, bs = bigrams(text), bigrams(source)
    return {"eojeol_per_sent": st.mean(eoj) if eoj else 0.0,
            "copy_sim": len(bt & bs) / len(bt | bs) if bt | bs else 0.0,       # 2-gram Jaccard
            "len_ratio": len(re.sub(r"\s+", "", text)) / max(1, len(re.sub(r"\s+", "", source))),
            "paren": text.count("(")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="API 없이 (Identity 조건 검사 생략)")
    args = ap.parse_args()
    if not BASE.exists():
        print("63번을 먼저 실행하세요")
        return 1
    rows = [json.loads(x) for x in BASE.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if not r.get("generation_error")]
    matcher = ProductMatcher.from_catalog(CATALOG)
    known = [e.canonical_name for e in ProductCatalog.load(CATALOG).all_named if e.canonical_name]
    llm = CachedLLM(None if args.offline else openai_client(), MODEL, BV_CACHE)
    bv_full = BackVerifier(llm, matcher, known, condition_check=True)
    bv_rules = BackVerifier(None, matcher, known, condition_check=False)

    lines: list[str] = []

    def out(x=""):
        print(x)
        lines.append(x)

    out("=" * 76)
    out("v2-B 개발 진단   ※ 기반 쉬운 설명의 역검증 결과는 일부러 출력하지 않음")
    out("=" * 76)

    # ---------- 1. 놓친 교란 ----------
    out("\n[1] 놓친 교란 상세 (full 구성, 64번 캐시)")
    nmiss = 0
    for r in rows:
        ctx, expl = rebuild(r)
        products = [c.product for c in ctx.ok_claims() if c.product]
        for t in TYPES:
            p = perturb(expl, t, seed_of(r["b_id"], t), products=products, known_products=known, matcher=matcher,
                        claim_texts={c.claim_id: c.text for c in ctx.claims.values()})
            if p is None:
                continue
            pe, detail = p
            res = bv_full.verify(pe, ctx) if not args.offline else bv_rules.verify(pe, ctx)
            if res.status != BVStatus.PASS:
                continue
            nmiss += 1
            items = {it.item_id: it for it in pe.customer_items()}
            orig = {it.item_id: it for it in expl.customer_items()}
            iid = detail.split(":")[0].strip() if ":" in detail else ""
            out(f"\n  ■ {t}  {r['b_id']}  — {detail[:120]}")
            if iid in items:
                cited = items[iid].claim_ids
                out(f"    문구 전: {orig[iid].text if iid in orig else '-'}")
                out(f"    문구 후: {items[iid].text}")
                for cid in cited:
                    c = ctx.claims.get(cid)
                    if c:
                        out(f"    출처 {cid}: {c.text}")
                        out(f"      원문 헤지 묶음 {sorted(_hedges(c.text)) or '없음'}")
                if t == "HEDGE_DROP":
                    for cid in cited:
                        joined = [f"{it.item_id}: {it.text}" for it in pe.customer_items() if cid in it.claim_ids]
                        hit = [g for g, pat in HEDGE_GROUPS.items() if pat.search(" ".join(joined))]
                        out(f"      {cid} 를 인용한 문구들 (헤지 묶음 {hit or '없음'}, '수 있' 포함 여부로도 판단):")
                        for j in joined:
                            out(f"        {j}")
                if t in ("NUM_COMPARATOR", "NUM_VALUE", "NUM_UNIT"):
                    out(f"    추출된 수치(후): {[(s.raw, s.unit, s.comparator) for s in extract_numerics(items[iid].text)]}")
            if t == "CONDITION_DROP":
                kid = detail.split()[0]
                out(f"    LLM 판단: {res.condition_results.get(kid, '-')}  위치 {res.condition_where.get(kid, [])}")
                for w in res.condition_where.get(kid, []):
                    if w in items:
                        out(f"      {w}: {items[w].text}")
                out("    → 위 문구에 그 조건이 실제로 남아 있으면 '교란 라벨 잡음', 없으면 'LLM 놓침'")
    out(f"\n  놓친 교란 {nmiss}건")

    # ---------- 2. Identity baseline ----------
    out("\n[2] Identity baseline — OK claim 원문을 그대로 넣었을 때 (전부 PASS 여야 정상)")
    id_fail = {"rules": [], "full": []}
    id_frag = 0
    simp_base, simp_id = [], []
    for r in rows:
        ctx, expl = rebuild(r)
        ide = identity(expl, ctx)
        src = "\n".join(c.text for c in ctx.ok_claims())
        simp_base.append((r["b_id"], simplicity(expl.customer_text(), src)))
        simp_id.append(simplicity(ide.customer_text(), src))
        # Identity 는 '뜻 보존' 평가기 점검이다. 조각 문구(SENTENCE_FRAGMENT)는 A 의 claim 분해 단위
        # ('…진행하시고')에서 오는 형식 문제라 따로 센다 (10-15 2차 실행: 9/51 이 이것만으로 실패).
        rr = bv_rules.verify(ide, ctx)
        sem = [x for x in rr.problems if x.code != BVCode.SENTENCE_FRAGMENT]
        if len(sem) < len(rr.problems):
            id_frag += 1
        if sem:
            id_fail["rules"].append((r["b_id"], sorted({x.code.value for x in sem}), sem[:2]))
        if not args.offline:
            rf = bv_full.verify(ide, ctx)
            sem = [x for x in rf.problems if x.code != BVCode.SENTENCE_FRAGMENT]
            if sem:
                id_fail["full"].append((r["b_id"], sorted({x.code.value for x in sem}), sem[:2]))
    out(f"  (형식) 원문 claim 자체가 조각 문장인 기반 {id_frag}/{len(rows)} — A 의 claim 분해 단위 문제, 뜻 보존과 별개")
    for cfg in (["rules"] if args.offline else ["rules", "full"]):
        out(f"  {cfg}: 뜻 보존 문제 {len(id_fail[cfg])}/{len(rows)}   (0 이어야 정상)")
        for b, codes, probs in id_fail[cfg][:8]:
            out(f"    {b} {codes}  " + " / ".join(p.detail[:70] for p in probs))

    # ---------- 3. 쉬움 지표 ----------
    out("\n[3] 쉬움 지표 — Simplifier 출력 vs Identity (같은 OK claim 기준)")

    def agg(xs, k):
        v = [x[k] for x in xs]
        return f"평균 {st.mean(v):.2f} · 중앙 {st.median(v):.2f}"
    b = [x for _, x in simp_base]
    for k, name in (("eojeol_per_sent", "문장당 어절"), ("copy_sim", "원문 복사 유사도(2-gram Jaccard)"),
                    ("len_ratio", "길이 비율(원문 대비)"), ("paren", "괄호 수(용어 풀이 추정)")):
        out(f"  {name:<30} 쉬운 설명 {agg(b, k):<24} Identity {agg(simp_id, k)}")
    out("  해석: 복사 유사도가 Identity 값에 가까우면 '거의 안 바꿈', 길이 비율이 Identity 보다 훨씬 크면 '덧붙임' 의심\n        (Identity 는 OK claim 원문 그대로 → 복사 유사도 1.00, 길이 비율 1.00)")
    top = sorted(simp_base, key=lambda x: -x[1]["len_ratio"])[:6]
    out("\n  길이 비율 상위 (내용 창작·부연 의심 — 검수 때 특히 볼 것)")
    for bid, x in top:
        out(f"    {bid}  ×{x['len_ratio']:.2f}  복사 {x['copy_sim']:.2f}")

    out(f"\n조건 검사 API {llm.calls}콜 / 캐시 {llm.hits} · 토큰 {llm.tokens:,}")
    if not args.offline:
        llm.flush()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"v2_b_dev_diagnose_{date.today().isoformat()}.md"
    path.write_text("# v2-B 개발 진단\n\n```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
    print(f"→ {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
