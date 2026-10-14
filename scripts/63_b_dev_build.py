"""63 — v2-B 개발 셋: 개발 셋 정상 초안 → (검증 캐시) → 쉬운 설명 생성 → 사람 검수 시트.

개발 셋만 쓴다 (새 held-out 은 읽지 않는다)
----------------------------------------
    정상 초안 25건 (clean_drafts_v1, Y)  +  개발 43건 중 사람=Y
    → ClaimVerifier(cv-v1) 결과는 LLM 캐시에서 읽는다 (54·58·59번을 돌렸다면 API 0콜)
    → OK claim 이 1개 이상인 초안만 Simplifier(sp-v1) 로 쉬운 설명을 만든다 (초안당 1콜, 캐시)

이 단계는 역검증을 돌리지 않는다. 검수할 때 역검증 결과를 보지 않게 하기 위해서다.

사람 검수 (개발용)
----------------
    data/eval/v2/b_dev_review_v0.csv 를 열어 각 행의 human_faithful 에
        Y  쉬운 설명이 OK claim 원문의 뜻을 그대로 담았다 (고객에게 보여도 된다)
        N  뜻이 바뀌었다 → reasons 에 하나 이상 (| 로 구분):
           NUMERIC · PRODUCT · INSTITUTION · STRENGTHENED · CONDITION · ADDED_INFO ·
           CHECK_LEAK · OMITTED · OTHER
        X  판단 불가
    를 적는다. memo 는 자유. 읽기 편한 화면: data/interim/b_dev_review_packet.html
    (하나은행 원문 문장이 들어 있으므로 커밋하지 않는다 — .gitignore 에 추가됨)
    다시 실행해도 이미 적은 라벨은 보존된다.

실행
----
    python scripts/63_b_dev_build.py --dry-run     # API 없이: 대상 수·캐시 상태만
    python scripts/63_b_dev_build.py               # 쉬운 설명 생성 (캐시 없으면 API 호출)

산출물
-----
    data/eval/v2/b_dev_base_v0.jsonl     입력(claim·판정)과 쉬운 설명 1차 출력
    data/eval/v2/b_dev_review_v0.csv     사람 검수 시트
    data/interim/b_dev_review_packet.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.product_catalog import ProductCatalog  # noqa: E402
from finguide_rag.generation.product_match import ProductMatcher  # noqa: E402
from finguide_rag.verification.back_verify import explanation_from_json  # noqa: E402
from finguide_rag.verification.claim_verifier import CachedLLM, ClaimVerifier, EvidenceChunk  # noqa: E402
from finguide_rag.verification.schema import Verdict  # noqa: E402
from finguide_rag.verification.simplify import Simplifier, partition  # noqa: E402

V2 = PROJECT_ROOT / "data" / "eval" / "v2"
INTERIM = PROJECT_ROOT / "data" / "interim"
ANSWERS = INTERIM / "g_full_answers.json"
CATALOG = INTERIM / "product_catalog.json"
CV_CACHE = INTERIM / "v2_verifier_cache.json"
SP_CACHE = INTERIM / "v2_simplify_cache.json"
BASE = V2 / "b_dev_base_v0.jsonl"
REVIEW = V2 / "b_dev_review_v0.csv"
PACKET = INTERIM / "b_dev_review_packet.html"
MODEL = "gpt-4.1-mini"
REVIEW_FIELDS = ["b_id", "group", "question", "ok_claims", "required_conditions", "staff_check",
                 "customer_text", "human_faithful", "reasons", "memo"]


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def dev_items() -> list[dict]:
    out = []
    for r in read_csv(V2 / "clean_drafts_v1.csv"):
        if r["human_is_correct"] == "Y":
            out.append({"id": r["draft_id"], "group": "정상25", "question": r["question"], "text": r["draft_text"]})
    for r in read_csv(V2 / "heldout_review_v0.csv"):          # 이제 개발 셋 (2026-10-14 결정)
        if r["human_is_correct"] == "Y":
            out.append({"id": r["draft_id"], "group": "개발43Y", "question": r["question"], "text": r["draft_text"]})
    return out


def openai_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(PROJECT_ROOT / ".env")
    return OpenAI()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 대상 수와 캐시 상태만")
    args = ap.parse_args()

    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    matcher = ProductMatcher.from_catalog(CATALOG)
    known = [e.canonical_name for e in ProductCatalog.load(CATALOG).all_named if e.canonical_name]
    cv = ClaimVerifier(CachedLLM(None, MODEL, CV_CACHE), matcher, known_products=known,
                       prompt_version="cv-v1")                       # 검증은 캐시만
    sp_llm = CachedLLM(None if args.dry_run else openai_client(), MODEL, SP_CACHE)
    simplifier = Simplifier(sp_llm)

    rows, skipped_cache, skipped_no_ok, failed = [], [], [], []
    for it in dev_items():
        chunks = [EvidenceChunk(e["chunk_id"], e.get("text", ""), e.get("citation", ""), e.get("doc_type", ""))
                  for e in (answers.get(it["question"]) or {}).get("evidences", [])]
        res = cv.verify(it["question"], it["text"], chunks)
        if res.errors and any(e in ("extract_failed", "align_failed") for e in res.errors):
            skipped_cache.append(it["id"])
            continue
        ok, check, proh, _ = partition(res.claims, res.verifications)
        if not ok:
            skipped_no_ok.append(it["id"])
            continue
        expl, gen, _ = simplifier.simplify(it["question"], res.claims, res.verifications)
        if gen:
            failed.append((it["id"], gen[0].detail))
        inst = sorted({e.institution for v in res.verifications for e in v.evidence if e.institution})
        rows.append({
            "b_id": f"b_{it['id']}", "src_id": it["id"], "group": it["group"], "question": it["question"],
            "claims": [c.to_dict() for c in res.claims],
            "verdicts": {v.claim_id: v.verdict.value for v in res.verifications},
            "reasons": {v.claim_id: [h.code.value for h in v.reasons] for v in res.verifications},
            "evidence_institutions": inst,
            "explanation": expl.to_dict(),
            "generation_error": gen[0].detail if gen else "",
        })

    print("=" * 70)
    print("v2-B 개발 셋 — 쉬운 설명 1차 출력")
    print("=" * 70)
    print(f"대상 {len(rows) + len(skipped_cache) + len(skipped_no_ok)}건 → 쉬운 설명 {len(rows)}건")
    print(f"  OK claim 없음(쉬운 설명 대상 아님) {len(skipped_no_ok)}건")
    if skipped_cache:
        print(f"  ※ 검증 캐시 없음 {len(skipped_cache)}건 {skipped_cache[:6]} — 54·58번(cv-v1)을 먼저 돌릴 것")
    n_ok = sum(sum(v == Verdict.OK.value for v in r["verdicts"].values()) for r in rows)
    n_check = sum(sum(v == Verdict.CHECK.value for v in r["verdicts"].values()) for r in rows)
    n_cond = sum(len(r["explanation"]["required_conditions"]) for r in rows)
    print(f"  OK claim {n_ok} · 직원 확인(CHECK) {n_check} · 지켜야 할 조건 {n_cond}")
    print(f"  Simplifier API {sp_llm.calls}콜 / 캐시 {sp_llm.hits} · 토큰 {sp_llm.tokens:,}")
    if failed:
        print(f"  생성 실패 {len(failed)}건: {failed[:4]}")
    if args.dry_run:
        missing = sum(1 for r in rows if r["generation_error"])
        print(f"\n[dry-run] 캐시에 없는 쉬운 설명 {missing}건 — 실제 실행 시 약 {missing}콜")
        return 0
    sp_llm.flush()

    BASE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    prev = {r["b_id"]: r for r in read_csv(REVIEW)} if REVIEW.exists() else {}
    out = []
    for r in rows:
        e = r["explanation"]
        ok_ids = [k for k, v in r["verdicts"].items() if v == "OK"]
        cmap = {c["claim_id"]: c["text"] for c in r["claims"]}
        ex = explanation_from_json(e)
        old = prev.get(r["b_id"], {})
        out.append({
            "b_id": r["b_id"], "group": r["group"], "question": r["question"],
            "ok_claims": "\n".join(f"{k}: {cmap[k]}" for k in ok_ids),
            "required_conditions": "\n".join(f"{k['cond_id']} ({k['claim_id']}): {k['text']}"
                                             for k in e["required_conditions"]),
            "staff_check": "\n".join(f"{s['claim_id']}: {s['text']}" for s in e["staff_check"]),
            "customer_text": ex.customer_text() if not r["generation_error"] else f"(생성 실패: {r['generation_error']})",
            "human_faithful": old.get("human_faithful", ""), "reasons": old.get("reasons", ""),
            "memo": old.get("memo", ""),
        })
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        w.writerows(out)
    write_packet(out)
    done = sum(1 for r in out if r["human_faithful"] in ("Y", "N", "X"))
    print(f"\n→ {BASE.relative_to(PROJECT_ROOT)}")
    print(f"→ {REVIEW.relative_to(PROJECT_ROOT)}   (검수 {done}/{len(out)})")
    print(f"→ {PACKET.relative_to(PROJECT_ROOT)}   (읽기용 · 커밋 금지)")
    return 0


def write_packet(rows: list[dict]) -> None:
    def esc(s):
        return html.escape(s).replace("\n", "<br>")
    parts = ["<!doctype html><meta charset='utf-8'><title>B 개발 셋 검수</title>",
             "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px;line-height:1.55}"
             ".c{border:1px solid #ccc;border-radius:8px;padding:12px;margin:14px 0}"
             ".g{display:grid;grid-template-columns:1fr 1fr;gap:12px}.k{color:#666;font-size:13px}"
             ".s{background:#f6f6f6;padding:8px;border-radius:6px}.w{background:#fff6e0;padding:8px;border-radius:6px}"
             "@media(max-width:700px){.g{grid-template-columns:1fr}}</style>",
             "<h2>v2-B 개발 셋 검수 — 쉬운 설명이 OK claim 원문의 뜻을 그대로 담았는가</h2>",
             "<p class=k>역검증 결과는 일부러 넣지 않았다. 라벨은 b_dev_review_v0.csv 에 적는다. "
             "Y=그대로 / N=뜻이 바뀜(reasons) / X=판단 불가</p>"]
    for r in rows:
        parts.append(f"<div class=c><b>{esc(r['b_id'])}</b> <span class=k>{esc(r['group'])}</span>"
                     f"<p>질문: {esc(r['question'])}</p><div class=g>"
                     f"<div><div class=k>OK claim 원문</div><div class=s>{esc(r['ok_claims'])}</div>"
                     f"<div class=k>지켜야 할 조건</div><div class=s>{esc(r['required_conditions']) or '(없음)'}</div></div>"
                     f"<div><div class=k>고객용 쉬운 설명</div><div class=s>{esc(r['customer_text'])}</div>"
                     f"<div class=k>직원 확인 필요 (고객에게 안 보임)</div>"
                     f"<div class=w>{esc(r['staff_check']) or '(없음)'}</div></div></div></div>")
    PACKET.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
