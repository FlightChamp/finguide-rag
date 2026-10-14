"""67 — v2-B held-out: 생성 → 블라인드 검수 → freeze → (수정본 생성 → 블라인드 검수 → freeze) → 1회 공식 평가.

이 셋의 정확한 성격
-----------------
    A held-out v1 에서 사람 라벨이 Y 였던 초안 37건.
    "A 평가에서는 사용되었지만, B 의 simplifier/back-verifier 개발·프롬프트 조정·B 라벨링에는
     사용되지 않은 task-specific held-out set" 이다. 프로젝트 전체에서 처음 보는 데이터가 아니다.
    B 가 평가하는 대상(쉬운 설명이 OK claim 의 뜻을 보존했는가)과 그 라벨은 이 단계에서 새로 만든다.

규칙 (2026-10-15 확정)
--------------------
    1. 쉬운 설명을 새로 생성한다 (sp-v2).
    2. 역검증 결과를 숨긴 채 사람이 라벨링한다.
    3. 라벨을 freeze 하고 코드·프롬프트 지문을 기록한다.
    4. freeze 이후 코드 변경 없이 단 1회 공식 평가한다.
    5. 평가 결과를 본 뒤 B 프롬프트·규칙을 고치면, 이 37건은 그 시점부터 개발 셋이다.

수정본(1회 자동 수정) 평가 때문에 freeze 가 두 단계다
------------------------------------------------
    수정본은 1차 역검증에서 NEEDS_FIX 가 나온 건에만 생긴다. 1차 역검증을 돌리면 '어느 건이 걸렸는지'가
    드러나므로, 기반 라벨을 먼저 freeze 한 뒤(freeze-base) 수정본을 만든다(fixgen). 수정본은 2차 역검증 결과를
    모르는 상태로 전부 사람이 본다. 그다음 최종 freeze → eval.
    두 freeze 사이에 코드가 바뀌면 fixgen·freeze 가 거부한다.

통과 기준 (결과를 보기 전에 freeze 파일에 기록)
-------------------------------------------
    규칙·일반 교란 False Pass            ≤ 0.05
    조건 교란 False Pass (사람 valid=Y)  ≤ 0.20
    정상 쉬운 설명 과차단 (사람 Y)        ≤ 0.30
    수정 후 PASS 의 사람 기준 False Pass  = 0건
    가독성: 의미 보존(사람 Y · 역검증 PASS) 설명의 평균 문장당 어절 수 < Identity
            보조 기록: 전체 어절 수, 문장 수, Identity 대비 어절 수 감소율 (핵심 설명만 / 조건 블록 포함)
    모든 비율은 분자/분모(n/N)와 Wilson 95% 구간을 함께 남긴다.

실행 순서
--------
    python scripts/67_b_heldout.py generate          # 개발 조정 종료 확인 → 쉬운 설명·조건 교란 생성 (역검증 없음)
      → 사람: b_heldout_review_v1.csv (human_faithful) · b_heldout_condpert_review_v1.csv (valid)
    python scripts/67_b_heldout.py freeze-base
      → git commit
    python scripts/67_b_heldout.py fixgen            # 1차 역검증(결과 비공개) → NEEDS_FIX 건 수정본 생성
      → 사람: b_heldout_fix_review_v1.csv (fix_ok)
    python scripts/67_b_heldout.py freeze
      → git commit
    python scripts/67_b_heldout.py eval              # 공식 평가 1회. 두 번째부터, 또는 코드가 바뀌었으면 UNOFFICIAL
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import importlib.util
import json
import math
import random
import re
import statistics as stt
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.product_catalog import ProductCatalog  # noqa: E402
from finguide_rag.generation.product_match import ProductMatcher  # noqa: E402
from finguide_rag.verification.back_verify import (  # noqa: E402
    BackVerifier,
    BVContext,
    bv_prompt_hash,
    claims_from_json,
    explanation_from_json,
)
from finguide_rag.verification.bv_perturb import TYPES, perturb  # noqa: E402
from finguide_rag.verification.claim_verifier import (  # noqa: E402
    CachedLLM,
    ClaimVerifier,
    EvidenceChunk,
    prompt_hash,
)
from finguide_rag.verification.schema import (  # noqa: E402
    BVStatus,
    ReasonCode,
    ReasonHit,
    SimpleItem,
    SimplifiedExplanation,
    Source,
    Verdict,
    Verification,
)
from finguide_rag.verification.simplify import (  # noqa: E402
    SIMPLIFY_PROMPT_VERSION,
    Simplifier,
    partition,
    simplify_prompt_hash,
)

V2 = PROJECT_ROOT / "data" / "eval" / "v2"
INTERIM = PROJECT_ROOT / "data" / "interim"
REPORT_DIR = PROJECT_ROOT / "reports"
CATALOG = INTERIM / "product_catalog.json"
MODEL = "gpt-4.1-mini"

# 입력 (A held-out v1 — 이미 freeze 된 파일, 읽기만 한다)
A_SHEET = V2 / "heldout_v1_review.csv"
A_ANSWERS = INTERIM / "heldout_v1_answers.json"
DEV_FIX_REVIEW = V2 / "b_dev_fix_review_v0.csv"

# 산출물
BASE = V2 / "b_heldout_base_v1.jsonl"
REVIEW = V2 / "b_heldout_review_v1.csv"
CONDPERT = V2 / "b_heldout_condpert_v1.jsonl"
CONDPERT_REVIEW = V2 / "b_heldout_condpert_review_v1.csv"
FIXES = V2 / "b_heldout_fix_v1.jsonl"
FIX_REVIEW = V2 / "b_heldout_fix_review_v1.csv"
FREEZE_BASE = V2 / "b_heldout_v1_freeze_base.json"
FREEZE = V2 / "b_heldout_v1_freeze.json"
EVAL_LOG = V2 / "b_heldout_v1_eval_log.jsonl"
PACKET = INTERIM / "b_heldout_review_packet.html"          # 하나은행 원문 포함 → 커밋 금지 (.gitignore)
CV_CACHE = INTERIM / "v2_verifier_cache.json"
SP_CACHE = INTERIM / "v2_simplify_cache.json"
BV_CACHE = INTERIM / "v2_bv_cache.json"
CP_CACHE = INTERIM / "v2_condpert_cache.json"

SET_DESCRIPTION = ("A 평가에서는 사용되었지만, B 의 simplifier/back-verifier 개발·프롬프트 조정·B 라벨링에는 "
                   "사용되지 않은 task-specific held-out set (A held-out v1 의 사람 Y 초안)")
GATES = {
    "rule_perturb_false_pass_max": 0.05,
    "cond_perturb_false_pass_max": 0.20,
    "overblock_max": 0.30,
    "fix_path_false_pass_max_count": 0,
    "readability": "의미 보존(사람 Y · 역검증 PASS) 설명의 평균 문장당 어절 < Identity",
}
REVIEW_FIELDS = ["b_id", "doc_type", "question", "ok_claims", "required_conditions", "staff_check",
                 "customer_text", "text_sha", "human_faithful", "reasons", "memo"]
CP_FIELDS = ["cp_id", "b_id", "kind", "condition", "before", "after", "text_sha", "valid", "memo"]
FIX_FIELDS = ["b_id", "ok_claims", "before", "after", "text_sha", "fix_ok", "memo"]
REASONS_HELP = "NUMERIC · PRODUCT · INSTITUTION · STRENGTHENED · CONDITION · ADDED_INFO · CHECK_LEAK · OMITTED · OTHER"
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


# ==================================================================
# 공통
# ==================================================================


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, fields: list[str], rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm_sha(p: Path) -> str:
    """줄바꿈 차이(엑셀 저장 CRLF)를 무시한 내용 해시."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def text_sha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:10]


def openai_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(PROJECT_ROOT / ".env")
    return OpenAI()


def load_cp_module():
    """66번의 조건 교란 프롬프트를 그대로 쓴다 (개발 셋과 같은 생성 방식)."""
    spec = importlib.util.spec_from_file_location("cp66", PROJECT_ROOT / "scripts" / "66_b_cond_perturb.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code_fingerprint() -> dict:
    files = sorted((PROJECT_ROOT / "src" / "finguide_rag" / "verification").glob("*.py"))
    files += [PROJECT_ROOT / "src" / "finguide_rag" / "generation" / f
              for f in ("product_match.py", "product_catalog.py")]
    files += [PROJECT_ROOT / "scripts" / "67_b_heldout.py", PROJECT_ROOT / "scripts" / "66_b_cond_perturb.py",
              CATALOG]
    h = hashlib.sha256()
    per = {}
    for f in files:
        if f.exists():
            b = f.read_bytes().replace(b"\r\n", b"\n")
            h.update(f.name.encode() + b)
            per[f.name] = hashlib.sha256(b).hexdigest()[:10]
    cp = load_cp_module()
    return {"all": h.hexdigest()[:16], "files": per,
            "prompts": {"claim_verifier": prompt_hash("cv-v1"),
                        "simplifier": f"{SIMPLIFY_PROMPT_VERSION}:{simplify_prompt_hash()}",
                        "back_verifier": bv_prompt_hash(), "condpert": cp.cp_tag()}}


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if not n:
        return "-"
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n}={p:.2f} [{max(0, c - h):.2f}–{min(1, c + h):.2f}]"


def matcher_known():
    matcher = ProductMatcher.from_catalog(CATALOG)
    known = [e.canonical_name for e in ProductCatalog.load(CATALOG).all_named if e.canonical_name]
    return matcher, known


def rebuild(row: dict):
    claims = claims_from_json(row["claims"])
    vers = [Verification(cid, Verdict(v), [ReasonHit(ReasonCode(c), Source.RULE) for c in row["reasons"].get(cid, [])])
            for cid, v in row["verdicts"].items()]
    ctx = BVContext.build(claims, vers, row.get("evidence_institutions") or ["하나은행"])
    return claims, vers, ctx, explanation_from_json(row["explanation"])


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def labels_of(p: Path, key: str, col: str) -> dict[str, str]:
    return {r[key]: (r.get(col) or "").strip().upper() for r in read_csv(p)} if p.exists() else {}


# ==================================================================
# 1. generate
# ==================================================================


def dev_guard() -> None:
    """B 개발 조정 종료 조건: 개발 셋 수정본 사람 확인이 모두 Y."""
    if not DEV_FIX_REVIEW.exists():
        sys.exit(f"{DEV_FIX_REVIEW.name} 이 없습니다. 64번 --with-fix 를 먼저 실행하세요.")
    rows = read_csv(DEV_FIX_REVIEW)
    vals = [(r["b_id"], (r.get("fix_ok") or "").strip().upper()) for r in rows]
    empty = [b for b, v in vals if v not in ("Y", "N")]
    bad = [b for b, v in vals if v == "N"]
    if empty:
        sys.exit(f"개발 셋 수정본 확인이 끝나지 않았습니다 (빈칸 {len(empty)}건: {empty[:5]}). held-out 을 시작하지 않습니다.")
    if bad:
        sys.exit(f"개발 셋 수정본에 사람 기준 N 이 있습니다 {bad}. 수정 프롬프트를 고친 뒤 64번을 다시 실행하세요.")


def cmd_generate(args) -> int:
    if FREEZE_BASE.exists():
        sys.exit(f"{FREEZE_BASE.name} 이 있습니다. freeze 이후에는 다시 생성하지 않습니다.")
    if not args.skip_dev_guard:
        dev_guard()
    a_rows = [r for r in read_csv(A_SHEET) if r["human_label"] == "Y"]
    answers = json.loads(A_ANSWERS.read_text(encoding="utf-8"))
    matcher, known = matcher_known()
    client = None if args.dry_run else openai_client()
    cv = ClaimVerifier(CachedLLM(client, MODEL, CV_CACHE), matcher, known_products=known, prompt_version="cv-v1")
    sp_llm = CachedLLM(client, MODEL, SP_CACHE)
    simplifier = Simplifier(sp_llm)
    cp = load_cp_module()
    cp_llm = CachedLLM(client, MODEL, CP_CACHE)

    rows, no_ok, errors = [], [], []
    for r in a_rows:
        a = answers[r["question"]]
        chunks = [EvidenceChunk(e["chunk_id"], e.get("text", ""), e.get("citation", ""), e.get("doc_type", ""))
                  for e in a.get("evidences", [])]
        res = cv.verify(r["question"], r["answer"], chunks)
        if any(e in ("extract_failed", "align_failed") for e in res.errors):
            errors.append((r["draft_id"], "검증 실패"))
            continue
        ok, _, _, _ = partition(res.claims, res.verifications)
        if not ok:
            no_ok.append(r["draft_id"])
            continue
        expl, gen, _ = simplifier.simplify(r["question"], res.claims, res.verifications)
        if gen:
            errors.append((r["draft_id"], gen[0].detail))
            continue
        rows.append({
            "b_id": f"hb_{r['draft_id']}", "src_id": r["draft_id"], "doc_type": r["doc_type"],
            "question": r["question"], "claims": [c.to_dict() for c in res.claims],
            "verdicts": {v.claim_id: v.verdict.value for v in res.verifications},
            "reasons": {v.claim_id: [h.code.value for h in v.reasons] for v in res.verifications},
            "evidence_institutions": sorted({e.institution for v in res.verifications for e in v.evidence
                                             if e.institution}),
            "explanation": expl.to_dict(), "prompt": SIMPLIFY_PROMPT_VERSION,
        })

    # 조건 교란 (66번과 같은 생성기)
    cps = []
    for r in rows:
        expl = explanation_from_json(r["explanation"])
        if not expl.required_conditions:
            continue
        rng = random.Random(int(hashlib.sha256(r["b_id"].encode()).hexdigest()[:8], 16))
        k = rng.choice(expl.required_conditions)
        kind = "ALTER" if (rng.random() < 0.5 and not re.search(r"\d", k.text)) else "DROP"
        items_txt = "\n".join(f"{it.item_id}: {it.text}" for it in expl.customer_items())
        data = cp_llm.json(cp.CP_SYSTEM, cp.CP_USER.format(kind=kind, cond=k.text, items=items_txt),
                           max_tokens=1200, tag=cp.cp_tag())
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            continue
        e = copy.deepcopy(expl)
        by_id = {it.item_id: it for it in e.customer_items()}
        changed = []
        for x in data["items"]:
            if not isinstance(x, dict) or str(x.get("item_id")) not in by_id:
                continue
            it = by_id[str(x["item_id"])]
            new = re.sub(r"\s+", " ", str(x.get("text", ""))).strip()
            if not re.search(r"[가-힣A-Za-z0-9]", new):
                new = ""
            if new != it.text:
                changed.append({"item_id": it.item_id, "before": it.text, "after": new})
                it.text = new
        e.core = [it for it in e.core if it.text]
        e.conditions = [it for it in e.conditions if it.text]
        if not changed or not e.core:
            continue
        cps.append({"cp_id": f"hcp_{r['b_id']}_{k.cond_id}_{kind}", "b_id": r["b_id"], "kind": kind,
                    "cond_id": k.cond_id, "cond_text": k.text, "changed": changed, "explanation": e.to_dict()})

    print("=" * 72)
    print("v2-B held-out — 생성 (역검증은 돌리지 않음)")
    print("=" * 72)
    print(f"입력: {SET_DESCRIPTION}")
    print(f"A held-out v1 사람 Y {len(a_rows)}건 → 쉬운 설명 {len(rows)}건 · OK claim 없음 {len(no_ok)} · 오류 {len(errors)}")
    if errors:
        print(f"  오류: {errors[:5]}")
    print(f"조건 교란 {len(cps)}건 (DROP {sum(c['kind'] == 'DROP' for c in cps)} · ALTER {sum(c['kind'] == 'ALTER' for c in cps)})")
    print(f"API: 검증 {cv.llm.calls} · 쉬운 설명 {sp_llm.calls} · 교란 {cp_llm.calls}콜")
    if args.dry_run:
        print("[dry-run] 저장하지 않음")
        return 0
    for llm in (cv.llm, sp_llm, cp_llm):
        llm.flush()

    BASE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    CONDPERT.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cps) + "\n", encoding="utf-8")

    prev = {r["b_id"]: r for r in read_csv(REVIEW)} if REVIEW.exists() else {}
    review = []
    for r in rows:
        e = explanation_from_json(r["explanation"])
        cmap = {c["claim_id"]: c["text"] for c in r["claims"]}
        ok_ids = [k for k, v in r["verdicts"].items() if v == "OK"]
        ct = e.customer_text()
        old = prev.get(r["b_id"], {})
        same = old.get("text_sha") == text_sha(ct)
        review.append({"b_id": r["b_id"], "doc_type": r["doc_type"], "question": r["question"],
                       "ok_claims": "\n".join(f"{k}: {cmap[k]}" for k in ok_ids),
                       "required_conditions": "\n".join(f"{k.cond_id} ({k.claim_id}): {k.text}"
                                                        for k in e.required_conditions),
                       "staff_check": "\n".join(f"{s.claim_id}: {s.text}" for s in e.staff_check),
                       "customer_text": ct, "text_sha": text_sha(ct),
                       "human_faithful": old.get("human_faithful", "") if same else "",
                       "reasons": old.get("reasons", "") if same else "",
                       "memo": old.get("memo", "") if same else ""})
    write_csv(REVIEW, REVIEW_FIELDS, review)

    prevc = {r["cp_id"]: r for r in read_csv(CONDPERT_REVIEW)} if CONDPERT_REVIEW.exists() else {}
    crev = []
    for c in cps:
        before = "\n".join(f"{x['item_id']}: {x['before']}" for x in c["changed"])
        after = "\n".join(f"{x['item_id']}: {x['after'] or '(삭제)'}" for x in c["changed"])
        old = prevc.get(c["cp_id"], {})
        same = old.get("text_sha") == text_sha(after)
        crev.append({"cp_id": c["cp_id"], "b_id": c["b_id"], "kind": c["kind"], "condition": c["cond_text"],
                     "before": before, "after": after, "text_sha": text_sha(after),
                     "valid": old.get("valid", "") if same else "", "memo": old.get("memo", "") if same else ""})
    write_csv(CONDPERT_REVIEW, CP_FIELDS, crev)
    write_packet(review)
    print(f"\n→ {REVIEW.relative_to(PROJECT_ROOT)}  human_faithful: Y / N(+reasons: {REASONS_HELP}) / X")
    print(f"→ {CONDPERT_REVIEW.relative_to(PROJECT_ROOT)}  valid: Y(조건이 정말 사라짐·바뀜) / N")
    print(f"→ {PACKET.relative_to(PROJECT_ROOT)}  (읽기용 · 커밋 금지)")
    print("다음: 두 시트를 채운 뒤  python scripts/67_b_heldout.py freeze-base")
    return 0


def write_packet(rows: list[dict]) -> None:
    def esc(s):
        return html.escape(s).replace("\n", "<br>")
    parts = ["<!doctype html><meta charset='utf-8'><title>B held-out 검수</title>",
             "<style>body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px;line-height:1.55}"
             ".c{border:1px solid #ccc;border-radius:8px;padding:12px;margin:14px 0}"
             ".g{display:grid;grid-template-columns:1fr 1fr;gap:12px}.k{color:#666;font-size:13px}"
             ".s{background:#f6f6f6;padding:8px;border-radius:6px}.w{background:#fff6e0;padding:8px;border-radius:6px}"
             "@media(max-width:700px){.g{grid-template-columns:1fr}}</style>",
             "<h2>v2-B held-out 검수 — 고객에게 그대로 보여도 되는가</h2>",
             "<p class=k>Y = OK claim 의 뜻을 그대로 담았고, 직원 확인(CHECK) 내용이 섞이지 않았고, 조각 문구가 없다. "
             "역검증 결과는 넣지 않았다. 라벨은 b_heldout_review_v1.csv 에 적는다.</p>"]
    for r in rows:
        parts.append(f"<div class=c><b>{esc(r['b_id'])}</b> <span class=k>{esc(r['doc_type'])}</span>"
                     f"<p>질문: {esc(r['question'])}</p><div class=g>"
                     f"<div><div class=k>OK claim 원문</div><div class=s>{esc(r['ok_claims'])}</div>"
                     f"<div class=k>지켜야 할 조건</div><div class=s>{esc(r['required_conditions']) or '(없음)'}</div></div>"
                     f"<div><div class=k>고객용 쉬운 설명</div><div class=s>{esc(r['customer_text'])}</div>"
                     f"<div class=k>직원 확인 필요 (고객에게 안 보임)</div>"
                     f"<div class=w>{esc(r['staff_check']) or '(없음)'}</div></div></div></div>")
    PACKET.write_text("\n".join(parts), encoding="utf-8")


# ==================================================================
# 2. freeze-base
# ==================================================================


def require_complete(p: Path, col: str, allowed=("Y", "N", "X")) -> list[dict]:
    rows = read_csv(p)
    empty = [r.get("b_id") or r.get("cp_id") for r in rows if (r.get(col) or "").strip().upper() not in allowed]
    if empty:
        sys.exit(f"{p.name}: {col} 이 비었거나 허용값({'/'.join(allowed)})이 아닌 행 {len(empty)}건 {empty[:5]}")
    return rows


def cmd_freeze_base(args) -> int:
    if FREEZE_BASE.exists():
        sys.exit(f"{FREEZE_BASE.name} 이 이미 있습니다.")
    rev = require_complete(REVIEW, "human_faithful")
    cpr = require_complete(CONDPERT_REVIEW, "valid", ("Y", "N"))
    meta = {
        "stage": "base", "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "set_description": SET_DESCRIPTION, "gates": GATES,
        "files": {p.name: norm_sha(p) for p in (BASE, REVIEW, CONDPERT, CONDPERT_REVIEW)},
        "labels": {"review": dict(Counter((r["human_faithful"] or "").strip().upper() for r in rev)),
                   "condpert": dict(Counter((r["valid"] or "").strip().upper() for r in cpr))},
        "code": code_fingerprint(),
    }
    FREEZE_BASE.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"기반 라벨 freeze: {meta['labels']}  코드 지문 {meta['code']['all']}")
    print(f"통과 기준(사전 고정): {json.dumps(GATES, ensure_ascii=False)}")
    print(f"→ {FREEZE_BASE.relative_to(PROJECT_ROOT)}   지금 커밋하세요. 그다음 fixgen.")
    return 0


def check_frozen(meta: dict, names) -> list[str]:
    bad = []
    for n in names:
        p = V2 / n
        if not p.exists() or norm_sha(p) != meta["files"].get(n):
            bad.append(n)
    return bad


# ==================================================================
# 3. fixgen
# ==================================================================


def cmd_fixgen(args) -> int:
    if not FREEZE_BASE.exists():
        sys.exit("freeze-base 를 먼저 하세요.")
    if FREEZE.exists():
        sys.exit(f"{FREEZE.name} 이 있습니다. 최종 freeze 이후에는 수정본을 다시 만들지 않습니다.")
    meta = json.loads(FREEZE_BASE.read_text(encoding="utf-8"))
    bad = check_frozen(meta, meta["files"])
    if bad:
        sys.exit(f"freeze 이후 바뀐 파일: {bad}")
    if code_fingerprint()["all"] != meta["code"]["all"]:
        sys.exit("freeze-base 이후 코드가 바뀌었습니다. 수정본을 만들지 않습니다 (held-out 무효 방지).")
    matcher, known = matcher_known()
    client = None if args.dry_run else openai_client()
    bv_llm = CachedLLM(client, MODEL, BV_CACHE)
    bv = BackVerifier(bv_llm, matcher, known, condition_check=True)
    sp = Simplifier(CachedLLM(client, MODEL, SP_CACHE))
    fixes = []
    for r in load_jsonl(BASE):
        claims, vers, ctx, expl = rebuild(r)
        res = bv.verify(expl, ctx)
        if res.status != BVStatus.NEEDS_FIX:
            continue
        e2, gen = sp.fix(r["question"], expl, ctx.ok_claims(), res.problems)
        if gen:
            continue   # 생성 실패 → 운영 경로에서는 BLOCK. 고객에게 보일 일이 없으므로 사람 확인 대상 아님
        fixes.append({"b_id": r["b_id"], "explanation": e2.to_dict()})
    # 결과(어느 건이 PASS/BLOCK 인지)는 출력하지 않는다
    print(f"수정본 {len(fixes)}건 생성 (1차 역검증 상태·2차 결과는 비공개)")
    if args.dry_run:
        return 0
    bv_llm.flush()
    sp.llm.flush()
    FIXES.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in fixes) + "\n", encoding="utf-8")
    base = {r["b_id"]: r for r in load_jsonl(BASE)}
    rows = []
    for x in fixes:
        r = base[x["b_id"]]
        _, _, ctx, expl = rebuild(r)
        after = explanation_from_json(x["explanation"]).customer_text()
        rows.append({"b_id": x["b_id"], "ok_claims": "\n".join(f"{c.claim_id}: {c.text}" for c in ctx.ok_claims()),
                     "before": expl.customer_text(), "after": after, "text_sha": text_sha(after),
                     "fix_ok": "", "memo": ""})
    write_csv(FIX_REVIEW, FIX_FIELDS, rows)
    print(f"→ {FIX_REVIEW.relative_to(PROJECT_ROOT)}  fix_ok: Y(뜻 보존 · CHECK 유입 없음 · 조각 없음) / N")
    print("다음: 시트를 채운 뒤  python scripts/67_b_heldout.py freeze")
    return 0


# ==================================================================
# 4. freeze
# ==================================================================


def cmd_freeze(args) -> int:
    if not FREEZE_BASE.exists():
        sys.exit("freeze-base 를 먼저 하세요.")
    if FREEZE.exists():
        sys.exit(f"{FREEZE.name} 이 이미 있습니다.")
    base_meta = json.loads(FREEZE_BASE.read_text(encoding="utf-8"))
    bad = check_frozen(base_meta, base_meta["files"])
    if bad:
        sys.exit(f"freeze-base 이후 바뀐 파일: {bad}")
    fp = code_fingerprint()
    if fp["all"] != base_meta["code"]["all"]:
        sys.exit("freeze-base 이후 코드가 바뀌었습니다.")
    fix_rows = require_complete(FIX_REVIEW, "fix_ok", ("Y", "N")) if FIX_REVIEW.exists() and FIXES.exists() else []
    files = dict(base_meta["files"])
    for p in (FIXES, FIX_REVIEW):
        if p.exists():
            files[p.name] = norm_sha(p)
    meta = dict(base_meta, stage="final", frozen_at=datetime.now().isoformat(timespec="seconds"),
                base_frozen_at=base_meta["frozen_at"], files=files, code=fp)
    meta["labels"]["fix"] = dict(Counter((r["fix_ok"] or "").strip().upper() for r in fix_rows))
    FREEZE.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"최종 freeze: {meta['labels']}  코드 지문 {fp['all']}")
    print(f"→ {FREEZE.relative_to(PROJECT_ROOT)}   지금 커밋하세요. 그다음 eval (1회).")
    return 0


# ==================================================================
# 5. eval
# ==================================================================


def readability(text: str) -> dict:
    sents = [s.strip() for s in _SENT.split(text) if s.strip() and s.strip() != "꼭 확인하세요"]
    eoj = [len(s.lstrip("- ").split()) for s in sents]
    return {"sent": len(sents), "eojeol": sum(eoj), "per_sent": (sum(eoj) / len(sents)) if sents else 0.0}


def cmd_eval(args) -> int:
    if not FREEZE.exists():
        sys.exit("최종 freeze 가 없습니다.")
    meta = json.loads(FREEZE.read_text(encoding="utf-8"))
    bad = check_frozen(meta, meta["files"])
    if bad:
        sys.exit(f"freeze 이후 바뀐 파일: {bad} — 평가하지 않습니다.")
    fp = code_fingerprint()
    prior = load_jsonl(EVAL_LOG)
    official = (not prior) and fp["all"] == meta["code"]["all"]
    why = [] if official else (["이전 평가 있음"] if prior else []) + \
        (["freeze 이후 코드 변경"] if fp["all"] != meta["code"]["all"] else [])

    matcher, known = matcher_known()
    client = openai_client()
    bv_llm = CachedLLM(client, MODEL, BV_CACHE)
    bv = BackVerifier(bv_llm, matcher, known, condition_check=True)
    lab = labels_of(REVIEW, "b_id", "human_faithful")
    valid = labels_of(CONDPERT_REVIEW, "cp_id", "valid")
    fix_ok = labels_of(FIX_REVIEW, "b_id", "fix_ok")
    base_rows = load_jsonl(BASE)
    by_id = {r["b_id"]: r for r in base_rows}

    lines: list[str] = []
    detail: list[dict] = []

    def out(x=""):
        print(x)
        lines.append(x)

    # 기반
    base_res = {}
    for r in base_rows:
        _, _, ctx, expl = rebuild(r)
        base_res[r["b_id"]] = (bv.verify(expl, ctx), ctx, expl)
    ys = [b for b in base_res if lab.get(b) == "Y"]
    ns = [b for b in base_res if lab.get(b) == "N"]
    over = [b for b in ys if base_res[b][0].status != BVStatus.PASS]
    nat_fp = [b for b in ns if base_res[b][0].status == BVStatus.PASS]

    # 규칙 교란 (사람 Y 기반)
    pert = {t: [] for t in TYPES if t != "CONDITION_DROP"}
    for b in ys:
        res, ctx, expl = base_res[b]
        products = [c.product for c in ctx.ok_claims() if c.product]
        ct = {c.claim_id: c.text for c in ctx.claims.values()}
        for t in pert:
            seed = int(hashlib.sha256(f"{b}|{t}".encode()).hexdigest()[:8], 16)
            p = perturb(expl, t, seed, products=products, known_products=known, matcher=matcher, claim_texts=ct)
            if p is None:
                continue
            pr = bv.verify(p[0], ctx)
            pert[t].append((b, p[1], pr.status))
            detail.append({"id": b, "kind": t, "label": "PERTURBED", "status": pr.status.value, "detail": p[1]})
    allp = [x for xs in pert.values() for x in xs]
    rule_fp = [x for x in allp if x[2] == BVStatus.PASS]

    # 조건 교란 (사람 valid=Y, 기반 사람 Y)
    cps = [c for c in load_jsonl(CONDPERT) if valid.get(c["cp_id"]) == "Y" and lab.get(c["b_id"]) == "Y"]
    cond_res = []
    for c in cps:
        _, ctx, _ = base_res[c["b_id"]]
        e = explanation_from_json(c["explanation"])
        r = bv.verify(e, ctx)
        cond_res.append((c, r))
        detail.append({"id": c["cp_id"], "kind": "CONDPERT_" + c["kind"], "label": "Y", "status": r.status.value,
                       "detail": c["cond_text"]})
    cond_fp = [c for c, r in cond_res if r.status == BVStatus.PASS]

    # 수정 경로: 수정본 2차 역검증 → PASS 인데 사람 N 이면 False Pass
    fix_pass, fix_fp = [], []
    for x in load_jsonl(FIXES):
        _, ctx, _ = base_res[x["b_id"]]
        r2 = bv.verify(explanation_from_json(x["explanation"]), ctx, attempts=1)
        detail.append({"id": x["b_id"], "kind": "FIX", "label": fix_ok.get(x["b_id"], "-"),
                       "status": r2.status.value, "detail": ""})
        if r2.status == BVStatus.PASS:
            fix_pass.append(x["b_id"])
            if fix_ok.get(x["b_id"]) == "N":
                fix_fp.append(x["b_id"])

    # 가독성: 의미 보존(사람 Y · 역검증 PASS) 설명만
    good = [b for b in ys if base_res[b][0].status == BVStatus.PASS]
    rd = []
    for b in good:
        _, ctx, expl = base_res[b]
        ident = "\n".join(c.text for c in ctx.ok_claims())
        core = "\n".join(i.text for i in expl.core)
        rd.append({"full": readability(expl.customer_text()), "core": readability(core), "id": readability(ident)})

    def mean(key, part):
        v = [x[part][key] for x in rd]
        return stt.mean(v) if v else 0.0

    # ---------- 출력 ----------
    head = "공식 평가 (1회)" if official else f"UNOFFICIAL — {', '.join(why)}"
    out("=" * 78)
    out(f"v2-B held-out 평가 — {head}")
    out("=" * 78)
    out(f"셋: {SET_DESCRIPTION}")
    out(f"base freeze {meta['base_frozen_at']} · 최종 freeze {meta['frozen_at']} · 코드 {meta['code']['all']}"
        f" · 지금 코드 {fp['all']}")
    out(f"프롬프트: {json.dumps(meta['code']['prompts'], ensure_ascii=False)}")
    out(f"사람 라벨: 기반 {meta['labels']['review']} · 조건 교란 {meta['labels']['condpert']} · 수정본 {meta['labels'].get('fix', {})}")

    g = GATES
    rows_gate = []

    def gate(name, ok, value):
        rows_gate.append((name, ok, value))

    rule_rate = len(rule_fp) / len(allp) if allp else 0.0
    cond_rate = len(cond_fp) / len(cond_res) if cond_res else 0.0
    over_rate = len(over) / len(ys) if ys else 0.0
    gate(f"규칙·일반 교란 False Pass ≤ {g['rule_perturb_false_pass_max']}", rule_rate <= g["rule_perturb_false_pass_max"],
         wilson(len(rule_fp), len(allp)))
    gate(f"조건 교란 False Pass ≤ {g['cond_perturb_false_pass_max']}", cond_rate <= g["cond_perturb_false_pass_max"],
         wilson(len(cond_fp), len(cond_res)))
    gate(f"정상 쉬운 설명 과차단 ≤ {g['overblock_max']}", over_rate <= g["overblock_max"], wilson(len(over), len(ys)))
    gate("수정 후 PASS 의 사람 기준 False Pass = 0", len(fix_fp) <= g["fix_path_false_pass_max_count"],
         f"{len(fix_fp)}/{len(fix_pass)} {fix_fp}")
    gate("가독성: 평균 문장당 어절 < Identity (의미 보존 설명)", bool(rd) and mean("per_sent", "full") < mean("per_sent", "id"),
         f"{mean('per_sent', 'full'):.2f} vs {mean('per_sent', 'id'):.2f} (n={len(rd)}/{len(ys)})")

    out("\n[통과 기준 — freeze 때 고정]")
    for name, ok, value in rows_gate:
        out(f"  {'PASS' if ok else 'FAIL'}  {name:<46} {value}")

    out("\n[기반 쉬운 설명]")
    out(f"  과차단 (사람 Y 인데 PASS 아님)        {wilson(len(over), len(ys))}   {over}")
    for b in over:
        for p in base_res[b][0].problems:
            out(f"      {b} [{p.code.value}] {p.item_id or '-'} {p.detail[:80]}")
    out(f"  자연 오류 놓침 (사람 N 인데 PASS)      {wilson(len(nat_fp), len(ns))}   {nat_fp}")
    out(f"  상태 분포 {dict(Counter(v[0].status.value for v in base_res.values()))}")

    out("\n[규칙·일반 교란 (사람 Y 기반)]")
    for t, xs in pert.items():
        miss = [x for x in xs if x[2] == BVStatus.PASS]
        out(f"  {t:<16} 놓침 {wilson(len(miss), len(xs)):<26}" + (f" 예: {miss[0][0]} {miss[0][1][:50]}" if miss else ""))

    out("\n[조건 교란 (LLM 생성 · 사람 valid=Y)]")
    for kind in ("DROP", "ALTER"):
        xs = [(c, r) for c, r in cond_res if c["kind"] == kind]
        out(f"  {kind:<6} 놓침 {wilson(sum(r.status == BVStatus.PASS for _, r in xs), len(xs))}")
    for c in cond_fp:
        out(f"    놓침 {c['cp_id']} '{c['cond_text'][:40]}'")

    out("\n[1회 수정 경로]")
    out(f"  수정본 {len(load_jsonl(FIXES))}건 → 2차 역검증 PASS {len(fix_pass)}건 → 그중 사람 N {len(fix_fp)}건 {fix_fp}")

    out("\n[가독성 — 의미 보존(사람 Y · 역검증 PASS) 설명만, 평균]")
    out(f"  대상 {len(rd)}/{len(ys)}")
    out(f"  {'':<22}{'쉬운 설명(전체)':>16}{'핵심 설명만':>14}{'Identity':>12}")
    for key, name in (("per_sent", "문장당 어절 (주지표)"), ("eojeol", "전체 어절 수"), ("sent", "문장 수")):
        out(f"  {name:<22}{mean(key, 'full'):>16.2f}{mean(key, 'core'):>14.2f}{mean(key, 'id'):>12.2f}")
    if rd:
        red_full = 1 - sum(x["full"]["eojeol"] for x in rd) / max(1, sum(x["id"]["eojeol"] for x in rd))
        red_core = 1 - sum(x["core"]["eojeol"] for x in rd) / max(1, sum(x["id"]["eojeol"] for x in rd))
        out(f"  Identity 대비 어절 수 감소율: 전체 {red_full:+.1%} · 핵심 설명만 {red_core:+.1%}"
            "  (음수 = 늘어남. 조건 블록이 조건을 한 번 더 싣기 때문)")

    out(f"\n조건 검사 API {bv_llm.calls}콜 / 캐시 {bv_llm.hits}")
    bv_llm.flush()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = REPORT_DIR / f"v2_b_heldout_v1_eval_{date.today().isoformat()}"
    stem.with_suffix(".md").write_text("# v2-B held-out 평가\n\n```\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), ["id", "kind", "label", "status", "detail"], detail)
    with EVAL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "official": official, "why": why,
                            "code": fp["all"], "gates": {n: ok for n, ok, _ in rows_gate},
                            "values": {n: v for n, _, v in rows_gate}}, ensure_ascii=False) + "\n")
    print(f"→ {stem.relative_to(PROJECT_ROOT)}.md / .csv   (로그 {EVAL_LOG.relative_to(PROJECT_ROOT)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--skip-dev-guard", action="store_true", help="(비권장) 개발 셋 수정본 확인 완료 검사 생략")
    sub.add_parser("freeze-base")
    f = sub.add_parser("fixgen")
    f.add_argument("--dry-run", action="store_true")
    sub.add_parser("freeze")
    sub.add_parser("eval")
    args = ap.parse_args()
    return {"generate": cmd_generate, "freeze-base": cmd_freeze_base, "fixgen": cmd_fixgen,
            "freeze": cmd_freeze, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
