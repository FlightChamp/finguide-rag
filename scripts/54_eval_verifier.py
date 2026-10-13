"""54 — ClaimVerifier(LLM + 규칙) 를 오류 주입 평가셋에서 평가하고, 3가지 구성을 비교한다.

비교 구성 (추가 API 호출 없이 한 번의 실행에서 모두 계산)
------------------------------------------------
    rules     규칙 사유 코드만 사용 (LLM 코드는 SUPPORTED 로 간주). 근거 span 은 LLM 정렬 결과
    llm       LLM 코드만 사용 (규칙 사유 코드 무시)
    combined  실제 시스템. LLM 코드 + 규칙 사유 코드 → policy
53번(규칙층 + proxy 정렬)과 다른 점: 여기서는 claim 분해와 근거 정렬을 LLM 이 한다.

지표 (주입 샘플)
--------------
    주입 위치(span)를 포함한 claim 을 '대상 claim' 으로 본다.
    탐지(Detected)      대상 claim 판정이 OK 가 아님
    심각도(Severity)    대상 claim 판정 ≥ 기대 최소 판정 (예: 수치 변경은 PROHIBITED 여야 함)
    코드(Code)          기대 사유 코드가 발급됨
    미정렬(Unaligned)    대상 claim 을 찾지 못함 (claim 이 원문 부분 문자열이 아니었거나 해당 문장이 claim 이 안 됨)
                        → 탐지 실패로 센다
지표 (대조군 NONE)
--------------
    Clean FA (hard)     claim 중 PROHIBITED 비율 — '틀렸다'고 잘못 막은 것
    Clean FA (soft)     claim 중 OK 가 아닌 비율 — CHECK 은 정당할 수도 있음 (claim 단위 사람 라벨 전)
    초안 PROHIBITED     정상 초안 중 최종 PROHIBITED 비율
기타: LLM–Rule Conflict Rate, VERIFIER_ERROR 비율, 호출 수·토큰

실행
----
    python scripts/54_eval_verifier.py --selftest     # API 없이 가짜 LLM 으로 배선·fail-closed 점검
    python scripts/54_eval_verifier.py --dry-run      # 예상 호출 수만 출력
    python scripts/54_eval_verifier.py --limit 10     # 앞 10건만 (시험)
    python scripts/54_eval_verifier.py                # 전체 120건 (캐시 없으면 약 240콜)

산출물
-----
    reports/v2_verifier_eval_<날짜>.md / .csv
    data/interim/v2_verifier_results.jsonl   샘플별 원자료 (claim·근거·사유 코드)
    data/interim/v2_verifier_cache.json      LLM 캐시
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.product_match import ProductMatcher  # noqa: E402
from finguide_rag.verification.claim_verifier import (  # noqa: E402
    CachedLLM,
    ClaimVerifier,
    EvidenceChunk,
    prompt_hash,
)
from finguide_rag.verification.policy import build_verification  # noqa: E402
from finguide_rag.verification.schema import ReasonCode, Verdict  # noqa: E402

INJECTED = PROJECT_ROOT / "data" / "eval" / "v2" / "injected_v0.csv"
NATURAL = PROJECT_ROOT / "data" / "eval" / "v2" / "natural_errors_v0.csv"   # 56번 산출 (있으면 함께 평가)
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
CATALOG = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"
CACHE = PROJECT_ROOT / "data" / "interim" / "v2_verifier_cache.json"
RAW_OUT = PROJECT_ROOT / "data" / "interim" / "v2_verifier_results.jsonl"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL = "gpt-4.1-mini"
SEV = {"OK": 0, "CHECK": 1, "PROHIBITED": 2}
CONFIGS = ("rules", "llm", "combined")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(k: int, n: int) -> str:
    if not n:
        return "  -"
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}–{hi:.2f}]"


def chunks_for(answers: dict, question: str) -> list[EvidenceChunk]:
    return [EvidenceChunk(chunk_id=e["chunk_id"], text=e.get("text", ""),
                          doc_name=e.get("citation", ""), doc_type=e.get("doc_type", ""))
            for e in (answers.get(question) or {}).get("evidences", [])]


def variant_verdicts(res, cid: str) -> dict[str, tuple[str, list[str]]]:
    """같은 claim 에 대해 rules / llm / combined 판정과 사유 코드."""
    rule_hits = res.rule_hits.get(cid, [])
    llm = res.llm_codes.get(cid, [])
    v_rules = build_verification(cid, ["SUPPORTED"], rule_hits)
    v_llm = build_verification(cid, llm, [])
    v_comb = next(v for v in res.verifications if v.claim_id == cid)
    return {k: (v.verdict.value, [h.code.value for h in v.reasons])
            for k, v in (("rules", v_rules), ("llm", v_llm), ("combined", v_comb))}


def target_claims(res, a: int, b: int) -> list[str]:
    """주입 위치를 포함하는 claim. 삽입(길이 0)은 그 위치를 포함하는 claim."""
    hit = []
    for c in res.claims:
        s, e = c.span
        if s < 0:
            continue
        if (b > a and s < b and a < e) or (b == a and s <= a < max(e, s + 1)):
            hit.append(c.claim_id)
    return hit


# ==================================================================
# 가짜 LLM (selftest)
# ==================================================================


class _FakeCompletions:
    """초안을 마침표로 잘라 claim 으로, 정렬은 모두 E1 SUPPORTED 로 답한다.
    mode 로 실패 상황을 흉내 낸다."""

    def __init__(self, mode: str):
        self.mode = mode

    def create(self, **kw):
        sys_msg = kw["messages"][0]["content"]
        user = kw["messages"][1]["content"]
        if self.mode == "bad_json":
            content = "{not json"
        elif sys_msg.startswith("너는 금융 상담 답변 초안"):
            draft = user.split("직원 답변 초안:\n", 1)[1]
            import re as _re   # 소수점(3.0%)에서 자르지 않는다
            sents = [m.group(0).strip() for m in _re.finditer(r"(?:[^.]|\.(?=\d))+\.?", draft)
                     if len(m.group(0).strip()) > 3]
            content = json.dumps({"claims": [{"text": s if self.mode != "paraphrase" else s + " 요약",
                                              "product": None, "event": None, "conditions": [],
                                              "needs_customer_data": False} for s in sents]},
                                 ensure_ascii=False)
        else:
            claim_lines = user.split("[claim]\n")[1].split("\n\n")[0].splitlines()
            ev_lines = user.split("[근거 문장]\n")[1].splitlines()
            bg = lambda t: {t[j:j + 2] for j in range(len(t) - 1)}   # noqa: E731
            code = "OK" if self.mode == "label" else "SUPPORTED"
            results = []
            for ln in claim_lines:
                cid, text = ln.split(": ", 1)
                if self.mode == "drop" and cid == "c1":
                    continue
                best = max(ev_lines, key=lambda e: len(bg(text) & bg(e)))   # 가장 겹치는 근거 1개
                results.append({"claim_id": cid,
                                "evidence_ids": [] if self.mode == "no_ev" else [best.split(":")[0]],
                                "code": code, "missing_conditions": [], "fix": None})
            content = json.dumps({"results": results})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def fake_client(mode: str):
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(mode)))


def selftest() -> int:
    matcher = ProductMatcher.from_catalog(CATALOG)
    ch = [EvidenceChunk("hana_desc_x_c001", "중도해지 시 우대금리는 적용되지 않습니다. 연 3.0%입니다.",
                        doc_name="『테스트』 상품설명서", doc_type="설명서")]
    q = "중도해지하면 우대금리는?"
    draft = "중도해지 시 우대금리는 적용되지 않습니다. 금리는 연 3.0%입니다."
    checks = []

    def run(mode, d=draft):
        return ClaimVerifier(CachedLLM(fake_client(mode)), matcher).verify(q, d, ch)

    r = run("ok")
    checks.append(("정상: claim 2개, 전부 OK, 최종 OK",
                   len(r.claims) == 2 and r.final.final.value == "OK"))
    checks.append(("근거 quote 는 우리 원문 문장 그대로",
                   r.verifications[0].evidence[0].quote == "중도해지 시 우대금리는 적용되지 않습니다."))
    checks.append(("claim span 이 초안 위치와 일치",
                   draft[r.claims[1].span[0]:r.claims[1].span[1]] == r.claims[1].text))
    r = run("ok", "중도해지 시 우대금리는 적용되지 않습니다. 금리는 연 3.5%입니다.")
    checks.append(("LLM SUPPORTED 여도 규칙 수치 불일치 → PROHIBITED, 충돌 기록",
                   r.verifications[1].verdict == Verdict.PROHIBITED and r.verifications[1].llm_rule_conflict))
    r = run("label")
    checks.append(("LLM 이 'OK' 라벨을 냄 → VERIFIER_ERROR, CHECK",
                   all(v.verdict == Verdict.CHECK for v in r.verifications)))
    r = run("drop")
    checks.append(("정렬 결과에서 c1 누락 → c1 만 CHECK",
                   r.verifications[0].verdict == Verdict.CHECK and r.verifications[1].verdict == Verdict.OK))
    r = run("no_ev")
    checks.append(("SUPPORTED 인데 근거 id 없음 → OK 금지",
                   all(v.verdict != Verdict.OK for v in r.verifications)))
    r = run("bad_json")
    checks.append(("LLM JSON 깨짐 → 대체 claim 1개, CHECK, 오류 기록",
                   len(r.claims) == 1 and r.final.final.value == "CHECK" and "extract_failed" in r.errors))
    r = run("paraphrase")
    checks.append(("claim 이 원문 부분 문자열 아님 → span -1, 오류 기록",
                   r.claims[0].span == (-1, -1) and any("not_substring" in e for e in r.errors)))
    llm = CachedLLM(fake_client("ok"))
    v = ClaimVerifier(llm, matcher)
    v.verify(q, draft, ch)
    v.verify(q, draft, ch)
    checks.append(("같은 입력 두 번 → 두 번째는 캐시 (호출 2, 적중 2)", llm.calls == 2 and llm.hits == 2))

    print("=" * 60)
    print("54 selftest (가짜 LLM, API 호출 없음)")
    print("=" * 60)
    for name, ok in checks:
        print(f"  {'✔' if ok else '✘'} {name}")
    fails = sum(not ok for _, ok in checks)
    print(f"\n{len(checks)}건 중 실패 {fails}건   prompt_hash={prompt_hash()}")
    return 1 if fails else 0


# ==================================================================
# 본 평가
# ==================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    with INJECTED.open(encoding="utf-8-sig", newline="") as f:
        samples = list(csv.DictReader(f))
    if args.limit:
        samples = samples[: args.limit]
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    matcher = ProductMatcher.from_catalog(CATALOG)

    if args.dry_run:
        llm = CachedLLM(None, MODEL, CACHE)
        cv = ClaimVerifier(llm, matcher)
        for s in samples:
            cv.verify(s["question"], s["text"], chunks_for(answers, s["question"]))
        need = 2 * len(samples) - llm.hits
        print(f"샘플 {len(samples)}건 → 캐시 적중 {llm.hits}콜, 새로 필요한 호출 약 {need}콜")
        print("(캐시에 없는 정렬 호출은 추출 결과에 따라 달라져 정확히 셀 수 없음 — 상한 추정)")
        return 0

    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(PROJECT_ROOT / ".env")
    llm = CachedLLM(OpenAI(), MODEL, CACHE)
    cv = ClaimVerifier(llm, matcher)

    stats = {k: defaultdict(lambda: [0, 0, 0, 0]) for k in CONFIGS}   # n, det, sev, code
    clean = {k: [0, 0, 0] for k in CONFIGS}                            # claims, soft, hard
    clean_draft = [0, 0]
    unaligned = defaultdict(int)
    conflicts = total_claims = verr = 0
    rows = []
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

    with RAW_OUT.open("w", encoding="utf-8") as raw:
        for i, s in enumerate(samples, 1):
            res = cv.verify(s["question"], s["text"], chunks_for(answers, s["question"]))
            if i % 10 == 0:
                llm.flush()
                print(f"  {i}/{len(samples)}  (API {llm.calls}콜, 캐시 {llm.hits})")
            raw.write(json.dumps({"sample_id": s["sample_id"], **res.to_dict()}, ensure_ascii=False) + "\n")
            total_claims += len(res.claims)
            conflicts += sum(v.llm_rule_conflict for v in res.verifications)
            verr += sum(any(h.code == ReasonCode.VERIFIER_ERROR for h in v.reasons) for v in res.verifications)

            if s["error_type"] == "NONE":
                clean_draft[0] += 1
                clean_draft[1] += res.final.final.value == "PROHIBITED"
                for c in res.claims:
                    vv = variant_verdicts(res, c.claim_id)
                    for k in CONFIGS:
                        clean[k][0] += 1
                        clean[k][1] += vv[k][0] != "OK"
                        clean[k][2] += vv[k][0] == "PROHIBITED"
                continue

            key = f"{s['error_type']}/{s['subtype']}"
            if s["subtype"] == "intensifier":
                key += "(목록 " + ("안)" if s["in_detector_vocab"] == "Y" else "밖)")
            expected = [c for c in s["expected_codes"].split("|") if c]
            tids = target_claims(res, int(s["span_start"]), int(s["span_end"]))
            if not tids:
                unaligned[key] += 1
            row = {"sample_id": s["sample_id"], "type": key,
                   "injected": f"{s['original_fragment']}→{s['injected_fragment']}",
                   "targets": "|".join(tids)}
            for k in CONFIGS:
                vs = [variant_verdicts(res, t)[k] for t in tids]
                worst = max((SEV[v] for v, _ in vs), default=0)
                codes = {c for _, cs in vs for c in cs}
                st = stats[k][key]
                st[0] += 1
                st[1] += worst > 0
                st[2] += worst >= SEV[s["expected_min_verdict"]]
                st[3] += any(c in codes for c in expected)
                row[f"{k}_verdict"] = ["OK", "CHECK", "PROHIBITED"][worst]
                row[f"{k}_codes"] = "|".join(sorted(codes))
            rows.append(row)
    llm.flush()

    # ---------- 출력 ----------
    lines: list[str] = []

    def out(x=""):
        print(x)
        lines.append(x)

    prov = sum(1 for s in samples if s["label_status"] != "confirmed")
    out("=" * 92)
    out(f"ClaimVerifier 평가 — 오류 주입 v0  (model={MODEL}, prompt={prompt_hash()})")
    out("=" * 92)
    if prov:
        out(f"※ 사람 검수 전 기반 초안 포함 {prov}/{len(samples)}건 — 잠정치")
    out(f"\n[탐지율]  {'유형':<30}" + "".join(f"{k:>20}" for k in CONFIGS) + "   미정렬")
    keys = sorted(stats["combined"])
    tot = {k: [0, 0, 0, 0] for k in CONFIGS}
    for key in keys:
        n = stats["combined"][key][0]
        line = f"  {key:<36}"
        for k in CONFIGS:
            st = stats[k][key]
            line += f"{rate(st[1], st[0]):>20}"
            for j in range(4):
                tot[k][j] += st[j]
        out(line + f"   {unaligned[key]}/{n}")
    out(f"  {'(전체)':<36}" + "".join(f"{rate(tot[k][1], tot[k][0]):>20}" for k in CONFIGS))
    out(f"\n[심각도 정확]  " + "  ".join(f"{k} {rate(tot[k][2], tot[k][0])}" for k in CONFIGS))
    out(f"[코드 정확]    " + "  ".join(f"{k} {rate(tot[k][3], tot[k][0])}" for k in CONFIGS))
    out("\n[대조군 — 정상 초안]")
    for k in CONFIGS:
        n, soft, hard = clean[k]
        out(f"  {k:<9} claim {n}개  soft(OK 아님) {rate(soft, n)}   hard(PROHIBITED) {rate(hard, n)}")
    out(f"  초안 최종 PROHIBITED  {rate(clean_draft[1], clean_draft[0])} (n={clean_draft[0]})")
    out(f"\nLLM–Rule 충돌 {rate(conflicts, total_claims)}   VERIFIER_ERROR {rate(verr, total_claims)}"
        f"   (claim {total_claims}개)")
    # ---------- 자연 오류 셋 (사람이 실제로 찾은 오류) ----------
    if NATURAL.exists() and not args.limit:
        with NATURAL.open(encoding="utf-8-sig", newline="") as f:
            nat = list(csv.DictReader(f))
        out("\n[자연 오류 셋] 초안 단위 — claim 하나라도 OK 가 아니면 탐지")
        out(f"  {'id':<8}{'gold':<34}" + "".join(f"{k:>12}" for k in CONFIGS) + "   발급 코드(combined)")
        hit = {k: 0 for k in CONFIGS}
        for r in nat:
            res = cv.verify(r["question"], r["text"], chunks_for(answers, r["question"]))
            worst = {k: max((SEV[variant_verdicts(res, c.claim_id)[k][0]] for c in res.claims), default=0)
                     for k in CONFIGS}
            codes = sorted({h.code.value for v in res.verifications for h in v.reasons} - {"SUPPORTED"})
            for k in CONFIGS:
                hit[k] += worst[k] > 0
            out(f"  {r['nat_id']:<8}{r['expected_codes'][:32]:<34}"
                + "".join(f"{['OK', 'CHECK', 'PROHIB'][worst[k]]:>12}" for k in CONFIGS)
                + f"   {','.join(codes)[:60]}")
        out("  탐지율  " + "  ".join(f"{k} {rate(hit[k], len(nat))}" for k in CONFIGS) + f"  (n={len(nat)})")
        llm.flush()

    out(f"호출 API {llm.calls} / 캐시 {llm.hits}, 토큰 {llm.tokens:,}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"v2_verifier_eval_{date.today().isoformat()}"
    with (REPORT_DIR / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        cols = ["sample_id", "type", "injected", "targets"] + \
               [f"{k}_{x}" for k in CONFIGS for x in ("verdict", "codes")]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (REPORT_DIR / f"{stem}.md").write_text("# v2 ClaimVerifier 평가\n\n```\n" + "\n".join(lines) + "\n```\n",
                                         encoding="utf-8")
    print(f"\n→ reports/{stem}.md / .csv,  {RAW_OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
