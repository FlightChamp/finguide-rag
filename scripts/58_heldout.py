"""58 — held-out 검수 셋: 한 번도 튜닝에 쓰지 않은 답변으로 검증기를 공정하게 잰다.

왜 필요한가
---------
정상 초안 25건·자연 오류 8건은 규칙과 정렬 프롬프트(cv-v2)를 고칠 때 결과를 보고 쓴 셋이다.
그 셋에서 잰 개선 폭은 낙관적이다. 최종 수치는 '고칠 때 본 적 없는' 답변으로 재야 한다.

대상
----
v1 생성 답변 중 자동 평가를 통과한 적격 후보 73건에서 정상 초안 풀 30건을 뺀 43건.
(51번 사전 점검의 dev 셋과 같은 집합. 규칙 튜닝은 dev 를 보며 했으므로 '규칙'에는 완전한
held-out 이 아니다 — 하지만 LLM 정렬 프롬프트와 사람 라벨은 이 셋을 본 적이 없다.)
구어체 변환 없이 v1 답변 원문을 그대로 초안으로 쓴다. 변환 비용·오류를 줄이기 위해서다.
따라서 문체가 정상 초안 셋(구어체)과 다르다는 점을 결과 해석 때 감안한다.

단계
----
    1) --build    검수 시트 생성: data/eval/v2/heldout_review_v0.csv
    2) 55번으로 근거 대조 검수 (문체 열 없음):
         python scripts/55_review_packet.py --sheet data/eval/v2/heldout_review_v0.csv --no-style --scan dev
         → 페이지에서 정답 Y/N, [CSV 내보내기]
         python scripts/55_review_packet.py --sheet data/eval/v2/heldout_review_v0.csv --no-style --import "<경로>"
    3) --eval     cv-v1 과 cv-v2 를 나란히 평가 (추출 공유, 정렬만 버전별 호출 — 처음 약 130콜)

평가 지표 (초안 단위)
------------------
    오류 초안(N)  탐지 = 최종 판정이 OK 가 아님
    정상 초안(Y)  오탐(soft) = 최종 판정이 OK 가 아님 / 오탐(hard) = PROHIBITED
    두 버전의 차이를 '같은 초안에서 결과가 갈린 건'으로 따로 보여준다 (표본이 작으므로
    비율보다 갈린 사례가 더 많은 것을 말해준다).

산출물
-----
    data/eval/v2/heldout_review_v0.csv
    reports/v2_heldout_eval_<날짜>.md / .csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CANDIDATES = PROJECT_ROOT / "data" / "eval" / "v2" / "clean_pool_candidates.csv"
POOL = PROJECT_ROOT / "data" / "eval" / "v2" / "clean_drafts_v1.csv"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
CATALOG = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"
CACHE = PROJECT_ROOT / "data" / "interim" / "v2_verifier_cache.json"
SHEET = PROJECT_ROOT / "data" / "eval" / "v2" / "heldout_review_v0.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
VERSIONS = ("cv-v1", "cv-v2")
SHEET_COLS = ["draft_id", "query_id", "question", "doc_type", "difficulty", "not_found",
              "v1_answer", "draft_text", "auto_check", "auto_diff",
              "human_is_correct", "human_style_ok", "note", "evidence_chunk_ids"]


def read_csv(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "cp949"):
        try:
            with path.open(encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    sys.exit(f"{path.name} 를 읽지 못했습니다.")


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if not n:
        return "  -"
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {p:.2f} [{max(0, c - h):.2f}–{min(1, c + h):.2f}]"


# ------------------------------------------------------------------


def build() -> int:
    if SHEET.exists():
        done = sum(1 for r in read_csv(SHEET) if r.get("human_is_correct"))
        if done:
            sys.exit(f"{SHEET.name} 에 이미 검수 {done}건이 있습니다. 덮어쓰지 않습니다.")
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    picked = {r["question"] for r in read_csv(POOL)}
    rows = []
    for c in read_csv(CANDIDATES):
        if c["eligible"] != "True" or c["question"] in picked:
            continue
        a = answers.get(c["question"]) or {}
        rows.append({
            "draft_id": c["query_id"], "query_id": c["query_id"], "question": c["question"],
            "doc_type": c["doc_type"], "difficulty": c["difficulty"], "not_found": c["not_found"],
            "v1_answer": a.get("answer", ""), "draft_text": a.get("answer", ""),
            "auto_check": "", "auto_diff": "", "human_is_correct": "", "human_style_ok": "",
            "note": "", "evidence_chunk_ids": "|".join(e["chunk_id"] for e in a.get("evidences", [])),
        })
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLS)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"held-out 검수 시트 {len(rows)}건 → {SHEET.relative_to(PROJECT_ROOT)}")
    print(f"  문서유형 {dict(Counter(r['doc_type'] for r in rows))}")
    print("다음: 55번으로 검수 (--sheet ... --no-style --scan dev)")
    return 0


# ------------------------------------------------------------------


def evaluate() -> int:
    from dotenv import load_dotenv
    from openai import OpenAI

    from finguide_rag.generation.product_catalog import ProductCatalog
    from finguide_rag.generation.product_match import ProductMatcher
    from finguide_rag.verification.claim_verifier import CachedLLM, ClaimVerifier, EvidenceChunk, prompt_hash

    rows = [r for r in read_csv(SHEET) if r.get("human_is_correct", "").strip().upper() in ("Y", "N")]
    if not rows:
        sys.exit("검수된 행이 없습니다. 55번 검수·병합을 먼저 하세요.")
    load_dotenv(PROJECT_ROOT / ".env")
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    matcher = ProductMatcher.from_catalog(CATALOG)
    known = [e.canonical_name for e in ProductCatalog.load(CATALOG).all_named if e.canonical_name]
    llm = CachedLLM(OpenAI(), "gpt-4.1-mini", CACHE)
    cvs = {v: ClaimVerifier(llm, matcher, known_products=known, prompt_version=v) for v in VERSIONS}

    out_rows = []
    for i, r in enumerate(rows, 1):
        chunks = [EvidenceChunk(e["chunk_id"], e.get("text", ""), e.get("citation", ""), e.get("doc_type", ""))
                  for e in (answers.get(r["question"]) or {}).get("evidences", [])]
        row = {"draft_id": r["draft_id"], "label": r["human_is_correct"].upper(),
               "doc_type": r["doc_type"], "note": r.get("note", "")[:120]}
        for v, cv in cvs.items():
            res = cv.verify(r["question"], r["draft_text"], chunks)
            row[f"{v}_final"] = res.final.final.value
            row[f"{v}_codes"] = "|".join(sorted({h.code.value for x in res.verifications
                                                 for h in x.reasons} - {"SUPPORTED"}))
        out_rows.append(row)
        if i % 10 == 0:
            llm.flush()
            print(f"  {i}/{len(rows)}  (API {llm.calls}콜, 캐시 {llm.hits})")
    llm.flush()

    lines: list[str] = []

    def out(x=""):
        print(x)
        lines.append(x)

    err = [r for r in out_rows if r["label"] == "N"]
    ok = [r for r in out_rows if r["label"] == "Y"]
    out("=" * 72)
    out(f"held-out 평가 — 검수 {len(out_rows)}건 (오류 N {len(err)} · 정상 Y {len(ok)})")
    out("=" * 72)
    out(f"{'':<20}" + "".join(f"{v + ' (' + prompt_hash(v) + ')':>26}" for v in VERSIONS))
    out(f"{'오류 탐지':<20}" + "".join(f"{wilson(sum(r[v + '_final'] != 'OK' for r in err), len(err)):>26}"
                                    for v in VERSIONS))
    out(f"{'정상 오탐 soft':<20}" + "".join(f"{wilson(sum(r[v + '_final'] != 'OK' for r in ok), len(ok)):>26}"
                                       for v in VERSIONS))
    out(f"{'정상 오탐 hard':<20}" + "".join(
        f"{wilson(sum(r[v + '_final'] == 'PROHIBITED' for r in ok), len(ok)):>26}" for v in VERSIONS))

    diff = [r for r in out_rows if (r["cv-v1_final"] == "OK") != (r["cv-v2_final"] == "OK")]
    out(f"\n[두 버전의 결과가 갈린 초안] {len(diff)}건")
    for r in diff:
        good = (r["label"] == "N") == (r["cv-v2_final"] != "OK")
        out(f"  {r['draft_id']:<6} 사람={r['label']}  v1={r['cv-v1_final']:<10} v2={r['cv-v2_final']:<10}"
            f" → {'v2 가 맞음' if good else 'v1 이 맞음'}  {r['cv-v2_codes'] or r['cv-v1_codes']}")
    out("\n[놓친 오류 (두 버전 모두 OK)]")
    for r in err:
        if r["cv-v1_final"] == "OK" and r["cv-v2_final"] == "OK":
            out(f"  {r['draft_id']:<6} {r['note']}")
    out(f"\n호출 API {llm.calls} / 캐시 {llm.hits}, 토큰 {llm.tokens:,}")
    out("※ 규칙은 51번 dev(=이 셋의 v1 답변)를 보며 조정됐으므로 규칙 기여분은 완전한 held-out 이 아니다.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"v2_heldout_eval_{date.today().isoformat()}"
    with (REPORT_DIR / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    (REPORT_DIR / f"{stem}.md").write_text("# v2 held-out 평가\n\n```\n" + "\n".join(lines) + "\n```\n",
                                         encoding="utf-8")
    print(f"→ reports/{stem}.md / .csv")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    return build() if args.build else evaluate()


if __name__ == "__main__":
    sys.exit(main())
