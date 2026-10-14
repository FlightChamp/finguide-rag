"""66 — v2-B 조건 교란 (LLM 생성 → 사람 확인): 조건 보존 검사의 진짜 시험지.

왜 필요한가 (64·65번 진단, 2026-10-15)
-----------------------------------
규칙 교란 CONDITION_DROP 은 '조건 블록 문구 삭제'였다. 놓친 4건을 보니 4건 모두 같은 조건이 핵심 설명에
다른 말로 남아 있었고(예: '미지정 시' ↔ '지정하지 않으면'), LLM 조건 검사는 그걸 정확히 '보존'이라고 했다.
즉 놓침이 아니라 교란 라벨 잡음이었고, LLM 조건 검사는 진짜 조건 삭제로 시험받은 적이 없다.

그래서 조건을 '어디에서도' 없애거나 뜻을 바꾼 교란을 LLM 으로 만들고, 사람이 그 교란이 정말 오류인지 확인한다.
    DROP   대상 조건이 쉬운 설명 어디에도 남지 않게 지움
    ALTER  대상 조건의 범위·대상·방향을 바꿈 (수치가 있는 조건은 ALTER 대신 DROP — 수치 오류는 규칙 교란이 따로 본다)
기반 쉬운 설명 1건당 조건 1개, 유형 1개 (시드 고정).

역검증은 돌리지 않는다. 확인할 때 역검증 결과를 보지 않게 하기 위해서다. 평가는 64번이 한다.

사람 확인
--------
    data/eval/v2/b_dev_condpert_review_v0.csv 의 valid 칸에
        Y  이 교란본을 고객이 보면 조건을 잘못 알게 된다 (DROP: 조건이 정말 사라짐 / ALTER: 뜻이 정말 바뀜)
        N  조건이 아직 남아 있거나, 교란이 망가졌다 (문장이 깨짐, 다른 내용까지 바뀜)
    다시 실행해도 같은 교란이면 라벨은 보존된다 (교란 문장이 바뀌면 라벨을 지운다).

실행
----
    python scripts/66_b_cond_perturb.py --dry-run
    python scripts/66_b_cond_perturb.py
산출물
-----
    data/eval/v2/b_dev_condpert_v0.jsonl
    data/eval/v2/b_dev_condpert_review_v0.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.verification.back_verify import explanation_from_json  # noqa: E402
from finguide_rag.verification.claim_verifier import CachedLLM  # noqa: E402

V2 = PROJECT_ROOT / "data" / "eval" / "v2"
INTERIM = PROJECT_ROOT / "data" / "interim"
BASE = V2 / "b_dev_base_v0.jsonl"
OUT = V2 / "b_dev_condpert_v0.jsonl"
REVIEW = V2 / "b_dev_condpert_review_v0.csv"
CACHE = INTERIM / "v2_condpert_cache.json"
MODEL = "gpt-4.1-mini"
FIELDS = ["cp_id", "b_id", "kind", "condition", "before", "after", "text_sha", "valid", "memo"]

CP_VERSION = "cp-v1"
CP_SYSTEM = """너는 검증기 평가용 '오류가 있는 설명'을 만드는 도구다.
고객용 쉬운 설명에서 [대상 조건] 하나만 지시대로 고친다.

DROP  대상 조건이 쉬운 설명 어디에도 남지 않게 한다.
      조건을 담은 문장은 조건 부분만 지우고 나머지 뜻은 그대로 둔다.
      조건만 담은 문장(조건을 빼면 남는 내용이 없는 문장)은 text 를 "" 로 한다.
      같은 조건을 다른 말로 쓴 부분도 모두 지운다.
ALTER 대상 조건의 범위·대상·방향을 바꾼다 (넓히기, 좁히기, 반대로). 그럴듯하게 바꾼다.
      예: '개인사업자에 한해' → '모든 고객에게', '지정하지 않으면' → '지정하면'

공통
- 대상 조건과 관계없는 부분은 한 글자도 바꾸지 마라.
- 수치·상품명·기관명은 바꾸지 마라.
- 문장은 자연스러운 존댓말 완결 문장으로 남긴다.
- 고친 문구만 출력한다.

출력은 JSON 하나:
{"items": [{"item_id": "s1", "text": "..."}]}"""

CP_USER = "지시: {kind}\n[대상 조건]\n{cond}\n\n[쉬운 설명]\n{items}"


def cp_tag() -> str:
    return CP_VERSION + ":" + hashlib.sha256((CP_SYSTEM + CP_USER).encode()).hexdigest()[:12]


def openai_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(PROJECT_ROOT / ".env")
    return OpenAI()


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not BASE.exists():
        print("63번을 먼저 실행하세요")
        return 1
    rows = [json.loads(x) for x in BASE.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if not r.get("generation_error")]
    llm = CachedLLM(None if args.dry_run else openai_client(), MODEL, CACHE)

    out_rows, review, skipped = [], [], []
    for r in rows:
        expl = explanation_from_json(r["explanation"])
        conds = expl.required_conditions
        if not conds:
            continue
        rng = random.Random(int(hashlib.sha256(r["b_id"].encode()).hexdigest()[:8], 16))
        k = rng.choice(conds)
        kind = "ALTER" if (rng.random() < 0.5 and not re.search(r"\d", k.text)) else "DROP"
        items = expl.customer_items()
        items_txt = "\n".join(f"{it.item_id}: {it.text}" for it in items)
        data = llm.json(CP_SYSTEM, CP_USER.format(kind=kind, cond=k.text, items=items_txt),
                        max_tokens=1200, tag=cp_tag())
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            skipped.append((r["b_id"], "응답 없음"))
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
                new = ""          # '""' 처럼 글자 없는 응답은 삭제로 본다
            if new != it.text:
                changed.append({"item_id": it.item_id, "before": it.text, "after": new})
                it.text = new
        if not changed:
            skipped.append((r["b_id"], "바뀐 문구 없음"))
            continue
        e.core = [it for it in e.core if it.text]
        e.conditions = [it for it in e.conditions if it.text]
        if not e.core:
            skipped.append((r["b_id"], "핵심 설명이 모두 지워짐"))
            continue
        cp_id = f"cp_{r['b_id']}_{k.cond_id}_{kind}"
        before = "\n".join(f"{c['item_id']}: {c['before']}" for c in changed)
        after = "\n".join(f"{c['item_id']}: {c['after'] or '(삭제)'}" for c in changed)
        sha = hashlib.sha256(after.encode()).hexdigest()[:10]
        out_rows.append({"cp_id": cp_id, "b_id": r["b_id"], "kind": kind, "cond_id": k.cond_id,
                         "cond_text": k.text, "changed": changed, "explanation": e.to_dict()})
        review.append({"cp_id": cp_id, "b_id": r["b_id"], "kind": kind, "condition": k.text,
                       "before": before, "after": after, "text_sha": sha})

    print("=" * 70)
    print(f"v2-B 조건 교란 (LLM 생성, {CP_VERSION})")
    print("=" * 70)
    print(f"기반 {len(rows)}건 중 조건 있는 기반 → 교란 {len(out_rows)}건 "
          f"(DROP {sum(x['kind'] == 'DROP' for x in out_rows)} · ALTER {sum(x['kind'] == 'ALTER' for x in out_rows)})")
    if skipped:
        print(f"  건너뜀 {len(skipped)}건: {skipped[:5]}")
    print(f"  API {llm.calls}콜 / 캐시 {llm.hits} · 토큰 {llm.tokens:,}")
    if args.dry_run:
        print(f"[dry-run] 캐시에 없는 교란 약 {sum(1 for s in skipped if s[1] == '응답 없음')}콜")
        return 0
    llm.flush()

    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out_rows) + "\n", encoding="utf-8")
    prev = {x["cp_id"]: x for x in read_csv(REVIEW)} if REVIEW.exists() else {}
    kept = 0
    for x in review:
        old = prev.get(x["cp_id"], {})
        same = old.get("text_sha") == x["text_sha"]
        x["valid"] = old.get("valid", "") if same else ""
        x["memo"] = old.get("memo", "") if same else ""
        kept += bool(same and x["valid"])
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(review)
    print(f"→ {OUT.relative_to(PROJECT_ROOT)}")
    print(f"→ {REVIEW.relative_to(PROJECT_ROOT)}   (확인 완료 {kept}/{len(review)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
