"""60 — 새 held-out v1: 생성 → 블라인드 사람 검수 → freeze → 1회 공식 평가.

원칙 (2026-10-14 합의)
-------------------
1. 기존 v1 답변 73건(정상 초안 30 + 개발 43)은 전부 개발 셋이다. 공식 개선 수치에 쓰지 않는다.
2. 새 held-out 은 '검증기 결과를 보기 전에' 사람 라벨을 확정하고 freeze 한다.
3. freeze 이후 eval 은 몇 번이든 돌릴 수 있지만, 공식 수치는 '첫 평가 시점의 코드' 결과 하나다.
   그 결과를 보고 규칙·프롬프트를 고치면 이 셋은 held-out 이 아니다 — eval 로그가 이를 자동 표시한다.

블라인드를 지키는 장치
-------------------
- questions / answers / review / import / freeze 단계는 검증 계층(verification 패키지)을 import 하지
  않는다. 검수 페이지에는 규칙 경고·수치 색칠·우선순위 정렬이 없다 (55번과 다른 점).
  55번의 플래그는 규칙 출력이라, 그걸 보고 라벨을 매기면 사람 라벨이 검증기 쪽으로 기운다.
- 카드 순서는 시드 고정 무작위.
- eval 은 freeze 파일의 해시가 맞을 때만 돈다. 라벨 파일을 한 글자라도 바꾸면 거부한다.

데이터 분리
---------
- 질문은 '개발 셋 답변이 인용한 적 없는 청크'에서만 만든다. 개발 셋이 본 적 없는 문서를 우선한다.
- 기존 평가셋 질문(검색 96, 거절 53)과 글자 2-gram 자카드 0.5 이상 겹치면 버린다.

단계
----
    python scripts/60_heldout_v1.py questions            # 질문 생성 (LLM, 약 80콜, 캐시)
    python scripts/60_heldout_v1.py answers              # v1 파이프라인으로 답변 생성 + 검수 시트 50건
    python scripts/60_heldout_v1.py review               # 블라인드 검수 페이지 열기
    python scripts/60_heldout_v1.py import "<CSV 경로>"   # 검수 결과 병합 (freeze 전까지 반복 가능)
    python scripts/60_heldout_v1.py freeze               # 라벨 확정·해시 기록·읽기 전용
    python scripts/60_heldout_v1.py eval                 # 공식 평가 (claim 단위 vs claim+범위)

N 사유 분류 (복수 선택, 자유 메모 유지)
-----------------------------------
    WRONG              근거와 다르다 (수치·방향·요건·논리)
    UNSUPPORTED        근거로 확인할 수 없는 구체 내용이 있다
    MISSING_CONDITION  근거의 조건·예외·한정이 빠져 더 넓게 읽힌다
    OVERSTATED         근거보다 단정적이다 ('~될 수 있습니다' → '~됩니다' 등)
    OMITTED            질문의 핵심에 답하지 않았다
    MIXED_SCOPE        서로 다른 상품·서비스·조항의 규정을 하나처럼 합쳤다
    SCOPE_MISMATCH     질문과 다른(더 좁거나 넓은) 범위 기준으로 답했다
    OUTDATED           시점에 따라 바뀌는 값을 고정값처럼 말했다 / 구버전 기준
    OTHER              위에 없음 (메모 필수)
X(평가 불가): 질문이 이상하거나 근거만으로 판단이 불가능한 경우. 메모 필수. 평가에서 제외.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import random
import re
import stat
import sys
import webbrowser
from collections import Counter
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

V2 = PROJECT_ROOT / "data" / "eval" / "v2"
INTERIM = PROJECT_ROOT / "data" / "interim"
CHUNKS = INTERIM / "chunks.jsonl"
DEV_ANSWERS = INTERIM / "g_full_answers.json"
RETRIEVAL_EVAL = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
REFUSAL_EVAL = PROJECT_ROOT / "data" / "eval" / "refusal_eval.csv"

Q_CSV = V2 / "heldout_v1_questions.csv"
Q_CACHE = INTERIM / "heldout_v1_question_cache.json"
A_CACHE = INTERIM / "heldout_v1_answers.json"
SHEET = V2 / "heldout_v1_review.csv"
FREEZE = V2 / "heldout_v1_freeze.json"
EVAL_LOG = V2 / "heldout_v1_eval_log.jsonl"
PAGE_OUT = INTERIM / "heldout_v1_review_packet.html"
REPORT_DIR = PROJECT_ROOT / "reports"

SEED = 20261014
MODEL = "gpt-4.1-mini"
Q_TARGET = {"설명서": 32, "약관": 26, "FAQ": 22}       # 생성 수 (거절·중복 손실 감안)
FINAL_TARGET = {"설명서": 20, "약관": 15, "FAQ": 15}   # 검수 대상 50건
ALLOWED_DIFF = {"FAQ": ["easy", "medium"], "설명서": ["easy", "medium", "hard"], "약관": ["medium", "hard"]}
REASONS = ["WRONG", "UNSUPPORTED", "MISSING_CONDITION", "OVERSTATED", "OMITTED",
           "MIXED_SCOPE", "SCOPE_MISMATCH", "OUTDATED", "OTHER"]
REASON_KO = {
    "WRONG": "근거와 다름", "UNSUPPORTED": "근거로 확인 불가", "MISSING_CONDITION": "조건·예외 누락",
    "OVERSTATED": "근거보다 단정적", "OMITTED": "질문 핵심 미답변", "MIXED_SCOPE": "다른 상품·조항 혼합",
    "SCOPE_MISMATCH": "질문과 다른 범위", "OUTDATED": "시점 의존·구버전", "OTHER": "기타(메모)",
}
SHEET_COLS = ["draft_id", "question", "doc_type", "difficulty", "source_chunk_id", "answer",
              "not_found", "evidence_chunk_ids", "human_label", "human_reasons", "note"]


# ==================================================================
# 공통
# ==================================================================


def read_csv(p: Path) -> list[dict]:
    for enc in ("utf-8-sig", "cp949"):
        try:
            with p.open(encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    sys.exit(f"{p.name} 를 읽지 못했습니다.")


def write_csv(p: Path, rows: list[dict], cols: list[str]) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def bigrams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / (len(a | b) or 1)


def frozen_guard() -> None:
    if FREEZE.exists():
        sys.exit(f"{FREEZE.name} 이 있습니다. freeze 이후에는 시트를 바꿀 수 없습니다.")


def load_v1_generator():
    """14번 스크립트의 질문 생성 프롬프트·청크 필터를 그대로 쓴다 (v1 평가셋과 같은 방식)."""
    spec = importlib.util.spec_from_file_location("gen14", PROJECT_ROOT / "scripts" / "14_generate_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def openai_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(PROJECT_ROOT / ".env")
    return OpenAI()


# ==================================================================
# 1. questions
# ==================================================================


def cmd_questions(args) -> int:
    frozen_guard()
    g14 = load_v1_generator()
    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8") if l.strip()]
    dev = json.loads(DEV_ANSWERS.read_text(encoding="utf-8"))
    used_chunks = {e["chunk_id"] for a in dev.values() for e in a.get("evidences", [])}
    for r in read_csv(RETRIEVAL_EVAL):
        used_chunks |= set(r.get("relevant_chunk_ids", "").split("|"))
    for r in read_csv(REFUSAL_EVAL):
        used_chunks.add(r.get("source_chunk_id", ""))
    seen_docs = {c.rsplit("_c", 1)[0] for c in used_chunks if c}
    existing_q = [r["question"] for r in read_csv(RETRIEVAL_EVAL)] + [r["question"] for r in read_csv(REFUSAL_EVAL)]
    existing_bg = [bigrams(q) for q in existing_q]

    repeated = g14.find_repeated_chunks(chunks)
    pool = [c for c in chunks if g14.is_suitable(c) and c["chunk_id"] not in repeated
            and c["chunk_id"] not in used_chunks]
    rng = random.Random(SEED)
    selected = []
    for dt, n in Q_TARGET.items():
        by_doc: dict[str, list[dict]] = {}
        for c in pool:
            if c["doc_type"] == dt:
                by_doc.setdefault(c["doc_id"], []).append(c)
        docs = sorted(by_doc)
        rng.shuffle(docs)
        docs.sort(key=lambda d: d in seen_docs)          # 개발 셋이 본 적 없는 문서 먼저 (안정 정렬)
        for d in docs[:n]:
            selected.append(rng.choice(by_doc[d]))
    print(f"후보 청크 {len(pool)}개 → 선정 {len(selected)}개 "
          f"(개발 셋 미사용 문서 {sum(c['doc_id'] not in seen_docs for c in selected)}개)")

    cache = json.loads(Q_CACHE.read_text(encoding="utf-8")) if Q_CACHE.exists() else {}
    client = None
    rows, dropped = [], Counter()
    new_bg: list[set] = []
    for i, c in enumerate(selected):
        diffs = ALLOWED_DIFF[c["doc_type"]]
        diff = diffs[i % len(diffs)]
        key = hashlib.sha256(f"{MODEL}|{c['chunk_id']}|{diff}|{g14.SYSTEM_PROMPT}".encode()).hexdigest()[:24]
        if key not in cache:
            if client is None:
                client = openai_client()
            user = g14.USER_TEMPLATE.format(doc_type=c["doc_type"], doc_name=c.get("doc_display_name", ""),
                                            text=c["text"], difficulty_guide=g14.DIFFICULTY_PROMPTS[diff])
            try:
                resp = client.chat.completions.create(
                    model=MODEL, temperature=0.8, max_tokens=200, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": g14.SYSTEM_PROMPT}, {"role": "user", "content": user}])
                cache[key] = json.loads(resp.choices[0].message.content)
            except Exception as e:
                dropped[f"생성 실패: {type(e).__name__}"] += 1
                continue
            if len(cache) % 10 == 0:
                Q_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        q = str(cache[key].get("question", "")).strip()
        if len(q) < 8:
            dropped["질문 비어 있음"] += 1
            continue
        bg = bigrams(q)
        if any(jaccard(bg, e) >= 0.5 for e in existing_bg):
            dropped["기존 평가셋 질문과 중복"] += 1
            continue
        if any(jaccard(bg, e) >= 0.5 for e in new_bg):
            dropped["새 질문끼리 중복"] += 1
            continue
        new_bg.append(bg)
        rows.append({"qid": f"h{len(rows) + 1:03d}", "question": q,
                     "answer_hint": cache[key].get("answer_hint", ""), "source_chunk_id": c["chunk_id"],
                     "doc_id": c["doc_id"], "doc_type": c["doc_type"], "difficulty": diff,
                     "doc_seen_in_dev": c["doc_id"] in seen_docs})
    Q_CACHE.parent.mkdir(parents=True, exist_ok=True)
    Q_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    write_csv(Q_CSV, rows, list(rows[0].keys()))
    print(f"질문 {len(rows)}건 → {Q_CSV.relative_to(PROJECT_ROOT)}   유형 {dict(Counter(r['doc_type'] for r in rows))}")
    for k, v in dropped.items():
        print(f"  버림: {k} {v}")
    return 0


# ==================================================================
# 2. answers
# ==================================================================

ENV_FILE = INTERIM / "heldout_v1_answer_env.json"
EMBED_MIN_COS = 0.999      # 인덱스 벡터와 이 이상 같아야 대체 인코더를 쓴다


def prepare_embedder(pipe) -> dict:
    """sentence-transformers 를 쓸 수 없으면 transformers 대체 인코더로 바꾸고 동일성을 검사한다.

    2026-10-14 이 환경에서 Windows 애플리케이션 제어 정책이 scipy DLL 을 차단해
    sentence-transformers import 가 실패했다 (임베딩 계산에는 scipy 가 필요 없다).
    대체 인코더는 기존 FAISS 인덱스의 청크 벡터를 다시 계산해 코사인 ≥ 0.999 일 때만 쓴다.
    통과 못 하면 답변 생성을 멈춘다 — 검색이 v1 과 달라지면 v1 답변 분포가 아니게 되기 때문이다.
    """
    from finguide_rag.embedding.hf_encoder import TransformersEncoder, check_against_index
    emb = pipe.retriever.embedder
    try:
        _ = emb.model                                  # sentence-transformers 로딩 시도
        backend = "sentence-transformers"
    except Exception as e:                             # ImportError, OSError(DLL 차단) 등
        print(f"  sentence-transformers 사용 불가 ({type(e).__name__}: {str(e)[:80]})")
        print("  → transformers 대체 인코더로 전환하고 인덱스 벡터와 동일성을 검사합니다")
        emb._model = TransformersEncoder(emb.spec.model_id, max_length=emb.spec.max_tokens)
        backend = "transformers-fallback"
    chk = check_against_index(emb, pipe.retriever.faiss)
    print(f"  임베딩 백엔드 {backend} · 인덱스 벡터 재현 코사인 min {chk['min_cos']:.5f} / mean {chk['mean_cos']:.5f}"
          f" (n={chk['n']})")
    if chk["min_cos"] < EMBED_MIN_COS:
        sys.exit(f"인덱스 벡터와 일치하지 않습니다 (min {chk['min_cos']:.4f} < {EMBED_MIN_COS}). 답변 생성을 멈춥니다.")
    return {"backend": backend, **chk}


def cmd_answers(args) -> int:
    frozen_guard()
    if SHEET.exists() and any(r.get("human_label") for r in read_csv(SHEET)):
        sys.exit("검수 시트에 이미 라벨이 있습니다. 다시 만들지 않습니다.")
    qs = read_csv(Q_CSV)
    cache = json.loads(A_CACHE.read_text(encoding="utf-8")) if A_CACHE.exists() else {}
    todo = [q for q in qs if q["question"] not in cache]
    if todo:
        from finguide_rag.generation.pipeline import Pipeline    # v1 파이프라인 그대로 (검증 계층 아님)
        pipe = Pipeline.build(PROJECT_ROOT, client=openai_client(), llm_model=MODEL)
        env = prepare_embedder(pipe)
        ENV_FILE.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
        for i, q in enumerate(todo, 1):
            cache[q["question"]] = pipe.answer(q["question"]).to_dict()
            if i % 10 == 0:
                A_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  {i}/{len(todo)}")
        A_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    answered = [q for q in qs if cache[q["question"]].get("decision") == "answer"]
    rng = random.Random(SEED)
    picked = []
    rest = []
    for dt, n in FINAL_TARGET.items():
        grp = [q for q in answered if q["doc_type"] == dt]
        rng.shuffle(grp)
        picked += grp[:n]
        rest += grp[n:]
    rng.shuffle(rest)
    picked += rest[: max(0, sum(FINAL_TARGET.values()) - len(picked))]
    rows = []
    for q in sorted(picked, key=lambda x: x["qid"]):
        a = cache[q["question"]]
        rows.append({"draft_id": q["qid"], "question": q["question"], "doc_type": q["doc_type"],
                     "difficulty": q["difficulty"], "source_chunk_id": q["source_chunk_id"],
                     "answer": a.get("answer", ""), "not_found": json.dumps(a.get("not_found", []), ensure_ascii=False),
                     "evidence_chunk_ids": "|".join(e["chunk_id"] for e in a.get("evidences", [])),
                     "human_label": "", "human_reasons": "", "note": ""})
    write_csv(SHEET, rows, SHEET_COLS)
    print(f"질문 {len(qs)}건 → 답변 {len(answered)}건 (v1 거절 {len(qs) - len(answered)}건) → 검수 {len(rows)}건")
    print(f"  유형 {dict(Counter(r['doc_type'] for r in rows))}")
    print(f"→ {SHEET.relative_to(PROJECT_ROOT)}   (검증기는 아직 돌리지 않았습니다)")
    return 0


# ==================================================================
# 3. review (블라인드)
# ==================================================================

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>held-out v1 검수</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e2e2dc;--acc:#2b5d8a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--mute:#a3a29b;--line:#34342f;--acc:#8ab4dc}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
main{max-width:980px;margin:0 auto;padding:16px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 16px}
.card.done{opacity:.7}h3{margin:0 0 4px;font-size:16px}.meta{color:var(--mute);font-size:13px}
.lab{font-size:12px;color:var(--mute);margin:12px 0 2px;letter-spacing:.04em}.box{border-left:3px solid var(--line);padding:2px 10px}
details{margin-top:8px}summary{cursor:pointer;color:var(--mute)}.ev{font-size:14px;margin:6px 0 10px}
.q{margin-top:12px}.q label{margin-right:10px}.rs{display:none;margin-top:8px;font-size:14px}.rs.on{display:block}
.rs label{display:inline-block;margin:2px 12px 2px 0}textarea{width:100%;font:inherit;margin-top:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.guide{font-size:14px;color:var(--mute)}.guide b{color:var(--ink)}
</style></head><body>
<header><b>held-out v1 검수 (블라인드)</b><span class="meta" id="prog"></span><button onclick="exportCsv()">CSV 내보내기</button></header>
<main><div class="guide"><p><b>정답(Y)</b>: 답변의 모든 내용이 근거 원문으로 뒷받침되고, 직원이 그대로 안내해도 되는 수준.
<b>오류(N)</b>: 그대로 안내하면 안 되는 이유가 하나라도 있음 → 사유를 1개 이상 고르고 메모.
<b>평가 불가(X)</b>: 질문이 이상하거나 근거만으로 판단 불가 → 메모 필수.</p>
<p>이 페이지에는 검증기 결과가 없습니다. 근거 원문만 보고 판단하세요. 카드 순서는 무작위입니다.</p>
<p>__REASON_GUIDE__</p></div>
__CARDS__
</main>
<script>
const KEY="heldout_v1_labels";let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){st={}}
const R=__REASONS__;
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}prog()}
function ok(s){if(!s||!s.l)return false;if(s.l==="N")return (s.r||[]).length>0&&(!(s.r||[]).includes("OTHER")||(s.n||"").trim());if(s.l==="X")return !!(s.n||"").trim();return true}
function lab(id,v){st[id]=st[id]||{};st[id].l=v;document.getElementById("rs_"+id).classList.toggle("on",v==="N");save();mark(id)}
function rsn(id,r,on){st[id]=st[id]||{};let a=new Set(st[id].r||[]);on?a.add(r):a.delete(r);st[id].r=[...a];save();mark(id)}
function memo(id,v){st[id]=st[id]||{};st[id].n=v;save();mark(id)}
function mark(id){document.getElementById("c_"+id).classList.toggle("done",ok(st[id]))}
function prog(){const ids=[...document.querySelectorAll(".card")].map(c=>c.dataset.id);document.getElementById("prog").textContent=ids.filter(i=>ok(st[i])).length+" / "+ids.length+" 완료"}
function exportCsv(){const ids=[...document.querySelectorAll(".card")].map(c=>c.dataset.id);const esc=x=>'"'+String(x||"").replace(/"/g,'""')+'"';
 const rows=["draft_id,human_label,human_reasons,note"].concat(ids.map(i=>{const s=st[i]||{};return [i,s.l||"",(s.l==="N"?(s.r||[]):[]).join("|"),esc(s.n)].join(",")}));
 const b=new Blob(["\\ufeff"+rows.join("\\n")],{type:"text/csv"});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="heldout_v1_labels.csv";a.click()}
document.querySelectorAll(".card").forEach(c=>{const id=c.dataset.id,s=st[id]||{};
 if(s.l){const x=c.querySelector(`input[name="l_${id}"][value="${s.l}"]`);if(x)x.checked=true;document.getElementById("rs_"+id).classList.toggle("on",s.l==="N")}
 (s.r||[]).forEach(r=>{const x=c.querySelector(`input[data-r="${r}"]`);if(x)x.checked=true});
 const t=c.querySelector("textarea");if(t&&s.n)t.value=s.n;mark(id)});prog();
</script></body></html>"""


def cmd_review(args) -> int:
    rows = read_csv(SHEET)
    answers = json.loads(A_CACHE.read_text(encoding="utf-8"))
    rng = random.Random(SEED + 1)
    rng.shuffle(rows)
    cards = []
    for r in rows:
        a = answers.get(r["question"]) or {}
        i = html.escape(r["draft_id"])
        nf = json.loads(r["not_found"] or "[]")
        ev = "".join(f'<div class="ev"><b>{html.escape(e.get("citation", ""))}</b> '
                     f'<span class="meta">{html.escape(e.get("chunk_id", ""))}</span><br>'
                     f'{html.escape(e.get("text", "")).replace(chr(10), "<br>")}</div>' for e in a.get("evidences", []))
        radios = "".join(f'<label><input type="radio" name="l_{i}" value="{v}" onchange="lab(\'{i}\',\'{v}\')"> {t}</label>'
                         for v, t in (("Y", "정답 Y"), ("N", "오류 N"), ("X", "평가 불가 X")))
        reasons = "".join(f'<label><input type="checkbox" data-r="{k}" onchange="rsn(\'{i}\',\'{k}\',this.checked)"> '
                          f'{k} <span class="meta">{REASON_KO[k]}</span></label>' for k in REASONS)
        cards.append(f"""<section class="card" id="c_{i}" data-id="{i}">
<h3>{i} · {html.escape(r['question'])}</h3><div class="meta">{html.escape(r['doc_type'])}</div>
<div class="lab">답변</div><div class="box">{html.escape(r['answer'])}</div>
{('<div class="lab">답변이 모른다고 한 항목</div><div class="box">' + html.escape(' / '.join(nf)) + '</div>') if nf else ''}
<details open><summary>근거 원문 {len(a.get('evidences', []))}개</summary>{ev}</details>
<div class="q">{radios}<div class="rs" id="rs_{i}">{reasons}</div></div>
<textarea rows="2" placeholder="메모 (N·X 는 이유)" oninput="memo('{i}',this.value)"></textarea></section>""")
    guide = " · ".join(f"<b>{k}</b> {REASON_KO[k]}" for k in REASONS)
    page = (PAGE.replace("__CARDS__", "".join(cards)).replace("__REASONS__", json.dumps(REASONS))
            .replace("__REASON_GUIDE__", guide))
    PAGE_OUT.write_text(page, encoding="utf-8")
    print(f"검수 페이지: {PAGE_OUT.relative_to(PROJECT_ROOT)}  ({len(rows)}건, 무작위 순서, 검증기 정보 없음)")
    webbrowser.open(PAGE_OUT.as_uri())
    return 0


# ==================================================================
# 4. import / 5. freeze
# ==================================================================


def validate(rows: list[dict]) -> list[str]:
    errs = []
    for r in rows:
        l = r.get("human_label", "").strip().upper()
        rs = [x for x in r.get("human_reasons", "").split("|") if x]
        if l not in ("Y", "N", "X"):
            errs.append(f"{r['draft_id']}: 라벨 없음")
        elif l == "N" and not rs:
            errs.append(f"{r['draft_id']}: N 인데 사유 없음")
        elif l == "N" and any(x not in REASONS for x in rs):
            errs.append(f"{r['draft_id']}: 알 수 없는 사유 {rs}")
        elif (l == "X" or "OTHER" in rs) and not r.get("note", "").strip():
            errs.append(f"{r['draft_id']}: {l}/OTHER 인데 메모 없음")
    return errs


def cmd_import(args) -> int:
    frozen_guard()
    labels = {r["draft_id"]: r for r in read_csv(Path(args.path))}
    rows = read_csv(SHEET)
    for r in rows:
        lab = labels.get(r["draft_id"])
        if lab:
            r["human_label"] = lab.get("human_label", "").strip().upper()
            r["human_reasons"] = lab.get("human_reasons", "").strip()
            r["note"] = lab.get("note", "").strip()
    write_csv(SHEET, rows, SHEET_COLS)
    errs = validate(rows)
    c = Counter(r["human_label"] or "-" for r in rows)
    print(f"병합: Y {c['Y']} · N {c['N']} · X {c['X']} · 미완료 {c['-']}")
    rc = Counter(x for r in rows if r["human_label"] == "N" for x in r["human_reasons"].split("|") if x)
    if rc:
        print("  N 사유: " + ", ".join(f"{k} {v}" for k, v in rc.most_common()))
    for e in errs:
        print(f"  ✘ {e}")
    print("→ 문제가 없으면 freeze" if not errs else "→ 페이지에서 고친 뒤 다시 import")
    return 1 if errs else 0


def code_fingerprint() -> dict:
    files = sorted((PROJECT_ROOT / "src" / "finguide_rag" / "verification").glob("*.py"))
    files += [PROJECT_ROOT / "src" / "finguide_rag" / "generation" / f
              for f in ("product_match.py", "product_catalog.py", "refusal.py")]
    files += [INTERIM / "product_catalog.json"]
    h = hashlib.sha256()
    per = {}
    for f in files:
        if f.exists():
            b = f.read_bytes().replace(b"\r\n", b"\n")
            h.update(f.name.encode() + b)
            per[f.name] = hashlib.sha256(b).hexdigest()[:10]
    return {"all": h.hexdigest()[:16], "files": per}


def cmd_freeze(args) -> int:
    frozen_guard()
    rows = read_csv(SHEET)
    errs = validate(rows)
    if errs:
        for e in errs:
            print(f"  ✘ {e}")
        sys.exit("라벨이 완전하지 않아 freeze 하지 않습니다.")
    c = Counter(r["human_label"] for r in rows)
    meta = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "sheet": str(SHEET.relative_to(PROJECT_ROOT)), "sheet_sha256": sha256(SHEET),
        "answers_sha256": sha256(A_CACHE), "questions_sha256": sha256(Q_CSV),
        "counts": dict(c),
        "reasons": dict(Counter(x for r in rows if r["human_label"] == "N" for x in r["human_reasons"].split("|"))),
        "code_at_freeze": code_fingerprint(),
        "answer_env": json.loads(ENV_FILE.read_text(encoding="utf-8")) if ENV_FILE.exists() else None,
        "note": "verifier 결과를 보기 전에 확정한 라벨. 이 파일 이후 첫 eval 이 공식 결과다.",
    }
    FREEZE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in (SHEET, FREEZE):
        os.chmod(p, stat.S_IREAD)            # 읽기 전용 (실수로 고치는 것 방지)
    print(f"freeze 완료: Y {c['Y']} · N {c['N']} · X {c['X']}")
    print(f"  sheet sha256 {meta['sheet_sha256'][:16]}…   code {meta['code_at_freeze']['all']}")
    print("→ 지금 커밋하세요. 그 다음 eval.")
    return 0


# ==================================================================
# 6. eval
# ==================================================================

CODE_FOR_REASON = {
    "WRONG": {"CONTRADICTED", "NUMERIC_MISMATCH", "PRODUCT_MISMATCH", "INSTITUTION_MISMATCH"},
    "UNSUPPORTED": {"NO_EVIDENCE"},
    "MISSING_CONDITION": {"MISSING_CONDITION"},
    "OVERSTATED": {"OVERSTATED", "MISSING_CONDITION"},
    "OMITTED": {"OMITTED_REQUIRED_INFO", "MISSING_CONDITION", "NO_EVIDENCE"},
    "MIXED_SCOPE": {"MIXED_SCOPE", "PRODUCT_MISMATCH", "CONTRADICTED"},
    "SCOPE_MISMATCH": {"SCOPE_MISMATCH", "PRODUCT_MISMATCH", "MISSING_CONDITION"},
    "OUTDATED": {"TIME_VARIANT", "OUTDATED_SOURCE"},
}


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if not n:
        return "-"
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n}={p:.2f} [{max(0, c - h):.2f}–{min(1, c + h):.2f}]"


def cmd_eval(args) -> int:
    if not FREEZE.exists():
        sys.exit("freeze 전에는 평가하지 않습니다. 라벨을 먼저 확정하세요.")
    meta = json.loads(FREEZE.read_text(encoding="utf-8"))
    if sha256(SHEET) != meta["sheet_sha256"] or sha256(A_CACHE) != meta["answers_sha256"]:
        sys.exit("freeze 이후 시트 또는 답변 파일이 바뀌었습니다. 평가를 거부합니다.")

    fp = code_fingerprint()
    log = [json.loads(l) for l in EVAL_LOG.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if EVAL_LOG.exists() else []
    first = log[0] if log else None
    official = first is None or first["code"]["all"] == fp["all"]

    from finguide_rag.generation.product_catalog import ProductCatalog
    from finguide_rag.generation.product_match import ProductMatcher
    from finguide_rag.verification.claim_verifier import (PROMPT_VERSION, CachedLLM, ClaimVerifier,
                                                           EvidenceChunk, prompt_hash)
    from finguide_rag.verification.scope import DocIndex

    catalog = INTERIM / "product_catalog.json"
    matcher = ProductMatcher.from_catalog(catalog)
    known = [e.canonical_name for e in ProductCatalog.load(catalog).all_named if e.canonical_name]
    index = DocIndex.build(INTERIM / "parsed" / "documents.jsonl", catalog, matcher)
    llm = CachedLLM(openai_client(), MODEL, INTERIM / "v2_verifier_cache.json")
    configs = {
        "claim": ClaimVerifier(llm, matcher, known_products=known),
        "claim+scope": ClaimVerifier(llm, matcher, known_products=known, doc_index=index),
    }
    answers = json.loads(A_CACHE.read_text(encoding="utf-8"))
    rows = [r for r in read_csv(SHEET) if r["human_label"] in ("Y", "N")]
    res_rows = []
    for i, r in enumerate(rows, 1):
        a = answers[r["question"]]
        chunks = [EvidenceChunk(e["chunk_id"], e.get("text", ""), e.get("citation", ""), e.get("doc_type", ""))
                  for e in a.get("evidences", [])]
        row = {"draft_id": r["draft_id"], "label": r["human_label"], "reasons": r["human_reasons"],
               "doc_type": r["doc_type"], "note": r["note"][:100]}
        for name, cv in configs.items():
            out = cv.verify(r["question"], r["answer"], chunks)
            codes = {h.code.value for v in out.verifications for h in v.reasons}
            codes |= {h.code.value for h in out.final.draft_level_reasons}
            row[f"{name}_final"] = out.final.final.value
            row[f"{name}_codes"] = "|".join(sorted(codes - {"SUPPORTED"}))
        res_rows.append(row)
        if i % 10 == 0:
            llm.flush()
            print(f"  {i}/{len(rows)}  (API {llm.calls}콜, 캐시 {llm.hits})")
    llm.flush()

    entry = {"at": datetime.now().isoformat(timespec="seconds"), "code": fp,
             "prompt": f"{PROMPT_VERSION}/{prompt_hash(PROMPT_VERSION)}", "official": official}
    with EVAL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    lines: list[str] = []

    def out(x=""):
        print(x)
        lines.append(x)

    n_rows = [r for r in res_rows if r["label"] == "N"]
    y_rows = [r for r in res_rows if r["label"] == "Y"]
    out("=" * 78)
    if official:
        out("held-out v1 공식 평가" + (" (재실행 — 첫 평가와 같은 코드)" if first else " (첫 평가)"))
    else:
        out("⚠ held-out v1 비공식 평가 — 첫 평가 이후 코드가 바뀌었다. 이 결과는 held-out 수치가 아니다.")
        out(f"   첫 평가 {first['at']} code {first['code']['all']} → 지금 code {fp['all']}")
    out(f"freeze {meta['frozen_at']} · 라벨 Y {len(y_rows)} · N {len(n_rows)} (X 제외) · prompt {entry['prompt']}")
    out("=" * 78)
    out(f"{'':<22}{'claim 단위':>28}{'claim+범위':>28}")
    for label, rs, cond in (("오류 탐지 (N)", n_rows, lambda f: f != "OK"),
                            ("정상 오탐 soft (Y)", y_rows, lambda f: f != "OK"),
                            ("정상 오탐 hard (Y)", y_rows, lambda f: f == "PROHIBITED")):
        out(f"{label:<22}" + "".join(f"{wilson(sum(cond(r[c + '_final']) for r in rs), len(rs)):>28}"
                                     for c in configs))

    out("\n[사유별 — 탐지(아무 경고) / 사유에 맞는 코드]   (한 초안이 여러 사유에 들어갈 수 있음)")
    for reason in REASONS:
        rs = [r for r in n_rows if reason in r["reasons"].split("|")]
        if not rs:
            continue
        cells = []
        for c in configs:
            det = sum(r[c + "_final"] != "OK" for r in rs)
            hit = sum(bool(set(r[c + "_codes"].split("|")) & CODE_FOR_REASON.get(reason, set())) for r in rs)
            cells.append(f"{det}/{len(rs)} · {hit}/{len(rs)}")
        out(f"  {reason:<20}" + "".join(f"{x:>28}" for x in cells))

    out("\n[문서 유형별 오류 탐지 (claim+범위)]")
    for dt in ("설명서", "약관", "FAQ"):
        rs = [r for r in n_rows if r["doc_type"] == dt]
        out(f"  {dt:<6} {wilson(sum(r['claim+scope_final'] != 'OK' for r in rs), len(rs))}")

    changed = [r for r in res_rows if (r["claim_final"] == "OK") != (r["claim+scope_final"] == "OK")]
    out(f"\n[범위 검사로 판정이 바뀐 초안] {len(changed)}건")
    for r in changed:
        right = (r["label"] == "N")
        out(f"  {r['draft_id']} 사람={r['label']} ({r['reasons'] or '-'}) → {'맞게 바뀜' if right else '오탐 추가'}"
            f"  {r['claim+scope_codes']}")
    out("\n[놓친 오류 (claim+범위 OK)]")
    for r in n_rows:
        if r["claim+scope_final"] == "OK":
            out(f"  {r['draft_id']} [{r['reasons']}] {r['note']}")
    out(f"\n호출 API {llm.calls} / 캐시 {llm.hits}, 토큰 {llm.tokens:,}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"v2_heldout_v1_eval_{date.today().isoformat()}" + ("" if official else "_UNOFFICIAL")
    write_csv(REPORT_DIR / f"{stem}.csv", res_rows, list(res_rows[0].keys()))
    (REPORT_DIR / f"{stem}.md").write_text("# held-out v1 평가\n\n```\n" + "\n".join(lines) + "\n```\n",
                                         encoding="utf-8")
    print(f"→ reports/{stem}.md / .csv   (로그 {EVAL_LOG.relative_to(PROJECT_ROOT)})")
    return 0


# ==================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("questions", "answers", "review", "freeze", "eval"):
        sub.add_parser(name)
    p = sub.add_parser("import")
    p.add_argument("path")
    args = ap.parse_args()
    return {"questions": cmd_questions, "answers": cmd_answers, "review": cmd_review,
            "import": cmd_import, "freeze": cmd_freeze, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
