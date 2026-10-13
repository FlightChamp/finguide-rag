"""55 — 정상 초안 30건 사람 검수용 리뷰 페이지를 만들고, 검수 결과를 시트에 병합한다.

왜 필요한가
---------
clean_drafts_v1.csv 에는 근거 '원문'이 없다(청크 ID 만 있다). 그 시트만 보고는
human_is_correct(근거 대비 정답인가)를 판단할 수 없다. 문체(human_style_ok)는
v1_answer 와 draft_text 를 나란히 보면 되지만, 정답 여부는 근거를 읽어야 한다.

이 스크립트는 문항마다 [질문 · v1 답변 · 구어체 초안 · 근거 원문]을 한 화면에 보여주고,
규칙으로 '먼저 볼 곳'을 표시한다.
    - 답변 수치: 근거에서 찾으면 초록, 못 찾으면 빨강 (numeric.compare_slot)
    - 우선 확인 플래그: 근거에 없는 수치 / 부분 답변(not_found>0) / v1 상품일치 재분류 건 /
      규칙 사전 점검 경고 / hard 난이도
플래그가 있는 문항이 위로 정렬된다. 플래그 없는 문항은 훑어봐도 되지만,
플래그 있는 문항은 근거와 문장 단위로 대조할 것.

페이지에서 Y/N 을 고르고 [CSV 내보내기]를 누르면 review_labels.csv 가 다운로드된다.
(선택은 브라우저에 임시 저장되어 새로고침해도 유지된다)

실행
----
    python scripts/55_review_packet.py                     # 리뷰 페이지 생성 후 브라우저로 열기
    python scripts/55_review_packet.py --import "<다운로드한 review_labels.csv 경로>"

산출물
-----
    data/interim/v2_review_packet.html   (하나은행 문서 원문이 들어 있으므로 커밋하지 않는다)
    --import 시: data/eval/v2/clean_drafts_v1.csv 의 human_* 열 갱신, note 에 [review:packet] 표시
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.verification.numeric import MatchStatus, compare_slot, extract_numerics  # noqa: E402

SHEET = PROJECT_ROOT / "data" / "eval" / "v2" / "clean_drafts_v1.csv"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
MISMATCH_REVIEW = PROJECT_ROOT / "data" / "eval" / "mismatch_review.md"
SCAN_GLOB = "v2_rules_clean_scan_test_*.csv"
OUT = PROJECT_ROOT / "data" / "interim" / "v2_review_packet.html"
OK_STATUSES = {MatchStatus.MATCH, MatchStatus.WITHIN_RANGE}


def read_csv_any(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "cp949"):
        try:
            with path.open(encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    sys.exit(f"{path.name} 를 읽지 못했습니다.")


def reclassified_questions() -> set[str]:
    if not MISMATCH_REVIEW.exists():
        return set()
    return {m.group(1).strip() for m in
            re.finditer(r"^## \[\d+\]\s*(.+)$", MISMATCH_REVIEW.read_text(encoding="utf-8"), re.M)}


def scan_warnings() -> dict[str, list[str]]:
    files = sorted((PROJECT_ROOT / "reports").glob(SCAN_GLOB))
    if not files:
        return {}
    out: dict[str, list[str]] = {}
    for r in read_csv_any(files[-1]):
        if r.get("proxy_codes"):
            out.setdefault(r["draft_id"], []).append(r["proxy_codes"])
    return out


def mark_numbers(text: str, evidence: str) -> tuple[str, int]:
    """답변 텍스트를 HTML 로 만들며 수치에 색을 입힌다. (html, 근거에 없는 수치 수)"""
    ev = extract_numerics(evidence)
    slots = sorted(extract_numerics(text), key=lambda s: s.span[0])
    parts, last, missing = [], 0, 0
    for s in slots:
        a, b = s.span
        st, _ = compare_slot(s, ev) if s.unit != "date" else (MatchStatus.MATCH, "")
        good = st in OK_STATUSES
        missing += not good
        parts.append(html.escape(text[last:a]))
        parts.append(f'<mark class="{"ok" if good else "bad"}" title="{st.value}">'
                     f'{html.escape(text[a:b])}</mark>')
        last = b
    parts.append(html.escape(text[last:]))
    return "".join(parts), missing


def highlight_in_evidence(evidence: str, answer: str) -> str:
    raws = {s.raw for s in extract_numerics(answer) if s.raw}
    out = html.escape(evidence)
    for r in sorted(raws, key=len, reverse=True):
        out = out.replace(html.escape(r), f'<mark class="ev">{html.escape(r)}</mark>')
    return out.replace("\n", "<br>")


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>정상 초안 검수</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e2e2dc;--ok:#d7f0dc;--bad:#ffd9d6;--ev:#fff1b8;--flag:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--mute:#a3a29b;--line:#34342f;--ok:#1f4a2a;--bad:#5a2421;--ev:#5a4b12;--flag:#ff8a80}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
header b{font-size:16px} .prog{color:var(--mute)} button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
main{max-width:980px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 16px}
.card.done{opacity:.72} h3{margin:0 0 4px;font-size:16px} .meta{color:var(--mute);font-size:13px}
.flags span{display:inline-block;font-size:12px;color:var(--flag);border:1px solid var(--flag);border-radius:999px;padding:0 8px;margin:4px 4px 0 0}
.lab{font-size:12px;color:var(--mute);margin:12px 0 2px;text-transform:uppercase;letter-spacing:.04em}
.box{border-left:3px solid var(--line);padding:2px 10px}
mark.ok{background:var(--ok);color:inherit} mark.bad{background:var(--bad);color:inherit} mark.ev{background:var(--ev);color:inherit}
details{margin-top:8px} summary{cursor:pointer;color:var(--mute)} .ev{font-size:14px;margin:6px 0 10px}
.q{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px} .q label{margin-right:8px} textarea{width:100%;font:inherit;margin-top:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.guide{font-size:14px;color:var(--mute)}
</style></head><body>
<header><b>정상 초안 검수</b><span class="prog" id="prog"></span>
<button onclick="exportCsv()">CSV 내보내기</button></header>
<main>
<p class="guide">초록=근거에서 찾은 수치, 빨강=근거에서 못 찾은 수치. <b>정답 여부</b>: v1 답변의 모든 문장이 근거 원문에 있는가, 수치·조건이 맞는가.
<b>문체</b>: 구어체 초안이 v1 답변과 같은 뜻인가(빠진 조건·바뀐 수치·추가된 내용 없음). 플래그 있는 문항은 근거를 펼쳐 문장 단위로 대조하세요.</p>
__CARDS__
</main>
<script>
const KEY="v2_review_labels";let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){st={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}prog()}
function set(id,k,v){st[id]=st[id]||{};st[id][k]=v;save();mark(id)}
function mark(id){const c=document.getElementById("c_"+id);const s=st[id]||{};c.classList.toggle("done",!!(s.c&&s.s))}
function prog(){const ids=[...document.querySelectorAll(".card")].map(c=>c.dataset.id);
 const d=ids.filter(i=>st[i]&&st[i].c&&st[i].s).length;document.getElementById("prog").textContent=d+" / "+ids.length+" 완료"}
function exportCsv(){const ids=[...document.querySelectorAll(".card")].map(c=>c.dataset.id);
 const esc=x=>'"'+String(x||"").replace(/"/g,'""')+'"';
 const rows=["draft_id,human_is_correct,human_style_ok,note"].concat(ids.map(i=>{const s=st[i]||{};return [i,s.c||"",s.s||"",esc(s.n)].join(",")}));
 const b=new Blob(["\\ufeff"+rows.join("\\n")],{type:"text/csv"});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="review_labels.csv";a.click()}
document.querySelectorAll(".card").forEach(c=>{const id=c.dataset.id,s=st[id]||{};
 ["c","s"].forEach(k=>{if(s[k]){const r=c.querySelector(`input[name="${k}_${id}"][value="${s[k]}"]`);if(r)r.checked=true}});
 const t=c.querySelector("textarea");if(t&&s.n)t.value=s.n;mark(id)});prog();
</script></body></html>"""


def build() -> int:
    rows = read_csv_any(SHEET)
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    recl = reclassified_questions()
    scans = scan_warnings()
    cards = []
    for r in rows:
        evid = (answers.get(r["question"]) or {}).get("evidences", [])
        ev_text = "\n".join(e.get("text", "") for e in evid)
        ans_html, missing = mark_numbers(r["v1_answer"], ev_text)
        draft_html, _ = mark_numbers(r["draft_text"], ev_text)
        flags = []
        if missing:
            flags.append(f"근거에 없는 수치 {missing}")
        if r.get("not_found", "0") not in ("", "0"):
            flags.append(f"부분 답변(not_found={r['not_found']})")
        if r["question"].strip() in recl:
            flags.append("v1 상품일치 재분류 건")
        if r["draft_id"] in scans:
            flags.append("규칙 사전 점검 경고: " + ", ".join(sorted(set("|".join(scans[r["draft_id"]]).split("|")))))
        if r.get("difficulty") == "hard":
            flags.append("hard")
        ev_html = "".join(
            f'<div class="ev"><b>{html.escape(e.get("citation", ""))}</b> '
            f'<span class="meta">{html.escape(e.get("chunk_id", ""))}</span><br>'
            f'{highlight_in_evidence(e.get("text", ""), r["v1_answer"])}</div>' for e in evid)
        i = html.escape(r["draft_id"])
        radio = lambda k, lbl: (f'<div>{lbl} ' + "".join(   # noqa: E731
            f'<label><input type="radio" name="{k}_{i}" value="{v}" onchange="set(\'{i}\',\'{k}\',\'{v}\')"> {v}</label>'
            for v in ("Y", "N")) + "</div>")
        cards.append((len(flags) == 0, r["draft_id"], f"""
<section class="card" id="c_{i}" data-id="{i}">
<h3>{i} · {html.escape(r['question'])}</h3>
<div class="meta">{html.escape(r['doc_type'])} · {html.escape(r['difficulty'])} · 현재 시트 값: 정답={html.escape(r.get('human_is_correct','') or '-')}, 문체={html.escape(r.get('human_style_ok','') or '-')}</div>
<div class="flags">{''.join(f'<span>{html.escape(f)}</span>' for f in flags)}</div>
<div class="lab">v1 답변</div><div class="box">{ans_html}</div>
<div class="lab">구어체 초안</div><div class="box">{draft_html}</div>
<details{' open' if flags else ''}><summary>근거 원문 {len(evid)}개</summary>{ev_html}</details>
<div class="q">{radio('c', '정답 여부')}{radio('s', '문체 보존')}</div>
<textarea rows="1" placeholder="메모 (N 이면 이유)" oninput="st['{i}']=st['{i}']||{{}};st['{i}'].n=this.value;save()"></textarea>
</section>"""))
    cards.sort(key=lambda x: (x[0], x[1]))   # 플래그 있는 문항 먼저
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__CARDS__", "".join(c[2] for c in cards)), encoding="utf-8")
    n_flag = sum(not c[0] for c in cards)
    print(f"리뷰 페이지 생성: {OUT.relative_to(PROJECT_ROOT)}")
    print(f"  {len(cards)}건 중 우선 확인 {n_flag}건 (위쪽에 정렬)")
    webbrowser.open(OUT.as_uri())
    return 0


def do_import(path: Path) -> int:
    labels = {r["draft_id"]: r for r in read_csv_any(path)}
    rows = read_csv_any(SHEET)
    fields = list(rows[0].keys())
    changed = incomplete = 0
    for r in rows:
        lab = labels.get(r["draft_id"])
        if not lab or not lab.get("human_is_correct") or not lab.get("human_style_ok"):
            incomplete += 1
            continue
        before = (r["human_is_correct"], r["human_style_ok"])
        r["human_is_correct"] = lab["human_is_correct"].strip().upper()
        r["human_style_ok"] = lab["human_style_ok"].strip().upper()
        note = (lab.get("note") or "").strip()
        base = re.sub(r"\s*\[review:packet\]", "", r.get("note", "")).strip()
        r["note"] = " ".join(x for x in (base, note, "[review:packet]") if x)
        changed += before != (r["human_is_correct"], r["human_style_ok"])
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    yy = sum(r["human_is_correct"] == "Y" and r["human_style_ok"] == "Y" for r in rows)
    n_n = sum("N" in (r["human_is_correct"], r["human_style_ok"]) for r in rows)
    print(f"병합 완료: 값이 바뀐 문항 {changed}건, 미완료 {incomplete}건")
    print(f"  Y/Y {yy}건 · N 포함 {n_n}건 (N 은 52번에서 제외됨)")
    for r in rows:
        if "N" in (r["human_is_correct"], r["human_style_ok"]):
            print(f"  - {r['draft_id']} 정답={r['human_is_correct']} 문체={r['human_style_ok']} {r['note'][:80]}")
    return 0 if not incomplete else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imp", type=Path)
    args = ap.parse_args()
    return do_import(args.imp) if args.imp else build()


if __name__ == "__main__":
    sys.exit(main())
