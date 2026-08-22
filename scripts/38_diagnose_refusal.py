"""
38_diagnose_refusal.py

오거절 16건이 왜 거절됐는지 진단한다.

배경
----
96건 전체 평가에서 오거절률 0.167 이 나왔다. 답할 수 있는 질문 여섯 중
하나를 거절한다는 뜻이고, 환각(0.037), 상품 불일치(0.058), 수치 오류
(0.027) 를 전부 합친 것의 두 배다. 개선 여지가 사실상 여기 하나다.

단계별 분포가 범인을 지목한다.

    stage=llm         10건   2단계 LLM 근거 검증
    stage=retrieval    3건   1단계 검색 신호
    stage=pattern      3건   0단계 질문 패턴

왜 이런 편향이 생겼나
------------------
2단계 프롬프트는 거절 평가셋 53건에 맞춰 조정됐다. 그 평가셋은 거절
케이스가 62% 인 적대적 구성이라, 프롬프트가 거절 쪽으로 기울도록
최적화됐다. 실제 트래픽은 반대로 답할 수 있는 질문이 다수다.
96건(전부 답변 가능)만 모아 보니 그 편향이 드러났다.

이 스크립트가 하는 일
------------------
프롬프트를 감으로 고치지 않기 위해, 거절된 16건이 실제로 어떤 근거를
받았고 LLM 이 무슨 이유를 댔는지 전부 출력한다. 아무것도 수정하지 않는다.

    - 0단계: 어떤 패턴 규칙에 걸렸는지
    - 1단계: top1, gap, blank 신호값
    - 2단계: LLM 이 남긴 llm_reason 과 실제로 받은 근거 원문

진단 후 판단할 것
--------------
    근거가 충분한데 거절했다        -> 프롬프트가 과하게 엄격
    근거가 실제로 부족했다          -> 검색 문제, 생성 단계로 해결 불가
    질문 유형이 규칙에 잘못 걸렸다   -> 패턴 규칙 수정

사용법
-----
    python scripts/38_diagnose_refusal.py
    python scripts/38_diagnose_refusal.py --full   # 근거 원문 전체 출력
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation import refusal as refusal_mod  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports"
ANSWER_CACHE = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"


def latest(pattern: str) -> Path | None:
    files = sorted(REPORT_DIR.glob(pattern))
    return files[-1] if files else None


def recover_reason(client, model: str, question: str, evidences: list[dict]) -> str:
    """LLM 이 거절한 이유를 다시 물어 복구한다.

    Answer.to_dict() 는 llm_reason 을 저장하지 않는다. 진단에는 이것이
    가장 중요한 정보이므로 해당 문항만 다시 호출한다.

    주의: 근거 블록을 캐시된 evidences 로 재구성하므로 원래 호출과
    완전히 같지 않을 수 있다. 헤더가 citation 기반이라 원본의
    doc_display_name + section 조합과 미세하게 다르다. 판정이 뒤집힐
    가능성은 낮지만, 여기서 얻은 사유는 참고용이다.
    """
    block = "\n\n".join(
        f"[{e['rank']}] {e.get('citation', '')}\n{e.get('text', '')[:700]}"
        for e in evidences
    ) or "(근거 없음)"
    prior = refusal_mod.RefusalResult(decision=refusal_mod.Decision.UNCERTAIN)
    try:
        r = refusal_mod.verify_with_llm(client, model, question, block, prior)
    except Exception as exc:
        return f"(복구 실패: {str(exc)[:50]})"
    verdict = "REFUSE" if r.decision == refusal_mod.Decision.REFUSE else "ANSWERABLE"
    why = (r.signals or {}).get("llm_reason", "")
    return f"{verdict} — {why}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="근거 원문을 자르지 않고 출력")
    ap.add_argument("--no-recover", action="store_true",
                    help="LLM 사유 복구를 건너뛴다 (비용 0)")
    ap.add_argument("--llm-model", default="gpt-4.1-mini")
    args = ap.parse_args()

    print("=" * 76)
    print("오거절 진단 — 답변 가능한 질문이 왜 거절됐는가")
    print("=" * 76)

    path = latest("g_eval_full_*.csv")
    if not path:
        sys.exit("reports/g_eval_full_*.csv 없음. 36번을 먼저 실행하세요.")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not ANSWER_CACHE.exists():
        sys.exit(f"{ANSWER_CACHE} 없음.")
    answers = json.loads(ANSWER_CACHE.read_text(encoding="utf-8"))

    fr = [r for r in rows if str(r.get("false_refusal", "")).lower() == "true"]
    print(f"  전체 {len(rows)}건 / 오거절 {len(fr)}건 = {len(fr) / len(rows):.3f}")

    by_stage = Counter(r["stage"] for r in fr)
    print(f"\n  단계별: " + " / ".join(f"{k} {v}" for k, v in by_stage.most_common()))

    print(f"\n  현재 임계값: top1={refusal_mod.HARD_REFUSE_TOP1} "
          f"gap15={refusal_mod.HARD_REFUSE_GAP15} "
          f"blank={refusal_mod.BLANK_RATIO_THRESHOLD}")

    limit = 100000 if args.full else 420

    # LLM 단계 거절 사유 복구
    reasons: dict[str, str] = {}
    llm_group = [r for r in fr if r["stage"] == "llm"]
    if llm_group and not args.no_recover:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI()
        print(f"\n  LLM 거절 사유 복구 {len(llm_group)}건 호출 중...")
        for r in llm_group:
            item = answers.get(r["question"], {})
            reasons[r["question"]] = recover_reason(
                client, args.llm_model, r["question"], item.get("evidences", []))
        print(f"  복구 완료. 근거 블록을 재구성했으므로 참고용이다.")

    for stage in ("pattern", "retrieval", "llm"):
        group = [r for r in fr if r["stage"] == stage]
        if not group:
            continue

        print("\n" + "=" * 76)
        title = {"pattern": "0단계 질문 패턴", "retrieval": "1단계 검색 신호",
                 "llm": "2단계 LLM 근거 검증"}[stage]
        print(f"{title} — {len(group)}건")
        print("=" * 76)

        for r in group:
            q = r["question"]
            item = answers.get(q, {})
            sig = item.get("signals", {})
            print(f"\n  ── {q}")
            print(f"     난이도 {r.get('difficulty', '?')} / "
                  f"{r.get('doc_type', '?')} / 사유 {item.get('refusal_reason', '?')}")

            if stage == "pattern":
                # 어떤 규칙에 걸렸는지 직접 확인한다.
                res = refusal_mod.check_question_pattern(q)
                reason = res.reason.value if res and res.reason else "?"
                print(f"     걸린 규칙: {reason}")
                hits = []
                for name, rx in [("RE_OWN_ATTR", refusal_mod.RE_OWN_ATTR),
                                 ("RE_OWN_ACCOUNT", refusal_mod.RE_OWN_ACCOUNT),
                                 ("RE_FIRST_PERSON", refusal_mod.RE_FIRST_PERSON),
                                 ("RE_ASK_PERSONAL", refusal_mod.RE_ASK_PERSONAL),
                                 ("RE_GENERAL_RULE", refusal_mod.RE_GENERAL_RULE),
                                 ("RE_TIME_VARIANT", refusal_mod.RE_TIME_VARIANT),
                                 ("RE_OUT_OF_SCOPE", refusal_mod.RE_OUT_OF_SCOPE)]:
                    m = rx.search(q)
                    if m:
                        hits.append(f"{name}='{m.group(0)}'")
                print(f"     매칭: {' / '.join(hits) or '(없음)'}")

            evs = item.get("evidences", [])
            if stage in ("retrieval", "llm") and evs:
                top = evs[0]
                print(f"     top1={top.get('score', 0):.3f} 근거 {len(evs)}건")

            if stage == "llm":
                rec = reasons.get(q) or item.get("llm_reason") or "(기록 없음)"
                print(f"     LLM 사유: {rec}")

            if evs:
                print(f"     근거:")
                for e in evs:
                    print(f"       [{e['rank']}] {e.get('doc_type', '?')} "
                          f"{e.get('citation', '')[:58]} "
                          f"(score={e.get('score', 0):.3f})")
                    body = " ".join(e.get("text", "").split())[:limit]
                    print(f"           {body}")
            else:
                print(f"     근거 없음")

    # ---------- 요약 ----------
    print("\n" + "=" * 76)
    print("판단 가이드")
    print("=" * 76)
    print("  위 근거를 읽고 각 건을 셋 중 하나로 분류하십시오.")
    print()
    print("    (1) 근거가 충분한데 거절  -> 프롬프트가 과하게 엄격. 수정 가능")
    print("    (2) 근거가 실제로 부족    -> 검색 문제. 생성 단계로 해결 불가")
    print("    (3) 규칙에 잘못 걸림      -> 패턴 규칙 수정. 비용 0")
    print()
    print("  (1) 이 많으면 2단계 프롬프트를 완화한다. 다만 완화하면 근거 없는")
    print("  답변이 늘 수 있으므로, 수정 후 27번(FAR 0.030)과 36번(환각 0.037)을")
    print("  함께 재측정해야 한다. 오거절만 보고 고치면 다른 지표가 나빠진다.")

    out = REPORT_DIR / f"false_refusal_diag_{date.today().isoformat()}.csv"
    recs = []
    for r in fr:
        item = answers.get(r["question"], {})
        recs.append({
            "question": r["question"], "stage": r["stage"],
            "difficulty": r.get("difficulty", ""), "doc_type": r.get("doc_type", ""),
            "refusal_reason": item.get("refusal_reason", ""),
            "llm_reason": reasons.get(r["question"], item.get("llm_reason", "")),
            "n_evidence": len(item.get("evidences", [])),
            "top1": (item.get("evidences") or [{}])[0].get("score", ""),
            "top_citation": (item.get("evidences") or [{}])[0].get("citation", ""),
            "분류": "",
        })
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"\n저장 → {out.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
