"""
33_calibrate_judge.py

목적
----
사람 검수 판정과 LLM 초안 판정을 대조해, 자동 평가기를 어디까지
신뢰할 수 있는지 정량화한다.

배경
----
G 평가셋 25건의 초안 판정은 환각 20/20 pass, 상품일치 fail 0건이었다.
사람 검수 결과는 환각 fail 3건, 상품일치 fail 3건이다.

이 프로젝트에서 LLM 이 상품 불일치를 놓친 것은 이번이 세 번째다.

    생성기   정기예금 질문에 주택청약예금 근거로 답변 (데모 첫 실행)
    검증기   같은 사례를 ANSWERABLE 로 통과 (2단계 근거 검증)
    평가기   같은 사례를 pass 로 판정 (G 초안)

같은 모델에 비슷한 프롬프트를 주면 판정자도 생성자와 같은 맹점을
갖는다. 자동 평가 수치를 검증 없이 신뢰하면 이 실패는 영원히 보이지
않는다. 그래서 사람 판정으로 평가기를 먼저 재는 순서를 택했다.

측정 항목
--------
1. 항목별 일치율과 Cohen's kappa
2. 놓친 실패(사람 fail / 초안 pass) — 가장 중요한 수치
3. 과잉 탐지(사람 pass / 초안 fail)
4. 초안 편향 — 백지 5건과 초안 20건에서 사람의 fail 비율이 다른가

주의
----
초안이 있는 문항은 20건뿐이다. kappa 는 이 크기에서 매우 불안정하며,
한 건이 바뀌면 크게 움직인다. 방향을 보는 용도로만 쓴다.
백지 5건으로 편향을 판정하는 것은 더 약하다. 신호가 보이면 표본을
늘려 다시 재야 한다.

사용법
-----
    python scripts/33_calibrate_judge.py
    python scripts/33_calibrate_judge.py --review data/eval/g_eval_final.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
REPORT_DIR = PROJECT_ROOT / "reports"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_eval_answers.json"

CRITERIA = ["환각", "상품일치", "수치정확", "미확인신고", "실무활용"]
VALID = {"pass", "fail", "na"}

# | 01 | 정상/FAQ | na | na | na | fail | fail | 메모 |
ROW = re.compile(
    r"^\|\s*(\d{1,2})\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|(.*)\|\s*$",
    re.M,
)

# 원본 템플릿 형식도 지원한다: "- 환각 판정: pass"
LINE = re.compile(r"-\s*(\S+?)\s*판정:\s*(\w+|\?)")
ITEM = re.compile(r"<!--\s*ITEM\s*(\d+)\s*-->")


def parse_table(text: str) -> dict[str, dict]:
    """판정표 형식을 읽는다."""
    out: dict[str, dict] = {}
    for m in ROW.findall(text):
        num, kind = m[0].strip(), m[1].strip()
        verdicts = [v.strip().lower() for v in m[2:7]]
        if not all(v in VALID for v in verdicts):
            continue
        out[f"{int(num):02d}"] = {
            "kind": kind,
            "memo": m[7].strip(),
            **dict(zip(CRITERIA, verdicts)),
        }
    return out


def parse_template(text: str) -> dict[str, dict]:
    """원본 체크 템플릿 형식을 읽는다."""
    out: dict[str, dict] = {}
    blocks = ITEM.split(text)
    for i in range(1, len(blocks), 2):
        iid = f"{int(blocks[i]):02d}"
        rec: dict = {"kind": "", "memo": ""}
        for name, val in LINE.findall(blocks[i + 1]):
            if name in CRITERIA:
                rec[name] = val.strip().lower()
        if any(c in rec for c in CRITERIA):
            out[iid] = rec
    return out


def load_review(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8")
    table = parse_table(text)
    tmpl = parse_template(text)
    # 판정값이 더 많이 채워진 쪽을 쓴다.
    def filled(d):
        return sum(1 for r in d.values()
                   for c in CRITERIA if r.get(c) in VALID)
    return table if filled(table) >= filled(tmpl) else tmpl


def kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa. 표본이 작으면 매우 불안정하다."""
    n = len(pairs)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    labels = {x for p in pairs for x in p}
    pe = sum((sum(1 for a, _ in pairs if a == l) / n)
             * (sum(1 for _, b in pairs if b == l) / n) for l in labels)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default=None,
                    help="검수 마크다운 경로 (기본: data/eval 에서 자동 탐색)")
    args = ap.parse_args()

    print("=" * 76)
    print("자동 평가기 캘리브레이션 — 사람 판정 대조")
    print("=" * 76)

    # --- 검수 파일 ---
    if args.review:
        review_path = Path(args.review)
        if not review_path.is_absolute():
            review_path = PROJECT_ROOT / review_path
    else:
        cands = sorted(EVAL_DIR.glob("g_eval*.md"))
        if not cands:
            sys.exit(f"{EVAL_DIR} 에서 g_eval*.md 를 찾지 못했습니다. "
                     f"--review 로 경로를 지정하세요.")
        review_path = cands[-1]
    if not review_path.exists():
        sys.exit(f"{review_path} 없음.")

    human = load_review(review_path)
    print(f"  검수 파일  {review_path.name} — {len(human)}건")

    if not ANSWERS.exists():
        sys.exit(f"{ANSWERS} 없음. 32번을 먼저 실행하세요.")
    items = {it["id"]: it for it in json.loads(ANSWERS.read_text(encoding="utf-8"))}
    print(f"  초안 파일  {ANSWERS.name} — {len(items)}건")

    missing = sorted(set(items) - set(human))
    if missing:
        print(f"  주의: 검수에 없는 문항 {len(missing)}건 -> {', '.join(missing)}")

    drafted = [i for i in human if i in items and not items[i]["blind"]]
    blind = [i for i in human if i in items and items[i]["blind"]]
    print(f"  초안 있음 {len(drafted)}건 / 백지 {len(blind)}건")

    # --- 1. 항목별 일치 ---
    print("\n" + "=" * 76)
    print("항목별 일치율 (초안이 있는 문항만)")
    print("=" * 76)
    print(f"  {'항목':<12}{'일치':>8}{'일치율':>9}{'kappa':>9}"
          f"{'놓친 실패':>11}{'과잉 탐지':>11}")
    print("  " + "-" * 60)

    rows_out = []
    total_missed = 0
    for c in CRITERIA:
        pairs = []
        missed, over = [], []
        for i in drafted:
            h = human[i].get(c)
            d = str(items[i]["draft"].get(c, "?")).lower()
            if h not in VALID or d not in VALID:
                continue
            pairs.append((d, h))
            if h == "fail" and d != "fail":
                missed.append(i)
            if d == "fail" and h != "fail":
                over.append(i)
        if not pairs:
            continue
        agree = sum(1 for a, b in pairs if a == b)
        k = kappa(pairs)
        total_missed += len(missed)
        print(f"  {c:<12}{agree:>4}/{len(pairs):<3}{agree / len(pairs):>9.2f}"
              f"{k:>9.2f}{len(missed):>11}{len(over):>11}")
        rows_out.append({
            "criterion": c, "n": len(pairs), "agree": agree,
            "agreement": round(agree / len(pairs), 3), "kappa": round(k, 3),
            "missed_fail": len(missed), "over_flag": len(over),
            "missed_ids": "|".join(missed), "over_ids": "|".join(over),
        })

    print(f"\n  놓친 실패 = 사람이 fail 인데 초안이 잡지 못한 건수. 총 {total_missed}건")
    print(f"  과잉 탐지 = 초안이 fail 인데 사람은 아닌 건수")
    print(f"  주의: 표본 {len(drafted)}건. kappa 는 이 크기에서 불안정하므로")
    print(f"        절대값보다 항목 간 상대 비교로 읽는다.")

    # --- 2. 놓친 실패 상세 ---
    print("\n" + "=" * 76)
    print("놓친 실패 상세 — 자동 평가기가 보지 못한 것")
    print("=" * 76)
    any_missed = False
    for i in sorted(drafted):
        miss = [c for c in CRITERIA
                if human[i].get(c) == "fail"
                and str(items[i]["draft"].get(c, "?")).lower() != "fail"]
        if not miss:
            continue
        any_missed = True
        it = items[i]
        print(f"\n  [{i}] {it['group']} — {it['question'][:52]}")
        print(f"      놓친 항목: {', '.join(miss)}")
        print(f"      시스템: {it['decision']} (stage={it['stage']})")
        if it["draft"].get("근거"):
            print(f"      초안 근거: {it['draft']['근거']}")
        if human[i].get("memo"):
            print(f"      사람 메모: {human[i]['memo'][:88]}")
    if not any_missed:
        print("  없음.")

    # --- 3. 초안 편향 ---
    print("\n" + "=" * 76)
    print("초안 편향 확인 — 백지 문항과 사람 판정 분포 비교")
    print("=" * 76)

    def fail_rate(ids: list[str]) -> tuple[int, int]:
        f = t = 0
        for i in ids:
            for c in CRITERIA:
                v = human[i].get(c)
                if v in {"pass", "fail"}:
                    t += 1
                    f += (v == "fail")
        return f, t

    fd, td = fail_rate(drafted)
    fb, tb = fail_rate(blind)
    print(f"  {'구분':<14}{'문항':>6}{'판정칸':>8}{'fail':>7}{'fail 비율':>11}")
    print("  " + "-" * 46)
    if td:
        print(f"  {'초안 있음':<14}{len(drafted):>6}{td:>8}{fd:>7}{fd / td:>11.2f}")
    if tb:
        print(f"  {'백지':<14}{len(blind):>6}{tb:>8}{fb:>7}{fb / tb:>11.2f}")

    if td and tb:
        diff = (fb / tb) - (fd / td)
        print(f"\n  차이 {diff:+.2f}")
        if abs(diff) < 0.10:
            print("  -> 뚜렷한 편향 신호 없음. 다만 백지가 "
                  f"{len(blind)}건뿐이라 결론으로 삼기에는 약하다.")
        else:
            direction = "백지에서 더 엄격" if diff > 0 else "백지에서 더 관대"
            print(f"  -> {direction}. 초안 제시가 판정을 끌어당겼을 가능성이 있다.")
            print(f"     표본을 늘려 재확인할 것.")

    # --- 4. 그룹별 실패 ---
    print("\n" + "=" * 76)
    print("그룹별 사람 판정 (전체 문항)")
    print("=" * 76)
    by_group: dict[str, list[str]] = defaultdict(list)
    for i in human:
        if i in items:
            by_group[items[i]["group"]].append(i)

    print(f"  {'그룹':<16}{'문항':>6}{'전항목 pass':>12}{'fail 포함':>11}")
    print("  " + "-" * 45)
    for g in sorted(by_group):
        ids = by_group[g]
        clean = sum(1 for i in ids
                    if not any(human[i].get(c) == "fail" for c in CRITERIA))
        print(f"  {g:<16}{len(ids):>6}{clean:>12}{len(ids) - clean:>11}")

    print("\n  [항목별 사람 판정 분포 — 전체]")
    for c in CRITERIA:
        cnt = Counter(human[i].get(c, "?") for i in human)
        print(f"    {c:<12}" + "  ".join(
            f"{k} {v}" for k, v in sorted(cnt.items())))

    # --- 5. 저장 ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"judge_calibration_{date.today().isoformat()}.csv"
    detail = REPORT_DIR / f"g_eval_labels_{date.today().isoformat()}.csv"

    if rows_out:
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n저장 → {out.relative_to(PROJECT_ROOT)}")

    merged = []
    for i in sorted(human):
        it = items.get(i, {})
        rec = {
            "id": i,
            "group": it.get("group", ""),
            "question": it.get("question", ""),
            "decision": it.get("decision", ""),
            "stage": it.get("stage", ""),
            "blind": it.get("blind", ""),
            "memo": human[i].get("memo", ""),
        }
        for c in CRITERIA:
            rec[f"human_{c}"] = human[i].get(c, "")
            rec[f"draft_{c}"] = str(it.get("draft", {}).get(c, "")).lower()
        merged.append(rec)
    with detail.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)
    print(f"저장 → {detail.relative_to(PROJECT_ROOT)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
