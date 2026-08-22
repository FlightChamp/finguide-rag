"""
39_check_patterns.py

0단계 질문 패턴 규칙을 고치기 전에 회귀를 잡는다.

왜 필요한가
---------
지난 시도에서 RE_OWN 을 개인 속성과 계좌·거래로 분리하다가 "제 거래"를
빠뜨렸다. 그 결과 "제 거래 중도해지 시 손실 예상 금액은?" 이 규칙을
빠져나가 FAR 이 0.030 에서 0.061 로 악화됐다.

원인은 검증 방식이었다. 손으로 고른 8건으로만 확인했고, 그 표본에
해당 유형이 없었다. 규칙을 고칠 때는 평가셋 전체를 통과시켜 봐야 한다.

이 스크립트는 refusal.py 를 수정하지 않는다. 현재 규칙과 제안 규칙을
같은 53건에 적용해 판정이 어떻게 달라지는지만 보여준다. 회귀가 있으면
배포하지 않는다.

LLM 을 호출하지 않으므로 비용이 0 이고, 몇 초면 끝난다.

무엇을 보는가
-----------
    현재 규칙  각 유형별 탐지율과 대조군 오탐
    제안 규칙  같은 지표
    변화       새로 잡힌 것(개선)과 놓친 것(회귀)을 문항 단위로

회귀가 한 건이라도 있으면 그 문항을 출력한다. 개선 건수가 많아도
회귀가 있으면 배포 판단은 사람이 해야 한다.

사용법
-----
    python scripts/39_check_patterns.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation import refusal as refusal_mod  # noqa: E402

EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "refusal_eval.csv"


# ==================================================================
# 제안 규칙
# ==================================================================
#
# 목표는 96건 평가에서 드러난 오탐 3건을 제거하되, 53건의 기존 탐지를
# 하나도 잃지 않는 것이다.
#
#   내 계좌이체 시 보안매체나 공동인증서 사용 여부는?    -> 통과시켜야 함
#   내 계좌에 돈이 부족하면 이체가 안 되나요?            -> 통과시켜야 함
#   가입 후 최초 다음달까지 제공되는 우대서비스는?        -> 통과시켜야 함
#
# 지난 시도의 실패를 반영해 두 가지를 바꿨다.
#
#   1. RE_OWN_ATTR 에 "거래" 를 되돌렸다. 빠뜨려서 회귀가 났던 부분이다.
#      다만 "내 계좌거래" 같은 표현과 구별하기 위해 제/내 + 거래 로 둔다.
#   2. RE_ASK_PERSONAL 에 금액·손실·예상을 추가했다. "손실 예상 금액은?"
#      처럼 개인별 값을 묻는 표현이 기존 목록에 없었다.

# 규칙을 추측하지 않고 평가셋의 personalized 7건에서 뽑았다.
#
#   제 연체 기록이 대출 연체 이자에 어떻게 영향을 미치나요?
#   제 거래 중도해지 시 손실 예상 금액은?
#   제 개인 신용등급으로 이 상품을 이용해도 위험이 큰가요?
#   제 신용등급과 거래내역을 고려한 CD 수익률은 얼마인가요?
#   제 개인 신용등급에 따른 대출 한도는?
#   제가 고정금리 지급자로서 이 거래에서 받을 수 있는 최대 변동금리 수취액은?
#   저는 사전지정운용방법 변경 통지를 받았나요?
#
# 두 가지가 보인다.
#
#   1. 7건이 전부 "제 / 제가 / 저는" 으로 시작한다. "내" 로 시작하는
#      거절 대상은 하나도 없다. 반면 96건에서 오탐이 난 두 건은 모두
#      "내 계좌" 였다. 1인칭 표현이 "제" 인지 "내" 인지가 실제로 갈린다.
#
#   2. "제" 와 개인 속성 사이에 수식어가 낀다. "제 개인 신용등급" 이
#      그 경우다. 지난 시도에서 이 형태를 고려하지 않아 회귀가 났다.
#      한 어절까지 건너뛰도록 허용한다. 두 어절 이상 허용하면
#      "제 계좌 이체 한도" 같은 것이 다시 잡힌다.
#
# "내 + 개인속성" 을 완전히 버리지는 않는다. 평가셋에 사례가 없어
# 근거가 없을 뿐이므로, 계좌 경로로 보내 값을 물을 때만 거절한다.
# 53건이라는 표본 크기에서 오는 한계이며, 표본이 커지면 재검토한다.

NEW_OWN_ATTR = re.compile(
    r"제\s*(\S+\s+)?(신용|연체|거래|소득|급여|자산|한도|등급|기록|명의)"
)

NEW_OWN_ACCOUNT = re.compile(
    r"(제|내)\s*(계좌|통장|카드|예금|적금|대출|보유|신용|등급|소득|한도)"
)

NEW_ASK_PERSONAL = re.compile(
    r"얼마(까지|나|인가)|한도는|몇\s*(퍼센트|%|원)"
    r"|받을\s*수\s*있|가능한(가|지)|해당(하나|되나|하는지)"
    r"|받았(나|는지)|되어\s*있(나|는지)|등록(됐|되었|돼)"
    r"|위험이\s*큰|적합한|맞나요"
    r"|금액은|손실|예상|얼마인가요"
)

# "다음 주/달/분기/년" 제거. 상품설명서의 기간 조건을 시점 의존으로
# 잘못 잡았다. "이번 주" 는 실제 거절 대상에 쓰이므로 유지한다.
NEW_TIME_VARIANT = re.compile(
    r"현재|오늘|지금|이번\s*(주|달|분기|년)"
    r"|올해|내년|작년|최근|당장|실시간|요즘"
)


def judge_current(q: str) -> str | None:
    """현재 refusal.py 의 0단계 판정."""
    r = refusal_mod.check_question_pattern(q)
    return r.reason.value if r and r.reason else None


def judge_proposed(q: str) -> str | None:
    """제안 규칙의 0단계 판정. 순서는 현재 구현과 동일하게 둔다."""
    q = q.strip()

    is_personal = bool(NEW_OWN_ATTR.search(q))
    if not is_personal and (NEW_OWN_ACCOUNT.search(q)
                            or refusal_mod.RE_FIRST_PERSON.search(q)):
        if (NEW_ASK_PERSONAL.search(q)
                and not refusal_mod.RE_GENERAL_RULE.search(q)):
            is_personal = True

    if is_personal:
        return "personalized"
    if NEW_TIME_VARIANT.search(q):
        return "time_variant"
    if refusal_mod.RE_OUT_OF_SCOPE.search(q):
        return "out_of_scope"
    return None


# 96건 평가에서 오탐으로 확인된 질문. 통과해야 한다.
FALSE_POSITIVES = [
    ("하나은행 내 계좌이체 시 보안매체나 공동인증서 사용 여부는?", "RE_OWN 오탐"),
    ("내 계좌에 돈이 부족하면 인터넷으로 돈 보내기가 안 되나요?", "RE_OWN 오탐"),
    ("사업자 주거래 우대통장 가입 후 최초 다음달까지 제공되는 수수료 우대서비스는 무엇인가요?",
     "time_variant 오탐"),
]


def main() -> None:
    print("=" * 76)
    print("0단계 패턴 규칙 회귀 검사 — 배포 전 확인")
    print("=" * 76)
    print("  refusal.py 를 수정하지 않습니다. 비교만 합니다.")

    if not EVAL_CSV.exists():
        sys.exit(f"{EVAL_CSV} 없음.")
    with EVAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    refuse = [r for r in rows if r["expected"] == "refuse"]
    control = [r for r in rows if r["expected"] == "answer"]
    print(f"  평가셋 {len(rows)}건 (거절 {len(refuse)} / 대조군 {len(control)})")

    # ---------- 유형별 탐지율 ----------
    print("\n" + "=" * 76)
    print("거절 대상 탐지율 — 0단계만")
    print("=" * 76)
    print(f"  {'유형':<16}{'건수':>6}{'현재':>10}{'제안':>10}{'변화':>8}")
    print("  " + "-" * 50)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in refuse:
        by_type[r["refusal_type"]].append(r)

    regressions, improvements = [], []
    for t in sorted(by_type):
        g = by_type[t]
        cur = sum(1 for r in g if judge_current(r["question"]) is not None)
        new = sum(1 for r in g if judge_proposed(r["question"]) is not None)
        mark = "" if cur == new else (f"  {new - cur:+d}")
        print(f"  {t:<16}{len(g):>6}{cur:>7}/{len(g):<2}{new:>7}/{len(g):<2}{mark:>8}")

    for r in refuse:
        c = judge_current(r["question"]) is not None
        n = judge_proposed(r["question"]) is not None
        if c and not n:
            regressions.append(r)
        elif n and not c:
            improvements.append(r)

    # ---------- 대조군 오탐 ----------
    print("\n" + "=" * 76)
    print("대조군 오탐 — 답변 가능한 질문을 거절하는가")
    print("=" * 76)
    cur_fp = [r for r in control if judge_current(r["question"]) is not None]
    new_fp = [r for r in control if judge_proposed(r["question"]) is not None]
    print(f"  현재 {len(cur_fp)}/{len(control)}   제안 {len(new_fp)}/{len(control)}")
    for r in cur_fp:
        print(f"    [현재] {judge_current(r['question']):<14} {r['question'][:46]}")
    for r in new_fp:
        print(f"    [제안] {judge_proposed(r['question']):<14} {r['question'][:46]}")
    if not cur_fp and not new_fp:
        print("    양쪽 모두 0건.")

    # ---------- 회귀 ----------
    print("\n" + "=" * 76)
    print("회귀 — 현재는 잡았으나 제안 규칙이 놓친 것")
    print("=" * 76)
    if regressions:
        for r in regressions:
            print(f"  [{r['refusal_type']}] {r['question'][:56]}")
            print(f"     현재 판정: {judge_current(r['question'])}")
        print(f"\n  회귀 {len(regressions)}건. 배포하면 FAR 이 악화된다.")
    else:
        print("  없음. 기존 탐지를 모두 유지한다.")

    if improvements:
        print("\n  [새로 잡힌 것]")
        for r in improvements:
            print(f"    [{r['refusal_type']}] {r['question'][:56]}")

    # ---------- 오탐 제거 확인 ----------
    print("\n" + "=" * 76)
    print("96건 평가에서 확인된 오탐 — 제거됐는가")
    print("=" * 76)
    fixed = 0
    for q, note in FALSE_POSITIVES:
        c, n = judge_current(q), judge_proposed(q)
        ok = c is not None and n is None
        fixed += ok
        mark = "해결" if ok else ("변화없음" if c == n else "주의")
        print(f"  [{mark}] {note}")
        print(f"     현재={c or '통과'} / 제안={n or '통과'}")
        print(f"     {q[:62]}")
    print(f"\n  {fixed}/{len(FALSE_POSITIVES)}건 해결")

    # ---------- 판정 ----------
    print("\n" + "=" * 76)
    print("배포 판단")
    print("=" * 76)
    if regressions:
        print("  배포하지 마십시오. 회귀가 있습니다.")
        print("  회귀 문항을 잡도록 규칙을 보완한 뒤 다시 검사하십시오.")
    elif len(new_fp) > len(cur_fp):
        print("  주의. 대조군 오탐이 늘었습니다.")
        print("  0단계 오탐은 과잉 거절로 직결되므로 재검토가 필요합니다.")
    elif fixed == 0:
        print("  변화가 없습니다. 규칙이 실제로 다르게 동작하는지 확인하십시오.")
    else:
        print(f"  배포 가능. 회귀 0건, 오탐 {fixed}건 해결.")
        print("  다만 0단계 변경은 LLM 에 도달하는 질문 집합을 바꾸므로,")
        print("  27번 재실행 시 '신규 호출' 건수를 확인하십시오. 그만큼은")
        print("  캐시가 없어 새 판정이 들어옵니다.")
    print("=" * 76)


if __name__ == "__main__":
    main()
