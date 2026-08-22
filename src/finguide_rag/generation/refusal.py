"""거절(Refusal) 판정.

근거가 불충분할 때 답변을 생성하지 않고 거절한다.

설계 근거
--------
거절 평가셋 53건(거절 33 / 대조군 20)으로 신호를 측정하고, 임계값을
전수 탐색해 구조를 정했다.

유형별 검색 신호 분리도(top1 점수 AUC)

    time_variant     6건   0.804   점수로 잘 구별됨
    blank_value      9건   0.806   점수로 구별되나 방향이 불안정
    out_of_scope    11건   0.695   어느 정도 구별됨
    personalized     7건   0.586   거의 구별 안 됨

personalized 는 "제 신용등급이면 얼마까지 대출되나요" 처럼 올바른 문서를
높은 점수로 찾아온다. 개인 정보가 없어 답할 수 없을 뿐이므로 검색 신호로는
잡히지 않는다. 반면 문면에 1인칭 소유 표현이 있어 규칙으로 7/7 탐지하면서
대조군 오탐이 0 이었다. 신호가 통하지 않는 유형은 규칙으로, 규칙이 통하지
않는 유형은 신호나 LLM 으로 — 3단계 구조는 이 관찰에서 나왔다.

단계별 기여도 (실측, 53건)
------------------------
                       FAR    과잉거절   BalAcc   LLM 호출
    0단계만           0.333    0.000    0.833      0
    0+1단계           0.091    0.050    0.929      0
    0+1+2단계         0.030    0.100    0.935     22

비용이 0 인 두 단계만으로 FAR 0.091 까지 내려간다. LLM 은 남은 22건을
판정해 FAR 을 0.030 으로 낮추는 대신 과잉 거절을 0.050 올린다.

이전 판단의 정정
--------------
v1 시점에는 "검색 신호를 쓸수록 전체 성능이 떨어진다"고 기록했다.
당시 측정값은 0단계만 BalAcc 0.808 / 0+1단계 0.804 였다.

v2 에서 이 관찰은 뒤집혔다(0.833 -> 0.929). 원인은 두 가지다.

  1. personalized 판정에 1인칭 분기를 넣어 0단계 오탐이 사라졌다.
     "저는 통지를 받았나요"(개인 이력)와 "저는 언제부터 갚아야 하나요"
     (일반 규정)를 구별하게 되면서 대조군 오탐이 1 -> 0 이 됐다.
  2. "확실한 답변" 경로를 제거했다. top1 이 높다는 이유로 LLM 검증을
     건너뛰던 사례들이 이제 정상적으로 1단계를 거친다.

즉 v1 의 결론은 신호 자체의 한계가 아니라 0단계 오탐과 우회 경로가
만든 착시였다. 설계 근거가 된 관찰이 무효가 됐으므로 결론도 함께
폐기한다.

임계값 선택
---------
504조합(top1 9 x gap15 7 x blank 8)을 전체 파이프라인 기준으로 전수
평가했다. LLM 판정은 (질문, 근거)만으로 결정되고 근거는 임계값과
무관하므로, 질문당 1회만 호출해 캐시하면 모든 조합을 추가 비용 없이
평가할 수 있다(11,000콜 -> 53콜).

결과는 파레토 경계 2점으로 수렴했다.

    A  top1 0.75~0.90   FAR 0.030 (1/33)   과잉 0.100 (2/20)
    B  top1 0.50        FAR 0.061 (2/33)   과잉 0.050 (1/20)

총 오답은 3건으로 같다. A 와 B 의 차이는 False Answer 1건과 과잉 거절
1건의 교환이며, 데이터로는 결정할 수 없는 정책 판단이다. 금융 도메인의
비용 비대칭에 따라 A 를 택했다.

트레이드오프의 실체는 질문 두 건이다.

    디폴트옵션 위험등급 변경  top1 0.592  정답: 답변
    이용계약 해지 신용등급    top1 0.740  정답: 거절

top1 임계값이 두 값 사이 어디에 놓이냐가 전부다. 0.75 이상이면 둘 다
규칙에서 거절되어 신용등급 건의 False Answer 가 해소되고, 0.50 이하면
둘 다 LLM 으로 넘어가 디폴트옵션 건의 과잉 거절이 해소된다. 이전 값
0.60 은 그 사이에 있어 두 실패를 동시에 안고 있었다.

A 안 안에서는 16조합이 완전히 동일한 혼동행렬을 낸다(top1 0.75~0.90,
blank 0.05~0.20). 평가셋에서 구별되지 않는 값들이므로, 미지의 질의에서
가장 덜 공격적으로 동작하는 모서리를 골랐다. top1 은 낮을수록, blank 는
높을수록 하드 거절이 보수적이다.

비용 비대칭과 그 한계
------------------
금융 도메인에서 두 오류의 비용은 다르다.

    False Answer   근거 없이 답함  -> 불완전판매, 법적 책임
    Over-refusal   과잉 거절       -> 직원이 직접 찾음, 불편

탐색에는 cost = 2 x FAR + 과잉거절 을 썼다. 다만 이 "2"는 임의 선택이며
1.5 이하에서는 B 안이 선택된다. 단일 지표로 자동 결정되는 문제가 아니므로
가중치를 바꿔가며 최적 조합이 어떻게 이동하는지 확인한 뒤 파레토 경계에서
직접 골랐다. Balanced Accuracy 는 두 오류를 동등 가중하므로 선택 기준으로
쓰지 않고 보고용 지표로만 둔다.

표본 한계
--------
n=53 이므로 1건이 FAR 을 0.030, 과잉 거절을 0.050 움직인다. 위 수치는
이 정밀도 안에서 읽어야 하며, 소수점 셋째 자리의 차이는 통계적으로
구별되지 않는다.

잔여 실패 3건과 다음 개선 방향
---------------------------
  False Answer  연체가산율 질문. 조항은 정확히 찾았으나(top1 0.975) 값이
                문서에 없다. 공란 기호가 없어(blank 0.00) 규칙으로는 잡을
                수 없고, LLM 이 "안내 가능"으로 판정했다. 값을 묻는 질문일
                때 검증 기준을 엄격하게 분기하는 것이 남은 해법이다.
  과잉 거절     디폴트옵션 건(top1 0.592)은 위 트레이드오프의 다른 쪽이다.
                마이데이터 건(top1 0.812)은 LLM 이 과하게 엄격했다.

또한 신용등급 질문이 규칙에 걸린 것은 0.740 대 0.75, 마진 0.01 이다.
문서나 임베딩이 바뀌면 다시 새어나갈 수 있는 취약한 여유다.

v3 — 답변 가능 질문 96건에서 드러난 오탐
-----------------------------------
거절 평가셋 53건은 거절 케이스가 62% 인 적대적 구성이다. 그 위에서
조정된 규칙과 프롬프트는 거절 쪽으로 기울어 있었다.

답변 가능한 질문만 모은 96건으로 재보니 오거절률이 0.167 이었다.
답할 수 있는 질문 여섯 중 하나를 거절한다는 뜻이고, 환각(0.037),
상품 불일치(0.058), 수치 오류(0.027) 를 전부 합친 것의 두 배다.

16건을 전수 진단해 셋으로 나눴다.

    규칙 오탐        3건   비용 0 으로 수정 가능
    프롬프트 과엄격   5건   완화로 개선 가능
    근거 실제 부족    8건   검색 문제이므로 이 단계에서 해결 불가

앞의 8건을 겨냥해 세 가지를 고쳤다.

  1. RE_OWN 을 개인 속성과 계좌·거래로 분리했다.
     "내 계좌이체 시 보안매체 사용 여부는?" 이 계좌라는 말만으로
     personalized 가 됐다. 신용등급·소득·한도는 그 자체가 개인
     정보지만 계좌·거래는 누구에게나 있으므로, 개인화된 값을 묻는
     표현이 함께 있을 때만 거절한다.

  2. time_variant 에서 "다음 주/달/분기/년" 을 뺐다.
     "가입 후 최초 다음달까지 제공되는 우대서비스" 의 다음달은
     실시간 시점이 아니라 상품설명서에 적힌 기간 조건이다.
     "이번 주" 는 실제 거절 대상에 쓰이므로 유지했다.

  3. LLM 검증에 조건 반대 추론을 좁게 허용했다.
     근거에 "만 14세 이상만 이용 가능" 이 있고 질문자가 13살이면
     "이용할 수 없다" 고 안내할 수 있다. LLM 이 답을 찾아놓고
     거절한 사례가 있었다. 다만 근거 문장을 그대로 인용할 수 있는
     경우로 한정했다. 범위를 넓히면 근거에 없는 조건을 추론해
     덧붙이는 환각으로 이어진다.

이 수정은 오거절만 보고 한 것이므로, FAR 0.030 과 환각률 0.037 이
악화되지 않았는지 반드시 함께 재측정해야 한다. 한쪽만 보고 완화하면
근거 없는 답변이 늘어난다.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):
    ANSWER = "answer"
    REFUSE = "refuse"
    UNCERTAIN = "uncertain"   # 판단 보류 -> 다음 단계로


class RefusalReason(str, Enum):
    PERSONALIZED = "personalized"
    TIME_VARIANT = "time_variant"
    OUT_OF_SCOPE = "out_of_scope"
    BLANK_VALUE = "blank_value"
    LOW_CONFIDENCE = "low_confidence"
    NO_EVIDENCE = "no_evidence"


@dataclass
class RefusalResult:
    decision: Decision
    reason: RefusalReason | None = None
    stage: str = ""
    confidence: float = 0.0
    message: str = ""
    signals: dict = field(default_factory=dict)

    @property
    def should_refuse(self) -> bool:
        return self.decision == Decision.REFUSE


# ==================================================================
# 0단계 — 질문 패턴 (비용 0)
# ==================================================================

# 1인칭 소유 + 개인 속성.
#
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
#      거절 대상은 하나도 없다. 반면 96건 평가에서 오탐이 난 두 건은
#      모두 "내 계좌" 였다.
#
#        내 계좌이체 시 보안매체나 공동인증서 사용 여부는?   -> 일반 정책
#        내 계좌에 돈이 부족하면 이체가 안 되나요?           -> 일반 규정
#
#      1인칭 표현이 "제" 인지 "내" 인지가 실제로 갈린다.
#
#   2. "제" 와 개인 속성 사이에 수식어가 낀다. "제 개인 신용등급" 이
#      그 경우다. 한 어절까지 건너뛰도록 허용한다. 두 어절 이상 허용하면
#      "제 계좌 이체 한도" 같은 것이 다시 잡힌다.
#
# "내 + 개인속성" 을 완전히 버리지는 않는다. 평가셋에 사례가 없어 근거가
# 없을 뿐이므로, 계좌 경로로 보내 값을 물을 때만 거절한다. 53건이라는
# 표본 크기에서 오는 한계이며, 평가셋이 커지면 재검토해야 한다.
RE_OWN_ATTR = re.compile(
    r"제\s*(\S+\s+)?(신용|연체|거래|소득|급여|자산|한도|등급|기록|명의)"
)

# 계좌·거래를 가리키는 1인칭 표현. 이것만으로는 판정하지 않는다.
# 신용·등급·소득·한도를 여기에도 둔 것은 "내 신용등급" 처럼 평가셋에
# 없는 표현을 값 질문과 함께일 때만이라도 잡기 위해서다.
RE_OWN_ACCOUNT = re.compile(
    r"(제|내)\s*(계좌|통장|카드|예금|적금|대출|보유|신용|등급|소득|한도)"
)

# 1인칭 대명사 단독. 이것만으로는 판정할 수 없다.
#
#   "저는 사전지정운용방법 변경 통지를 받았나요?"  -> 개인 이력 조회 필요
#   "은행이 통보를 늦게 하면 저는 언제부터 갚아야 하나요?"  -> 일반 규정 질문
#
# 앞의 것은 개인 정보가 있어야 답할 수 있지만, 뒤의 것은 약관 조항으로
# 답할 수 있다. 대명사만 보고 거절하면 후자가 과잉 거절된다.
# 이 분기를 넣기 전에는 대조군 오탐이 1건 있었고, 넣은 뒤 0 이 됐다.
RE_FIRST_PERSON = re.compile(r"(제가|저는|내가|저에게|제게)\s")

# 개인화된 값이나 이력을 요구하는 표현.
# 1인칭 대명사나 계좌 표현과 함께 나타날 때만 personalized 로 본다.
#
# 금액·손실·예상을 추가했다. "제 거래 중도해지 시 손실 예상 금액은?" 이
# 기존 목록에 걸리지 않아 규칙을 좁혔을 때 새어나갔다.
RE_ASK_PERSONAL = re.compile(
    r"얼마(까지|나|인가)|한도는|몇\s*(퍼센트|%|원)"
    r"|받을\s*수\s*있|가능한(가|지)|해당(하나|되나|하는지)"
    r"|받았(나|는지)|되어\s*있(나|는지)|등록(됐|되었|돼)"
    r"|위험이\s*큰|적합한|맞나요"
    r"|금액은|손실|예상|얼마인가요"
)

# 일반 규정을 묻는 표현.
# 1인칭이 있어도 이 패턴이 함께 나오면 답변 가능한 질문으로 본다.
# 약관·설명서로 답할 수 있는 조건·절차 질문이기 때문이다.
RE_GENERAL_RULE = re.compile(
    r"언제(부터|까지|까지나)|어떻게\s*(되나|하나|해야)|무엇을|어떤\s*(서류|절차|조건|방법)"
    r"|문제가\s*생기|불이익|해야\s*하나|되나요\s*$"
)

# 시점 의존. 문서에는 산정 방식만 있고 현재 값은 없다.
#
# "다음 주/달/분기/년" 을 뺐다. 96건 평가에서 오탐이 나왔다.
#
#   가입 후 최초 다음달까지 제공되는 수수료 우대서비스는?
#
# 이때의 "다음달" 은 실시간 시점이 아니라 상품설명서에 적힌 기간
# 조건이다. 문서로 답할 수 있는 질문인데 시점 의존으로 잡혔다.
# "이번 주" 는 실제 거절 대상(이번 주 CD수익률)에 쓰이므로 유지한다.
RE_TIME_VARIANT = re.compile(
    r"현재|오늘|지금|이번\s*(주|달|분기|년)"
    r"|올해|내년|작년|최근|당장|실시간|요즘"
)

# 타행 비교나 내부 정책. 공개 문서로는 답할 수 없다.
RE_OUT_OF_SCOPE = re.compile(
    r"타\s*은행|타행|다른\s*은행|경쟁력|비교(해|하면|하여|해서)"
    r"|우수한|장점은|어떤\s*점이|내부\s*(심사|정책|기준|규정)|심사\s*기준"
    r"|추천(해|하나|할까)|좋을까요|어느\s*것이\s*나"
)


def check_question_pattern(question: str) -> RefusalResult | None:
    """질문 문면만으로 거절할 수 있는지 검사한다.

    검색과 LLM 호출 이전에 수행하므로 비용이 0 이다.

    실측 탐지율: personalized 7/7, time_variant 6/6, out_of_scope 9/11
    대조군 오탐 0/20. 이 단계에서 전체의 42%가 처리된다.

    out_of_scope 2건은 문면에 비교·내부정책 표현이 없어 여기서 걸리지
    않는다. 그중 1건(신용등급 영향)은 1단계 검색 신호가, 나머지 1건은
    LLM 이 잡는다.
    """
    q = question.strip()

    # 개인 속성을 직접 가리키면 그것만으로 개인 정보가 필요하다.
    is_personal = bool(RE_OWN_ATTR.search(q))

    # 계좌·거래를 가리키거나 1인칭 대명사만 있으면, 무엇을 묻는지 함께 본다.
    # 개인 이력이나 개인별 값을 요구하면 personalized, 일반 규정이면 통과.
    if not is_personal and (RE_OWN_ACCOUNT.search(q) or RE_FIRST_PERSON.search(q)):
        if RE_ASK_PERSONAL.search(q) and not RE_GENERAL_RULE.search(q):
            is_personal = True

    if is_personal:
        return RefusalResult(
            decision=Decision.REFUSE,
            reason=RefusalReason.PERSONALIZED,
            stage="pattern",
            confidence=0.85,
            message="고객님의 개인 정보가 필요한 문의입니다. "
                    "고객 정보를 조회한 뒤 안내해 주시기 바랍니다.",
        )

    if RE_TIME_VARIANT.search(q):
        return RefusalResult(
            decision=Decision.REFUSE,
            reason=RefusalReason.TIME_VARIANT,
            stage="pattern",
            confidence=0.80,
            message="현재 시점의 금리·환율 등은 문서에 담기지 않습니다. "
                    "실시간 조회 시스템에서 확인해 주시기 바랍니다.",
        )

    if RE_OUT_OF_SCOPE.search(q):
        return RefusalResult(
            decision=Decision.REFUSE,
            reason=RefusalReason.OUT_OF_SCOPE,
            stage="pattern",
            confidence=0.75,
            message="타행 비교나 내부 심사 기준은 공개 문서에 포함되지 않습니다.",
        )

    return None


# ==================================================================
# 1단계 — 검색 신호 (비용 0)
# ==================================================================

# 검색이 사실상 실패한 것으로 볼 기준.
#
# 504조합 전수 탐색에서 확정했다. top1 은 0.75~0.90 구간이, blank 는
# 0.05~0.20 구간이 모두 동일한 결과를 낸다(16조합 동률). 평가셋으로
# 구별되지 않는 값들이므로 그 구간에서 가장 보수적으로 동작하는 모서리를
# 택했다. top1 은 낮을수록, blank 는 높을수록 하드 거절이 덜 공격적이다.
#
# 두 조건을 AND 로 묶는 이유는, 점수가 낮아도 상위 문서 간 격차가 크면
# 검색이 실패한 것이 아니라 정답이 하나뿐인 경우이기 때문이다.
HARD_REFUSE_TOP1 = 0.75
HARD_REFUSE_GAP15 = 0.08

# 상위 청크에 공란이 이 비율 이상이면 blank_value 로 본다.
# 실측: 대조군 0.01 vs blank_value 0.40. 대조군에 거의 나타나지 않아
# 과잉 거절 위험이 낮다.
BLANK_RATIO_THRESHOLD = 0.2

RE_BLANK = re.compile(r"\(\s*\)|\(\s*%\s*\)|（\s*）|_{3,}|\.{5,}|☐|□")

# 값을 묻는 질문인지 판단한다.
# blank_value 는 "그 값이 얼마인가"를 물을 때만 성립한다.
# 같은 문서를 근거로 절차나 조건을 물으면 답할 수 있다.
RE_VALUE_QUESTION = re.compile(
    r"얼마|몇\s*(퍼센트|%|원|년|개월|일|회)|요율|이율|금액은|비용은|수수료는"
    r"|얼마인가|어떻게\s*되나|정해지나|산정되나"
)


def compute_signals(hits, chunk_texts: dict[str, str] | None = None) -> dict:
    """검색 결과에서 판정 신호를 뽑는다."""
    if not hits:
        return {"score_top1": 0.0, "gap_1_2": 0.0, "gap_1_5": 0.0,
                "std_top5": 0.0, "blank_ratio": 0.0, "n_hits": 0}

    scores = [h.score for h in hits]
    top5 = scores[:5]

    blank_ratio = 0.0
    if chunk_texts:
        n = min(5, len(hits))
        blank = sum(
            1 for h in hits[:5]
            if RE_BLANK.search(chunk_texts.get(h.chunk_id, ""))
        )
        blank_ratio = blank / n if n else 0.0

    return {
        "score_top1": scores[0],
        "gap_1_2": scores[0] - scores[1] if len(scores) > 1 else 0.0,
        "gap_1_5": scores[0] - scores[4] if len(scores) > 4 else 0.0,
        "std_top5": statistics.pstdev(top5) if len(top5) > 1 else 0.0,
        "blank_ratio": blank_ratio,
        "n_hits": len(hits),
    }


def check_retrieval_signals(signals: dict, question: str = "") -> RefusalResult:
    """검색 신호로 판정한다.

    확실한 거절만 결정하고 나머지는 UNCERTAIN 으로 넘긴다.
    "확실한 답변" 판정은 두지 않는다. 실측에서 이 경로가 LLM 검증을
    건너뛰게 만들어 False Answer 를 유발했기 때문이다.

    이 단계는 전체의 17%를 처리하며, 0단계만 쓸 때보다 FAR 을
    0.333 -> 0.091 로 낮춘다.
    """
    if signals["n_hits"] == 0:
        return RefusalResult(
            decision=Decision.REFUSE,
            reason=RefusalReason.OUT_OF_SCOPE,
            stage="retrieval",
            confidence=0.95,
            message="관련 문서를 찾지 못했습니다.",
            signals=signals,
        )

    # 검색이 사실상 실패한 경우
    if (signals["score_top1"] < HARD_REFUSE_TOP1
            and signals["gap_1_5"] < HARD_REFUSE_GAP15):
        return RefusalResult(
            decision=Decision.REFUSE,
            reason=RefusalReason.LOW_CONFIDENCE,
            stage="retrieval",
            confidence=0.85,
            message="문서에서 관련 근거를 충분히 찾지 못했습니다.",
            signals=signals,
        )

    # 공란 신호: 값을 묻는 질문인데 찾아온 문서에 값이 비어 있다.
    #
    # 두 조건을 함께 보는 이유는, 같은 서식 문서라도 "절차가 어떻게
    # 되나요" 같은 질문에는 답할 수 있기 때문이다. 공란이 있다는
    # 사실만으로 거절하면 과잉 거절이 늘어난다.
    #
    # 한계: 문서가 "은행이 정하는 바에 따른다"처럼 값을 아예 언급하지
    # 않으면 공란 기호가 없어 이 규칙이 동작하지 않는다. 실제로 남은
    # False Answer 1건이 이 경우다.
    if (signals["blank_ratio"] >= BLANK_RATIO_THRESHOLD
            and RE_VALUE_QUESTION.search(question)):
        return RefusalResult(
            decision=Decision.REFUSE,
            reason=RefusalReason.BLANK_VALUE,
            stage="retrieval",
            confidence=0.75,
            message="문서에 해당 항목은 있으나 구체적인 값이 기재되어 있지 "
                    "않습니다. 계약 조건에 따라 결정되므로 담당 부서에 확인해 "
                    "주시기 바랍니다.",
            signals=signals,
        )

    # 나머지는 모두 LLM 검증으로
    return RefusalResult(
        decision=Decision.UNCERTAIN,
        stage="retrieval",
        confidence=0.5,
        signals=signals,
    )


# ==================================================================
# 2단계 — LLM 근거 검증
# ==================================================================

# 프롬프트 설계 주의점
# -----------------
# 초기 버전은 "애매하면 REFUSE" 라고만 지시했는데, 과잉 거절이 25%까지
# 올랐다. 답할 수 있는 질문 5건을 no_evidence 로 거절한 것이다.
#
# 원인은 "완전한 답"을 요구한 데 있다. 실제 업무에서는 부분적인 근거라도
# 직원이 판단에 쓸 수 있으면 유용하다. 따라서 판정 기준을 "질문에 대한
# 답이 근거에 있는가"가 아니라 "직원이 이 근거로 고객에게 안내할 수
# 있는가"로 바꾼다.
#
# 대신 값이 공란인 경우는 명확히 거절하도록 별도 항목으로 강조한다.
#
# 남은 문제: 이 완화된 기준은 값을 묻는 질문에 취약하다. 조항을 정확히
# 찾아왔지만 그 안에 값이 없는 경우, LLM 은 주제 적합성만 보고
# ANSWERABLE 로 판정하는 경향이 있다(연체가산율 사례, top1 0.975).
# RE_VALUE_QUESTION 으로 값 질문을 이미 식별하고 있으므로, 그 경우에만
# "근거에 해당 수치가 명시되어 있는가"로 기준을 바꾸는 분기가 다음
# 개선안이다.

VERIFY_SYSTEM = """당신은 은행 직원용 문서 검색 시스템의 근거 검증기입니다.
질문과 검색된 문서 조각이 주어지면, 직원이 이 근거로 고객에게 안내할 수
있는지 판정합니다.

ANSWERABLE 로 판정하십시오:
- 질문이 묻는 내용이 근거에 들어 있다.
- 완전하지 않더라도 직원이 안내에 활용할 만한 구체적 정보가 있다.
- 조건, 절차, 기준, 범위 등이 제시되어 있다.
- 근거에 자격 요건이 명시되어 있고, 질문자가 그 요건을 충족하지 못하는
  것이 명백한 경우. "이용할 수 없다"는 것도 직원이 안내할 수 있는
  답변입니다.
    예) 근거 "만 14세 이상만 이용 가능" + 질문 "13살인데 이용할 수 있나요"
        -> ANSWERABLE. 근거 문장을 그대로 제시하며 불가하다고 안내할 수 있다.
  단, 근거 문장을 그대로 인용해 설명할 수 있는 경우에만 해당합니다.
  근거에 없는 조건을 추론해 덧붙이는 것은 허용하지 않습니다.

REFUSE 로 판정하십시오:
- 항목명이나 계산식만 있고 실제 값이 공란(괄호, 밑줄, 체크박스)이다.
- 근거가 질문과 다른 상품·다른 제도를 다룬다.
- 개인 정보나 실시간 시세가 있어야만 답할 수 있다.
- 근거에 질문과 관련된 내용이 사실상 없다.

판정 원칙:
- 근거가 부분적이어도 실무에 도움이 되면 ANSWERABLE 입니다.
- 다만 구체적인 '값'을 묻는데 그 값이 공란이면 REFUSE 입니다.

JSON으로만 답하십시오:
{"verdict": "ANSWERABLE" 또는 "REFUSE", "reason": "25자 이내"}"""

VERIFY_TEMPLATE = """[질문]
{question}

[검색된 문서 조각]
{evidence}

직원이 이 근거로 고객에게 안내할 수 있습니까?"""


def build_evidence(hits, chunk_texts: dict[str, str], top_n: int = 3,
                   max_chars: int = 700) -> str:
    """LLM 에게 넘길 근거 텍스트를 만든다.

    상위 3개만 넣는다. 토큰 비용을 줄이고, 관련도가 낮은 조각이 섞여
    판단을 흐리는 것을 막는다.

    이 함수의 출력은 검색 결과에만 의존하고 임계값과 무관하다. 임계값
    탐색에서 LLM 판정을 질문당 1회만 호출해 캐시할 수 있는 근거다.
    """
    parts = []
    for i, h in enumerate(hits[:top_n], 1):
        text = chunk_texts.get(h.chunk_id, "")[:max_chars]
        meta = getattr(h, "meta", {}) or {}
        header = " ".join(
            p for p in [meta.get("doc_display_name", ""), meta.get("section", "")] if p
        )
        parts.append(f"[{i}] {header}\n{text}")
    return "\n\n".join(parts)


def verify_with_llm(client, model: str, question: str, evidence: str,
                    prior: RefusalResult) -> RefusalResult:
    """LLM 으로 근거 충분성을 판정한다.

    호출에 실패하면 거절 쪽으로 기운다. 금융 도메인에서는 잘못된 답변이
    거절보다 비싸기 때문이다.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": VERIFY_TEMPLATE.format(
                    question=question, evidence=evidence
                )},
            ],
            temperature=0,
            max_tokens=60,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        verdict = str(data.get("verdict", "")).upper()
        reason_text = str(data.get("reason", ""))[:40]
        tokens = resp.usage.prompt_tokens + resp.usage.completion_tokens
    except Exception as exc:
        return RefusalResult(
            decision=Decision.REFUSE,
            reason=RefusalReason.NO_EVIDENCE,
            stage="llm_error",
            confidence=0.5,
            message="근거 확인 중 오류가 발생했습니다. 직접 확인해 주시기 바랍니다.",
            signals={**prior.signals, "error": str(exc)[:60]},
        )

    if verdict.startswith("ANSWER"):
        return RefusalResult(
            decision=Decision.ANSWER,
            stage="llm",
            confidence=0.80,
            signals={**prior.signals, "llm_reason": reason_text, "llm_tokens": tokens},
        )

    reason = prior.reason or RefusalReason.NO_EVIDENCE
    if reason == RefusalReason.BLANK_VALUE:
        message = ("문서에 해당 항목은 있으나 구체적인 값이 기재되어 있지 않습니다. "
                   "계약 조건에 따라 결정되므로 담당 부서에 확인해 주시기 바랍니다.")
    else:
        message = "문서에서 이 질문에 답할 근거를 찾지 못했습니다."

    return RefusalResult(
        decision=Decision.REFUSE,
        reason=reason,
        stage="llm",
        confidence=0.80,
        message=message,
        signals={**prior.signals, "llm_reason": reason_text, "llm_tokens": tokens},
    )


# ==================================================================
# 통합
# ==================================================================


class RefusalJudge:
    """다단계 거절 판정기.

    단계를 순서대로 거치며 확신이 서는 즉시 반환한다.
    LLM 호출은 마지막 단계에서만 일어난다.

    실측 판정 단계 분포(53건 기준)

        pattern      22건 (42%)   비용 0
        retrieval     9건 (17%)   비용 0
        llm          22건 (42%)   호출 발생

    이 분포는 거절 사례가 62%인 평가셋에서 측정한 것이다. 실제 트래픽은
    답변 가능한 질문이 다수이므로 LLM 호출 비중은 달라진다. 운영 비용
    추정에는 그대로 쓰지 않는다.
    """

    def __init__(self, client=None, model: str = "gpt-4.1-mini",
                 use_llm: bool = True, use_pattern: bool = True,
                 use_signals: bool = True):
        self.client = client
        self.model = model
        self.use_llm = use_llm and client is not None
        self.use_pattern = use_pattern
        self.use_signals = use_signals

    def judge(self, question: str, hits, chunk_texts: dict[str, str]) -> RefusalResult:
        # 0단계: 질문 패턴 (비용 0)
        if self.use_pattern:
            result = check_question_pattern(question)
            if result is not None:
                return result

        # 1단계: 검색 신호 (비용 0)
        signals = compute_signals(hits, chunk_texts)
        result = RefusalResult(decision=Decision.UNCERTAIN, signals=signals)
        if self.use_signals:
            result = check_retrieval_signals(signals, question)
            if result.decision != Decision.UNCERTAIN:
                return result

        # 2단계: LLM 근거 검증 (비용 발생)
        if not self.use_llm:
            # LLM 을 쓸 수 없으면 보수적으로 거절한다.
            return RefusalResult(
                decision=Decision.REFUSE,
                reason=result.reason or RefusalReason.LOW_CONFIDENCE,
                stage="retrieval_fallback",
                confidence=0.6,
                message="근거가 충분하지 않아 답변을 생성하지 않았습니다.",
                signals=signals,
            )

        evidence = build_evidence(hits, chunk_texts)
        return verify_with_llm(self.client, self.model, question, evidence, result)
