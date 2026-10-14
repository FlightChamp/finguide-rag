# FinGuide-RAG v2-B — 쉬운 설명과 역검증

> 공개 문서 기반 개인 포트폴리오 프로젝트의 설계 기록입니다. 어떤 금융기관의 공식 서비스가 아닙니다.

## B 가 하는 일과 하지 않는 일

B 는 **A 에서 '안내 가능(OK)'으로 판정된 내용만** 고객이 이해하기 쉬운 말로 옮기고, 옮긴 결과의 뜻이 원문과 같은지 다시 확인한다.

B 는 틀린 답을 고치는 장치가 아니다. 입력이 틀렸으면 A 가 막아야 하고, B 는 표현만 바꾼다.

| claim 판정 (A) | B 에서의 처리 | 고객에게 보이는가 |
|---|---|---|
| OK | 쉬운 설명으로 옮김 (LLM) | 역검증 **PASS 일 때만** |
| CHECK | '직원 확인 필요' 영역에 **원문 그대로** (LLM 을 거치지 않음) | 아니오 |
| PROHIBITED | 고객 안내문 생성 대상에서 제외, 직원 화면 '제외' 목록 | 아니오 |
| 판정 없음 | CHECK 로 취급 (fail-closed) | 아니오 |

## 출력 스키마 (`schema.py`, `SCHEMA_VERSION = v2.0-draft4`)

```
SimplifiedExplanation
  core:          [SimpleItem]        핵심 설명 — OK claim 만. 문구마다 출처 claim_ids
  conditions:    [SimpleItem]        꼭 확인할 조건·예외 — 문구마다 condition_ids
  staff_check:   [StaffCheckItem]    직원 확인 필요 — CHECK claim 원문 + 사유 코드 (코드가 만듦)
  excluded:      [claim_id]          PROHIBITED claim (코드가 만듦)
  required_conditions: [RequiredCondition]   지켜야 할 조건 목록
BackVerification
  status: PASS | NEEDS_FIX | BLOCK,  problems: [BVProblem(code, detail, item_id)],
  condition_results: {cond_id: PRESERVED|DROPPED|ALTERED|NO_RESULT},  attempts,  customer_visible
```

**지켜야 할 조건 목록은 Simplifier 가 정하지 않는다.** A 의 claim 추출 단계에서 나온 `Claim.conditions`(초안 표기 그대로)를 쓴다. 쉬운 설명을 만드는 쪽이 목록을 정하면, 빠뜨린 조건은 목록에서도 같이 빠지기 때문이다.

## 역검증 — 비교 대상은 'OK claim 원문'

OK claim 은 이미 근거 문서와 대조를 마쳤다. 역검증은 **옮기는 동안 뜻이 바뀌었는가**만 본다.

| 검사 | 방식 | 문제 코드 |
|---|---|---|
| 수치 보존 | 규칙 — A 의 `numeric.py` 재사용. 값·단위·연/월·비교어·상한이 그대로인가. 범위를 한 값으로 줄여도 변경 | `NUMERIC_CHANGED` · `NUMERIC_ADDED` · `NUMERIC_DROPPED` |
| 상품 보존 | 규칙 — 원문 상품명 누락, 원문에 없는 카탈로그 상품명 추가 | `PRODUCT_DROPPED` · `PRODUCT_ADDED` |
| 기관 보존 | 규칙 — 원문·근거에 없는 기관명 (A 의 기관 별칭표 재사용) | `INSTITUTION_ADDED` |
| 단정 강화 | 규칙 — 강조어 추가(A 와 같은 동의어 표), 상한 누락, 가능성·한정 표현 소실(claim 단위) | `STRENGTHENED` · `HEDGE_LOST` |
| 구조 | 규칙 — 출처 id, CHECK·PROHIBITED 내용 유입(id·문구 겹침), OK claim 누락, 직원 확인 항목 누락 | `CHECK_LEAK` · `PROHIBITED_LEAK` · `UNKNOWN_SOURCE` · `CLAIM_UNCOVERED` · `LOST_CHECK` |
| 조건·예외 보존 | **LLM 이 조건마다 판단, 코드가 판정.** LLM 은 PRESERVED/DROPPED/ALTERED 와 '어느 문구에 있는지(문구 id)'를 낸다. PRESERVED 인데 위치를 못 대면 보존으로 치지 않는다 | `CONDITION_DROPPED` · `CONDITION_ALTERED` · `CONDITION_UNVERIFIED` |

목록 밖 조건(LLM 이 추가로 '빠졌다'고 한 것)은 **원문 claim 에 실제로 있는 표현을 인용했을 때만** 받는다. LLM 이 지어낸 조건으로 차단하지 않기 위해서다.

## 판정 규칙 (`policy.decide_bv_status`)

```
문제 없음                                  → PASS     (고객용 쉬운 설명 노출 가능)
고칠 수 없는 문제가 하나라도                 → BLOCK    (수정 시도 없음)
    SIMPLIFIER_ERROR · BV_ERROR · LOST_CHECK — 검증 자체가 안 된 상태에서 고쳐 봐야 고친 것도 검증할 수 없다
그 외 (1차)                                → NEEDS_FIX → 문제 목록을 주고 1회 수정 → 재검증
재검증에서도 문제                            → BLOCK
```

BLOCK 이면 고객용 쉬운 설명은 숨기고, 직원 화면에 OK claim **원문**을 대체 안내로 보여 준다. 반복 수정 루프는 없다 — 시도 횟수에 따라 결과가 달라지는 구조는 감사가 어렵다.

## 평가 계획

### 지표

| 지표 | 정의 | 성격 |
|---|---|---|
| **False Pass Rate** | 뜻이 바뀐 쉬운 설명 중 역검증 PASS 비율 | **핵심 안전 지표** (낮을수록 좋음) |
| Over-block Rate | 뜻이 그대로인 쉬운 설명 중 PASS 가 아닌 비율 | 사용성 (낮을수록 좋음) |
| Condition Preservation | 조건 삭제를 잡은 비율 / PASS 된 설명에서 조건이 보존된 비율 | |
| Numeric · Product Preservation | 수치·상품 교란을 잡은 비율 / PASS 된 설명에서 보존된 비율 | |

모든 비율은 1차 역검증 기준으로 내고, 1회 수정 경로(NEEDS_FIX → PASS/BLOCK)는 따로 센다. 신뢰구간은 Wilson 95%.

### 개발 평가 (지금 단계) — 개발 셋만

1. `63_b_dev_build.py` — 개발 셋 정상 초안(정상 25 + 개발 43 중 Y)의 OK claim 으로 쉬운 설명 1차 출력을 만든다. 역검증은 돌리지 않는다.
2. 사람이 `b_dev_review_v0.csv` 에 Y/N(+사유)을 적는다. **역검증 결과를 보기 전에.**
3. `64_b_dev_eval.py` — 사람 Y 기반에 교란 10종(수치 4 · 상품 · 기관 · 강조어 · 헤지 · 조건 삭제 · CHECK 유입)을 넣고, rules / full 두 구성으로 False Pass·과차단을 본다.
4. 이 결과로 규칙·프롬프트를 고친다. 개발 셋 수치는 공식 수치로 쓰지 않는다.

### 공식 평가 (다음 단계) — 새 held-out

A 의 held-out 규칙을 그대로 따른다.
- 개발 평가로 규칙을 확정한 뒤, 새 입력으로 쉬운 설명을 만들고 **사람 라벨을 먼저 확정·freeze** 한 다음 역검증을 1회 돌린다.
- freeze 후 결과를 보고 규칙을 고치면, 그 셋은 더 이상 held-out 이 아니다.
- 교란 held-out 도 freeze 시점에 생성 시드와 교란 목록을 해시로 고정한다.
- 1회 수정본은 수정 경로 결과를 모르는 상태에서 사람이 따로 라벨링한다.
- 입력원(새 질문 생성 또는 held-out v1 의 Y 초안 재사용)은 개발 평가 결과를 본 뒤 정하고, 어느 쪽이든 그 선택과 이유를 이 문서에 남긴다. held-out v1 초안을 쓰더라도 B 라벨(쉬운 설명의 뜻 보존)은 새로 만드는 것이다.

## 알려진 한계

- **조건 목록은 A 의 추출 품질에 묶인다.** `Claim.conditions` 가 비면 LLM 의 '목록 외 조건' 보고만 남는다.
- **조건 삭제 교란에는 라벨 잡음이 있다.** 조건 블록 문구를 지워도 같은 조건이 다른 문구에 다른 말로 남을 수 있다. 수치·핵심어가 남으면 교란을 만들지 않지만 완전히 바꿔 말한 경우는 거르지 못한다. 놓친 사례는 보고서에 목록으로 남겨 사람이 확인한다.
- **같은 모델 계열.** 쉬운 설명 생성과 조건 보존 검사가 모두 `gpt-4.1-mini` 다.
- **CHECK 유입의 문구 겹침 검사**는 거의 그대로 옮긴 경우만 잡는다 (2-gram 포함률 0.6 이상, OK 원문보다 0.25 이상 높을 때). 바꿔 말한 유입은 수치·상품 규칙과 출처 id 검사에 기댄다.

## 화면 흐름 (B 평가 후 연결)

```
직원 초안 입력 → 검증(A) → 쉬운 설명(B) → 역검증(B) → 직원 확인
                                  PASS: 고객용 문구 + '꼭 확인하세요' 조건 (직원이 최종 확인 후 사용)
                                  BLOCK: 고객용 문구 숨김, OK claim 원문 대체 안내
                     직원 확인 필요(CHECK)·제외(PROHIBITED)는 항상 직원 화면에만
```

## 재현

```
python scripts/62_test_v2_simplify.py            # LLM 없이 49건
python scripts/63_b_dev_build.py --dry-run
python scripts/63_b_dev_build.py
python scripts/64_b_dev_eval.py --config rules
python scripts/64_b_dev_eval.py --config full --with-fix
```
