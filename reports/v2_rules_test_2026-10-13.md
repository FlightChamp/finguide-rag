# v2 claim 규칙 테스트 (2026-10-13)

| 그룹 | 케이스 | 결과 | 실제 |
|---|---|---|---|
| E.단정·헤지 | 강조어 추가 '그대로' | 통과 | `fail ["원문에 없는 강조어: ['그대로']"]` |
| E.단정·헤지 | 강조어 추가 '무조건' | 통과 | `fail ["원문에 없는 강조어: ['무조건']"]` |
| E.단정·헤지 | 근거에도 있는 '반드시'는 정상 | 통과 | `pass []` |
| E.단정·헤지 | 헤지 소실 '수 있' | 통과 | `fail ["원문의 한정 표현 소실: ['가능성']"]` |
| E.단정·헤지 | 헤지 소실 '에 한해' | 통과 | `fail ["원문의 한정 표현 소실: ['한정']"]` |
| E.단정·헤지 | 단서('다만')는 규칙이 아닌 LLM(MISSING_CONDITION) 담당 → 규칙 통과 | 통과 | `pass []` |
| E.단정·헤지 | 동등 표현: 근거 '지참하여야' → claim '반드시 지참'은 정상 | 통과 | `pass []` |
| E.단정·헤지 | 표현을 바꾼 한정은 통과 | 통과 | `pass []` |
| E.단정·헤지 | 헤지 없는 근거, 헤지 없는 claim | 통과 | `pass []` |
| E.단정·헤지 | 능력·방법 안내 '할 수 있습니다'는 헤지 아님 | 통과 | `pass []` |
| E.단정·헤지 | '변동될 수 있' 소실은 탐지 | 통과 | `fail ["원문의 한정 표현 소실: ['가능성']"]` |
| E.단정·헤지 | assertiveness_of: strong / hedged / neutral | 통과 | `('strong', 'hedged', 'neutral')` |
| F.기관 | 별칭 정규화: KB국민은행 → 국민은행 (국민 이중 매칭 없음) | 통과 | `['국민은행', '하나은행']` |
| F.기관 | 타 기관 언급 → INSTITUTION_MISMATCH | 통과 | `fail` |
| F.기관 | 같은 기관 언급 → pass | 통과 | `pass` |
| F.기관 | 기관 언급 없음 → n/a | 통과 | `n/a` |
| F.기관 | '우리' 같은 일반어는 기관으로 보지 않음 | 통과 | `set()` |
| G.상품 | 같은 상품 → pass | 통과 | `pass` |
| G.상품 | 유사 상품명 hard negative: 주택청약예금 ↔ 주택청약종합저축 → PRODUCT_MISMATCH | 통과 | `fail ["claim 상품 '주택청약예금' ↔ 근거 설명서 ['주택청약종합저축']"]` |
| G.상품 | v1 실패 사례: 정기예금 ↔ 주택청약예금 설명서 → PRODUCT_MISMATCH | 통과 | `fail` |
| G.상품 | 근거가 약관·FAQ 뿐 → n/a (이름 대조 불가) | 통과 | `n/a` |
| G.상품 | 설명서 여럿 중 하나라도 같으면 pass | 통과 | `pass` |
| G.상품 | 변별력 없는 이름('하나은행 서비스') → n/a | 통과 | `n/a` |
| G.상품 | catalog ambiguous → AMBIGUOUS_PRODUCT | 통과 | `[<ReasonCode.AMBIGUOUS_PRODUCT: 'AMBIGUOUS_PRODUCT'>]` |
| H.고객·시점·공란 | 확정형 '받으실 수 있' | 통과 | `["고객별 확인이 필요한 확정 표현: '고객님은 우대금리를 받으실 수 있'"]` |
| H.고객·시점·공란 | 고객 속성 '신용등급' | 통과 | `["고객별 확인이 필요한 확정 표현: '고객님 신용'"]` |
| H.고객·시점·공란 | 고객 속성 '거래실적' | 통과 | `["고객별 확인이 필요한 확정 표현: '고객님의 거래실적'"]` |
| H.고객·시점·공란 | 일반 규정 존댓말은 통과 | 통과 | `[]` |
| H.고객·시점·공란 | 일반 조건 설명은 통과 | 통과 | `[]` |
| H.고객·시점·공란 | 시점 의존 '현재' → TIME_VARIANT (v1 패턴) | 통과 | `` |
| H.고객·시점·공란 | '다음달까지' 같은 문서상 기간은 시점 의존 아님 (v1 결정 유지) | 통과 | `` |
| H.고객·시점·공란 | '최근 3개월 이내 발급'은 문서상 기간 → 통과 | 통과 | `` |
| H.고객·시점·공란 | 원문에 이미 있는 '현재'를 옮긴 것 → 통과 | 통과 | `` |
| H.고객·시점·공란 | 근거 공란 + 금리 확정 → BLANK_VALUE | 통과 | `["근거는 공란인데 수치 확정: ['3%']"]` |
| H.고객·시점·공란 | 근거 공란이어도 수치 없는 claim → 통과 | 통과 | `` |
| H.고객·시점·공란 | 구버전 근거 → OUTDATED_SOURCE | 통과 | `` |
| I.묶음+정책 | 기획 예시: '중도해지해도 우대금리는 그대로' + LLM SUPPORTED → CHECK 이상, 충돌 기록 | 통과 | `CHECK conflict=True {'numeric': 'n/a', 'assertiveness': 'fail', 'institution': 'n/a', 'product': 'n/a', 'customer_data': 'pass', 'time_variant': 'pass', 'blank_value': 'pass', 'outdated': 'pass'}` |
| I.묶음+정책 | 상품 불일치 + 수치 일치 → PROHIBITED | 통과 | `PROHIBITED {'numeric': 'pass', 'assertiveness': 'pass', 'institution': 'n/a', 'product': 'fail', 'customer_data': 'pass', 'time_variant': 'pass', 'blank_value': 'pass', 'outdated': 'pass'}` |
| I.묶음+정책 | 깨끗한 claim → 규칙 hit 없음, OK | 통과 | `OK {'numeric': 'n/a', 'assertiveness': 'pass', 'institution': 'n/a', 'product': 'pass', 'customer_data': 'pass', 'time_variant': 'pass', 'blank_value': 'pass', 'outdated': 'pass'}` |
| I.묶음+정책 | rule_checks 에 8개 검사 키가 모두 있음 | 통과 | `['assertiveness', 'blank_value', 'customer_data', 'institution', 'numeric', 'outdated', 'product', 'time_variant']` |

## 정상 답변 사전 점검 (dev)

- 문장 114개
- 근거=청크 전체: 경고 27개
- 근거=유사문장(LLM 정렬 대용): 경고 9개, {'NO_EVIDENCE': 3, 'OVERSTATED': 3, 'NUMERIC_MISMATCH': 2, 'TIME_VARIANT': 1}