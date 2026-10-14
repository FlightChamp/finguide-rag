# 초안 수준 범위 검사 — 개발 셋

```
========================================================================
초안 수준 범위 검사 — 개발 셋 (정렬=llm)
========================================================================
※ LLM 캐시에 없는 항목 5건 — 대체 claim 으로 처리됨 (54·58번 cv-v1 실행 후 다시 돌릴 것)

[오탐] 사람=Y 53건 중 범위 경고 1건
  정상25  d020   MIXED_SCOPE: '고객센터 환전지갑 서비스 설명서' 근거와 그 상품을 언급하지 않는 FAQ 근거가 섞임: ['hana_faq_121']

[개발 표적 4건]
  ✔ q043   MIXED_SCOPE: '해외 ATM 출금 서비스' 근거와 그 상품을 언급하지 않는 FAQ 근거가 섞임: ['hana_faq_189']
  ✔ q055   SCOPE_MISMATCH: 질문에 없는 'easy-one 외화송금서비스' 기준으로 답함 (c1)
  ✔ B_01   SCOPE_MISMATCH: 질문에 없는 '사업자 주거래 우대통장' 기준으로 답함 (c2)
  ✔ B_03   MIXED_SCOPE: 서로 다른 상품·서비스 근거가 한 답변에 섞임: ['기업대출', '보금자리론'] / SCOPE_MISMATCH: 질문에 없는 '기업대출' 기준으로 답함 (c1)

[사람=N 23건 전체]  범위 경고 5건
  최종 판정 OK 아님: 범위 검사 전 12/23 → 후 16/23
  정상(Y) OK 아님:   범위 검사 전 21/53 → 후 21/53
  개발43  q043   MIXED_SCOPE
  개발43  q055   SCOPE_MISMATCH
  자연8   A_d027 MIXED_SCOPE
  자연8   B_01   SCOPE_MISMATCH
  자연8   B_03   MIXED_SCOPE / SCOPE_MISMATCH

[사유별 범위 경고]  (61번 사유 분류 기준)
  MIXED_SCOPE        3/9   q043, B_01, B_03
  SCOPE_MISMATCH     2/3   q055, B_01
  WRONG              0/4   
  UNSUPPORTED        0/5   
  MISSING_CONDITION  0/2   
  OVERSTATED         0/2   
  OMITTED            1/1   A_d027
  OUTDATED           0/1   

※ 개발 셋 수치다. 공식 효과는 새 held-out(60번)으로만 보고한다.
```
