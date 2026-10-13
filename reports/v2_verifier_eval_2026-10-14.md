# v2 ClaimVerifier 평가

```
============================================================================================
ClaimVerifier 평가 — 오류 주입 v0  (model=gpt-4.1-mini, prompt=5ebe0746f3a6)
============================================================================================

[탐지율]  유형                                           rules                 llm            combined   미정렬
  INSTITUTION/insert                      0.88 [0.66–0.97]    0.41 [0.22–0.64]    0.88 [0.66–0.97]   2/17
  INSTITUTION/replace                     1.00 [0.68–1.00]    0.75 [0.41–0.93]    1.00 [0.68–1.00]   0/8
  NUMERIC/comparator                      1.00 [0.51–1.00]    0.50 [0.15–0.85]    1.00 [0.51–1.00]   0/4
  NUMERIC/unit                            1.00 [0.44–1.00]    0.67 [0.21–0.94]    1.00 [0.44–1.00]   0/3
  NUMERIC/value                           0.91 [0.62–0.98]    0.82 [0.52–0.95]    0.91 [0.62–0.98]   1/11
  OVERSTATED/hedge_drop                   1.00 [0.57–1.00]    0.60 [0.23–0.88]    1.00 [0.57–1.00]   0/5
  OVERSTATED/intensifier(목록 밖)            0.18 [0.05–0.48]    0.36 [0.15–0.65]    0.45 [0.21–0.72]   1/11
  OVERSTATED/intensifier(목록 안)            0.92 [0.67–0.99]    0.23 [0.08–0.50]    0.92 [0.67–0.99]   1/13
  PRODUCT/easy_negative                   1.00 [0.44–1.00]    0.00 [0.00–0.56]    1.00 [0.44–1.00]   0/3
  (전체)                                    0.83 [0.73–0.90]    0.48 [0.37–0.59]    0.87 [0.77–0.93]

[심각도 정확]  rules 0.81 [0.71–0.89]  llm 0.25 [0.17–0.36]  combined 0.85 [0.76–0.92]
[코드 정확]    rules 0.75 [0.64–0.83]  llm 0.00 [0.00–0.05]  combined 0.75 [0.64–0.83]

[대조군 — 정상 초안]
  rules     claim 70개  soft(OK 아님) 0.53 [0.41–0.64]   hard(PROHIBITED) 0.10 [0.05–0.19]
  llm       claim 70개  soft(OK 아님) 0.14 [0.08–0.24]   hard(PROHIBITED) 0.00 [0.00–0.05]
  combined  claim 70개  soft(OK 아님) 0.57 [0.45–0.68]   hard(PROHIBITED) 0.10 [0.05–0.19]
  초안 최종 PROHIBITED  0.08 [0.02–0.25] (n=25)

LLM–Rule 충돌 0.38 [0.32–0.43]   VERIFIER_ERROR 0.00 [0.00–0.01]   (claim 295개)

[자연 오류 셋] 초안 단위 — claim 하나라도 OK 가 아니면 탐지
  id      gold                                     rules         llm    combined   발급 코드(combined)
  A_d010  MISSING_CONDITION|CONTRADICTED           CHECK       CHECK       CHECK   MISSING_CONDITION,NEEDS_CUSTOMER_DATA
  A_d013  MISSING_CONDITION|CONTRADICTED|N         CHECK          OK       CHECK   NEEDS_CUSTOMER_DATA
  A_d027  OMITTED_REQUIRED_INFO|NO_EVIDENC        PROHIB       CHECK      PROHIB   MISSING_CONDITION,NEEDS_CUSTOMER_DATA,NO_EVIDENCE,OVERSTATED
  A_d029  MISSING_CONDITION                        CHECK          OK       CHECK   NEEDS_CUSTOMER_DATA
  A_d030  CONTRADICTED|MISSING_CONDITION              OK       CHECK       CHECK   MISSING_CONDITION
  B_01    PRODUCT_MISMATCH|CONTRADICTED               OK          OK          OK   
  B_02    PRODUCT_MISMATCH|CONTRADICTED            CHECK          OK       CHECK   NEEDS_CUSTOMER_DATA,OVERSTATED
  B_03    PRODUCT_MISMATCH|CONTRADICTED               OK          OK          OK   
  탐지율  rules 0.62 [0.31–0.86]  llm 0.38 [0.14–0.69]  combined 0.75 [0.41–0.93]  (n=8)
호출 API 216 / 캐시 0, 토큰 311,593
```
