# v2 ClaimVerifier 평가

```
============================================================================================
ClaimVerifier 평가 — 오류 주입 v0  (model=gpt-4.1-mini, prompt=5ebe0746f3a6)
============================================================================================

[탐지율]  유형                                           rules                 llm            combined   미정렬
  INSTITUTION/insert                      0.88 [0.66–0.97]    0.41 [0.22–0.64]    0.88 [0.66–0.97]   2/17
  INSTITUTION/replace                     1.00 [0.68–1.00]    0.75 [0.41–0.93]    1.00 [0.68–1.00]   0/8
  NUMERIC/comparator                      0.75 [0.30–0.95]    0.50 [0.15–0.85]    0.75 [0.30–0.95]   0/4
  NUMERIC/unit                            1.00 [0.44–1.00]    0.67 [0.21–0.94]    1.00 [0.44–1.00]   0/3
  NUMERIC/value                           0.91 [0.62–0.98]    0.82 [0.52–0.95]    0.91 [0.62–0.98]   1/11
  OVERSTATED/hedge_drop                   0.40 [0.12–0.77]    0.60 [0.23–0.88]    0.80 [0.38–0.96]   0/5
  OVERSTATED/intensifier(목록 밖)            0.00 [0.00–0.26]    0.36 [0.15–0.65]    0.36 [0.15–0.65]   1/11
  OVERSTATED/intensifier(목록 안)            0.92 [0.67–0.99]    0.23 [0.08–0.50]    0.92 [0.67–0.99]   1/13
  PRODUCT/easy_negative                   1.00 [0.44–1.00]    0.00 [0.00–0.56]    1.00 [0.44–1.00]   0/3
  (전체)                                    0.75 [0.64–0.83]    0.48 [0.37–0.59]    0.83 [0.73–0.90]

[심각도 정확]  rules 0.75 [0.64–0.83]  llm 0.25 [0.17–0.36]  combined 0.83 [0.73–0.90]
[코드 정확]    rules 0.75 [0.64–0.83]  llm 0.00 [0.00–0.05]  combined 0.75 [0.64–0.83]
[코드≈ 동등]   rules 0.75 [0.64–0.83]  llm 0.25 [0.17–0.36]  combined 0.83 [0.73–0.90]   (CONTRADICTED≈수치·상품·기관, MISSING_CONDITION≈과장)
※ '탐지'는 아무 코드로나 걸린 것. 엉뚱한 코드로 걸린 우연 탐지가 섞이므로 '코드 정확'을 함께 볼 것
   설정: promote_customer_slot=False, product_guard=True

[대조군 — 정상 초안]
  rules     claim 70개  soft(OK 아님) 0.06 [0.02–0.14]   hard(PROHIBITED) 0.00 [0.00–0.05]
  llm       claim 70개  soft(OK 아님) 0.14 [0.08–0.24]   hard(PROHIBITED) 0.00 [0.00–0.05]
  combined  claim 70개  soft(OK 아님) 0.19 [0.11–0.29]   hard(PROHIBITED) 0.00 [0.00–0.05]
  초안 최종 PROHIBITED  0.00 [0.00–0.13] (n=25)

LLM–Rule 충돌 0.12 [0.09–0.16]   VERIFIER_ERROR 0.00 [0.00–0.01]   (claim 295개)

[자연 오류 셋] 초안 단위 — claim 하나라도 OK 가 아니면 탐지, '적중'은 기대 코드가 나온 것
  id      gold                                     rules         llm    combined   발급 코드(combined)
  ✔A_d010 MISSING_CONDITION|CONTRADICTED              OK       CHECK       CHECK   MISSING_CONDITION
   A_d013 MISSING_CONDITION|CONTRADICTED|N            OK          OK          OK   
  ✔A_d027 OMITTED_REQUIRED_INFO|NO_EVIDENC         CHECK       CHECK       CHECK   MISSING_CONDITION,NO_EVIDENCE,OVERSTATED
   A_d029 MISSING_CONDITION                           OK          OK          OK   
  ✔A_d030 CONTRADICTED|MISSING_CONDITION              OK       CHECK       CHECK   MISSING_CONDITION
   B_01   PRODUCT_MISMATCH|CONTRADICTED               OK          OK          OK   
   B_02   PRODUCT_MISMATCH|CONTRADICTED            CHECK          OK       CHECK   OVERSTATED
   B_03   PRODUCT_MISMATCH|CONTRADICTED               OK          OK          OK   
  탐지율  rules 0.25 [0.07–0.59]  llm 0.38 [0.14–0.69]  combined 0.50 [0.22–0.78]  (n=8)
  기대 코드 적중(combined)  0.38 [0.14–0.69]
호출 API 0 / 캐시 216, 토큰 0
```
