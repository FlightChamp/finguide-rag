# RAGAS 사전 탐지 — 2026-08-23 11:41

- 루트: `C:\Programming\MyProject\financial_rag_project`


## 1. 실행 환경

- Python: 3.14.6
- 실행 파일: C:\Programming\MyProject\financial_rag_project\.venv\Scripts\python.exe
- ragas: (미설치)
- langchain: 1.3.14
- langchain-core: 1.5.3
- langchain-openai: (미설치)
- datasets: (미설치)
- openai: 2.52.0
- pandas: 3.0.5
- numpy: 2.5.1
- sentence-transformers: 5.6.1
- faiss-cpu: 1.14.3
- pydantic: 2.13.4

## 2. 평가 산출물 후보

총 56개 발견, 최신 40개 분석


### `reports/g_eval_full_2026-08-23.csv`

- 크기 19.7KB / CSV, 총 96행 / 수정 2026-08-23 03:44
- **RAGAS 적합도: 부분** — 누락: answer
- 역할 추정:
  - question: `﻿question`
  - refusal: `false_refusal`
  - contexts: `n_evidence`, `cited_docs`
  - verdict: `환각`, `수치정확`
- 키 17개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿question` | 14살 미만 애들 현금IC카드 본인이 직접 받을 수 있어? |
| `difficulty` | medium |
| `doc_type` | FAQ |
| `decision` | answer |
| `stage` | llm |
| `false_refusal` | False |
| `n_evidence` | 3 |
| `not_found` | 0 |
| `tokens` | 2296 |
| `latency_ms` | 14986 |
| `환각` | pass |
| `상품일치` | pass |
| `수치정확` | pass |
| `unsupported` | 0 |
| `n_sentences` | 2 |
| `cited_docs` | 1 |
| `why_상품일치` | 인용된 근거가 모두 질문 상품에 속함 |

### `data/interim/query_analysis_cache.json`

- 크기 43.0KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-23 03:44
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 152개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `46c7b489c4a7280f9dbfa46b` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `4fdac56e422ef9710833c12e` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `774451091ebd0bed5e0beb01` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `253e9c6a50776601918a9035` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `290674bc9f5ffe6933c15a40` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `fab9de3284f344020fd6cc32` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `86a4e241415bc4bef3169f1e` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `94b3e77ad612d7ca0d5dc7ee` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `8cfb43aaa0d2645ef00f113e` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `805e36002bbea4cda629a85a` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `3c1a95c1975eac4ac20fe231` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `9005bb0add0145815b243c40` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `0fe7860ac695af9b72e86be7` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `20ee53861c495b0beb036562` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `182b4264ea896f2ed039135d` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `bd00a068d5c7f0be19a3f870` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `95b25d19a5579d490e0831bc` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `e2438aba03d636e10db1b350` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `c1d699d7da16562185fb85fb` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `b2ac219ec67191b1e23f014a` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `070872aa974041054a2a8b31` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `bd6f46182c8df82c53d81a05` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `fb20cde2ffe9160876ddc785` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `2a6ec8436842a8742039d3a2` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `b2bc12ee82b964b7db6d2166` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `5ee633dce964d7b43772e6f3` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `0732e5907b93692635bc3e0c` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `c664f46c674e3278efe79f40` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `744cb5ed5b4100c5766fd4a2` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `0763005c4660ea19472e3e1b` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `e45893be25923b99dcf7427a` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `21ca41ba50d45335d7f4fe5c` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `fedffddb9a05f7b5acee6865` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `3866c9263364b55c599775a4` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `64eb177435dc6f21a09725f9` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `8faf6fe28948353ca790f8d2` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `dfc349bbaccba3738c3c28e9` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `c17857948b767fe054d19e53` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `0d438fee7e33d37fbd3907fe` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `c879191eed3c914dcaad9ddd` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `4614e06c483c0c9ba1b49da6` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `721220f7c4c6fd396e7fd8df` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `fcbc94c7525e91778076c65e` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `df7f01e254fce58aec13330c` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `5ee941e404361637a6c52120` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `cf13bf25412d6c56a328779f` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `0218b174e13c2fdf68b18fdd` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `31c55688a8faa9cf88a1cad8` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `66c0bc92d6a90955052306e0` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `5802a62b91079a64fd1d8ce2` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `b1c4bc63b88e685a47c23ba3` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `42ae9260bf8864970225abd4` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `8c3ea547f15294edd1b66b60` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `17767e6f4b6bb76f021f6585` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `eb10ba1fa551a1bbd7072152` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `cb0c90d08e47c82cf16b2660` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `483b5b79b28d19193bc5a62f` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `6c04dbee652ea7b193a8c69b` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `3dfc9343bbb34c17f320f61e` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `932737f23191887e51f02358` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `bd1c6d4f764065bbd81d83c3` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `59c828fd5ec78f707dfbe648` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `6f52d92fc2b294939376eb1b` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `a194b3dadbbe3b58e3c0b333` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `f1f25958475d2be2dc7319e9` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `0ff420b46b8b1c84b9410dee` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `b4446108b1dd8e6df617c193` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `cd2dc45d960f6f180d8bd600` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `0bfa66663ffc5e0487d1bd65` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `5b65bec8f268139fc39616fb` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `a25cd41339bad6294577673b` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `577890666d558950c6b4c2f3` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `e6a32037a257b8665cd32075` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `1cc4e091b2921e6ad8f5d4aa` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `d8b7e0ff87eae5f1f05c9e7d` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `9678b222bf82d6be034246d1` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `dacf8e9159b1571f53cbded5` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `3560c2603de622968017c02d` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `a5888e4da6df6bb903b251c9` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `87fe847a1e87c8cc8172f128` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `0e45a49703b20c72216a8e8a` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `e5b3f793cb31af3e310fbe30` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `52d35756ed0cbe8dc27514f1` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `deb080e8d8bc16576002d4f6` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `ad011cb1a4d504bf7f8d5100` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `c094962d7a8f16ec6d9f5afb` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `5f111ac06c474f2fb2095050` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `8056c185140767deb89834ef` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `24f925221e78cc5401138f72` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `fb21ff8004d938449e0d9f02` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `76aa9cacfdef67d2c848ac4e` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `15337433276b39b3621fb028` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `f213daccfabbf7031a629b91` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `a81ccf179ef2f345e3ca0e43` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `26fd92e110a0bb738432717e` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `62859de5bf1d1c5ee9f2e421` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `67cd39d7e4d4368d4456ee44` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `20067c6f0df06744e7204ddb` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `256321f1bd594134e5f86ab4` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `9dd357580c12e0c7606cb5fe` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `3ab13816f8f9fd7ba37d74a0` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `0bd4a26b81511eab51450923` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `09c1fc692ed58dfde5d596d7` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `7a89506f56afff563c23c595` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `aa2730b292d5e4327a2c9687` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `bf918d2901482b0b26504e93` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `bd8544910836844c2ccf7dcc` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `454255215d0a7beb71221e13` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `ffbaf2708b3227f7393ca98c` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `f03b9ba4d36de8c7febcf9c6` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `f703c0056d2f3861b2014d69` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `f25886206e0574d31729aae3` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `23ac82f9b09b448b4fefe11a` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `6c0445d6bcf1766a9e3473a5` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `35bb101dbd610e4b4d43c0c3` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `3f80b99449f77919e35366ff` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `47157b73bedcf9c27893a9e9` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `daff88f9db3fcb122e3032a4` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `92e36074240f0394b0c3cd0a` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `6dcdfb022c6c5b108e591062` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `00983b0a884ba3964ee7a9e4` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `e4498ba2ad62081b1cdad664` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `8d20cce6da8c67490bf0361b` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `b08ffb27e6bbd9bbf06faa85` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `67e9d46ff741836a19d8e5a3` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `ec92a7a93470d367acc98a7b` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `953fe164d296417a5d5ffbbc` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `6b3aa43177b58a92b38d0f4d` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `ec67e53410891d3998f54a0b` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `1e182d25e513b1ce35ac114d` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `ba72b994456441283ab45ed7` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `33129f2a2eca1b4aefd35e9f` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `c0ad4e95657b97836bdd270d` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `3b88c647c8f1133e71d00314` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `3366d60af01e9f84491f5f1d` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `80a70871d05cdad2058c1ce4` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `c6a8d43c6936931be75e3a40` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `59d37d9715798ecf578dcb51` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `20c6dd57efa7c666d76a0793` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `149acd1f9c229fb35a7a7c16` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `d43fb98d06f0ba4043b5715a` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `665072a3d90806349bda5e79` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `f1eda39a6863a1811dd2e0f1` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `add6395485120b2290339679` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `3722c9a0b990c6e2c2b10d03` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `bbd6d9c498a108df9d1296d9` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `91ab9968ee0bd4f9e934f079` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `29da51864a393f6422109e1f` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `4c14a22e2a4cd7cdeaa15571` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `c51c513ba982ee9dac7b569a` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `03d23bc026f46f91d128324a` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |
| `c73289b278509421a140b852` | dict(keys=['extracted_product', 'product_granularity', 'intent', 'requires_product_specific_doc', 'can_answer_with_general_terms', 'confidence']) |

### `data/interim/g_judge_v2_cache.json`

- 크기 81.3KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-23 03:44
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 551개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `1292c5292010679ee3b03eab` | dict(keys=['supported', 'evidence', 'why']) |
| `f1be053db39265e079e959db` | dict(keys=['verdict', 'why']) |
| `55a0b67e3926d8f570e4e243` | dict(keys=['verdict', 'why']) |
| `b6a063515e21f226bbec5158` | dict(keys=['verdict', 'why']) |
| `6966a0fbd5a8f5bebb1cd427` | dict(keys=['supported', 'evidence', 'why']) |
| `19a2ddf616eb545b60b0b507` | dict(keys=['supported', 'evidence', 'why']) |
| `65396b705c9b8cef88c70954` | dict(keys=['verdict', 'why']) |
| `43ed32946daf097a597649d3` | dict(keys=['verdict', 'why']) |
| `06e41de5a290f4884841820b` | dict(keys=['verdict', 'why']) |
| `a77dbecdfaf757bd7dbd865b` | dict(keys=['supported', 'evidence', 'why']) |
| `ca65a21ae4c8cf255597c944` | dict(keys=['supported', 'evidence', 'why']) |
| `4592821cc9409cb6102eff2f` | dict(keys=['supported', 'evidence', 'why']) |
| `9ed25607107268ae24ad1904` | dict(keys=['verdict', 'why']) |
| `00fc1d2cc8d42d6d873cd550` | dict(keys=['verdict', 'why']) |
| `4b8cae8d1799c5c127bd697b` | dict(keys=['verdict', 'why']) |
| `7c4604ca67b0ebb36a93af5b` | dict(keys=['supported', 'evidence', 'why']) |
| `31e8e55072e877202c9ddad3` | dict(keys=['supported', 'evidence', 'why']) |
| `7fc7f7a47792e2abee0181d4` | dict(keys=['supported', 'evidence', 'why']) |
| `973315eb08a6acf066c7593f` | dict(keys=['verdict', 'why']) |
| `bd155fd53485e0076e282f2a` | dict(keys=['verdict', 'why']) |
| `3315d9b31a246529d9dd7ff4` | dict(keys=['verdict', 'why']) |
| `ed360bf1001141c5c4585c82` | dict(keys=['supported', 'evidence', 'why']) |
| `b522f0da0489bb76a487bb8a` | dict(keys=['supported', 'evidence', 'why']) |
| `e3c08fd27d7cec357d8a4aba` | dict(keys=['supported', 'evidence', 'why']) |
| `a4718e83a57f1cbc89e345b6` | dict(keys=['verdict', 'why']) |
| `d8ebf6f10ba7016ec29b5f8d` | dict(keys=['verdict', 'why']) |
| `4a6ddf347a69f98f3b8cfd61` | dict(keys=['verdict', 'why']) |
| `9d4104dc81b0f3d140bc7f9c` | dict(keys=['supported', 'evidence', 'why']) |
| `ffa1d3825bc28b66e347bb3f` | dict(keys=['supported', 'evidence', 'why']) |
| `b4564289cd0a18b91151a46f` | dict(keys=['verdict', 'why']) |
| `7d31ff053ec236459cfefde6` | dict(keys=['verdict', 'why']) |
| `96af102172513ef09733dd04` | dict(keys=['verdict', 'why']) |
| `e50bee2605b364ca1891700e` | dict(keys=['supported', 'evidence', 'why']) |
| `e4a9a5695125418290b7c018` | dict(keys=['supported', 'evidence', 'why']) |
| `e4498968c18142ea16669589` | dict(keys=['supported', 'evidence', 'why']) |
| `6743c28c167804a3cc2c9efe` | dict(keys=['verdict', 'why']) |
| `c1f9944fa61fe7708c5fb7ac` | dict(keys=['verdict', 'why']) |
| `7ce0d7e0827188bd279cdc0c` | dict(keys=['verdict', 'why']) |
| `2640e06679f7c9775e617824` | dict(keys=['supported', 'evidence', 'why']) |
| `a89f35b4f0eee95ed93f5647` | dict(keys=['supported', 'evidence', 'why']) |
| `fb607e5f1752b315876f6c23` | dict(keys=['verdict', 'why']) |
| `e294804679cb7dc9b45018b7` | dict(keys=['verdict', 'why']) |
| `d63b7714092dbdac7e86b401` | dict(keys=['verdict', 'why']) |
| `418259d5c4a90dbee9ff9f94` | dict(keys=['supported', 'evidence', 'why']) |
| `12b995d0eddd50d965e8f6f4` | dict(keys=['supported', 'evidence', 'why']) |
| `f13cfc6e9106a854899a0b80` | dict(keys=['supported', 'evidence', 'why']) |
| `f35c72fad5d19bd6e5aa2acc` | dict(keys=['verdict', 'why']) |
| `a49f2aa8e7980e245ffd75c6` | dict(keys=['verdict', 'why']) |
| `6fe24049c4e12c1ed14f5f43` | dict(keys=['verdict', 'why']) |
| `04261beca48ff10b16e4482b` | dict(keys=['supported', 'evidence', 'why']) |
| `070e1d873ec455a3967ea8e1` | dict(keys=['supported', 'evidence', 'why']) |
| `d88d62699276256a31c69b05` | dict(keys=['supported', 'evidence', 'why']) |
| `9a472a1ff12968e6259cb524` | dict(keys=['verdict', 'why']) |
| `582f24d38cadb8bd482ae37a` | dict(keys=['verdict', 'why']) |
| `9c8c9a6a00e8953cd8ef8435` | dict(keys=['verdict', 'why']) |
| `6c652595986d731a23078e64` | dict(keys=['supported', 'evidence', 'why']) |
| `950bfa0a6d4bbb8306a66921` | dict(keys=['verdict', 'why']) |
| `487c5abb26d1d7b105448945` | dict(keys=['verdict', 'why']) |
| `e450a9fe549c121c275b1a4e` | dict(keys=['verdict', 'why']) |
| `11ebd84864404d9df6cc3def` | dict(keys=['supported', 'evidence', 'why']) |
| `e3d44b4fb98a432547d573dd` | dict(keys=['supported', 'evidence', 'why']) |
| `296d467041c0ccbf8858e6f1` | dict(keys=['supported', 'evidence', 'why']) |
| `8eab10059599fcfe5684691c` | dict(keys=['verdict', 'why']) |
| `6a31bf5b9a9f2644853a51be` | dict(keys=['verdict', 'why']) |
| `dfbf5f3757a9cb639f3d15e1` | dict(keys=['verdict', 'why']) |
| `0d78400cf093b0213bfe3c7b` | dict(keys=['supported', 'evidence', 'why']) |
| `44b969bec27b41e0d270a076` | dict(keys=['supported', 'evidence', 'why']) |
| `a8ccc17dc04cee9200ba246a` | dict(keys=['supported', 'evidence', 'why']) |
| `5327b97d9010c2653230bca4` | dict(keys=['supported', 'evidence', 'why']) |
| `70234796f30a65f15fc84acc` | dict(keys=['verdict', 'why']) |
| `2774030bb8375d79b7dd10bb` | dict(keys=['verdict', 'why']) |
| `c0b4a0eb7bcf51b307385cfa` | dict(keys=['verdict', 'why']) |
| `ad8f605be239f2f8daacec45` | dict(keys=['supported', 'evidence', 'why']) |
| `50e0b084865bd6871f79460b` | dict(keys=['verdict', 'why']) |
| `f1c4f28094944b8ede5c2e35` | dict(keys=['verdict', 'why']) |
| `c852398e17d142ca237b2c6c` | dict(keys=['verdict', 'why']) |
| `49a3472b323ecda4fb2d7a04` | dict(keys=['supported', 'evidence', 'why']) |
| `63eec5245a14acfb11fff1f6` | dict(keys=['supported', 'evidence', 'why']) |
| `f187b9fb55be0f459018fbfa` | dict(keys=['supported', 'evidence', 'why']) |
| `a3f4f1d2760ed5bdc26bd7b1` | dict(keys=['verdict', 'why']) |
| `bde97e21739027341ef1bb73` | dict(keys=['verdict', 'why']) |
| `15cd6f8b076a78e3f5053187` | dict(keys=['verdict', 'why']) |
| `e944e1add7cdf88476518907` | dict(keys=['supported', 'evidence', 'why']) |
| `f3ded7d6d8297d4f1857357f` | dict(keys=['supported', 'evidence', 'why']) |
| `2f4e4c08275890fbdc74a9b1` | dict(keys=['supported', 'evidence', 'why']) |
| `53a4acea044750ca306a44fc` | dict(keys=['supported', 'evidence', 'why']) |
| `b4629b7ad65a44b452498943` | dict(keys=['supported', 'evidence', 'why']) |
| `ee83c884f9263a49764d51ce` | dict(keys=['supported', 'evidence', 'why']) |
| `29a09f09d708f69edc6bfb21` | dict(keys=['supported', 'evidence', 'why']) |
| `03d4a0667e3ac0b959974984` | dict(keys=['supported', 'evidence', 'why']) |
| `d1022d7c22db57f1f19db3b5` | dict(keys=['supported', 'evidence', 'why']) |
| `251d0bd788c608cc836f111b` | dict(keys=['supported', 'evidence', 'why']) |
| `dd476e227016e9326ab45d1e` | dict(keys=['supported', 'evidence', 'why']) |
| `75ec633e087239bd834fc5ef` | dict(keys=['supported', 'evidence', 'why']) |
| `f72b4e8ae4067feb3788171d` | dict(keys=['supported', 'evidence', 'why']) |
| `92b5f4875a56336baa755021` | dict(keys=['supported', 'evidence', 'why']) |
| `4ac67ea635ac30b877627bfb` | dict(keys=['supported', 'evidence', 'why']) |
| `41e8c7e838b72cb247daf126` | dict(keys=['supported', 'evidence', 'why']) |
| `903e1a088954f1a1f21c2f5d` | dict(keys=['supported', 'evidence', 'why']) |
| `5cfe15fc2c0ab837c41bf8d7` | dict(keys=['supported', 'evidence', 'why']) |
| `8d72518d600e7ce203c5ebcc` | dict(keys=['supported', 'evidence', 'why']) |
| `9675ea1190a27ee87e5fa0d5` | dict(keys=['supported', 'evidence', 'why']) |
| `95c23d7d8fd6e15887384ae7` | dict(keys=['supported', 'evidence', 'why']) |
| `12ef70b6f97f0fb749f85e80` | dict(keys=['supported', 'evidence', 'why']) |
| `c9a49b637bb040820d8a332e` | dict(keys=['supported', 'evidence', 'why']) |
| `7923b9f1c887d00d4c12b45f` | dict(keys=['supported', 'evidence', 'why']) |
| `f80a63d0cbb96b666a6ae986` | dict(keys=['supported', 'evidence', 'why']) |
| `478ce296506d2d5a9639b864` | dict(keys=['supported', 'evidence', 'why']) |
| `b5cfe4c0524a27c22209e6a9` | dict(keys=['supported', 'evidence', 'why']) |
| `2a82f3b9b8f6f6ebf7f4e687` | dict(keys=['supported', 'evidence', 'why']) |
| `b7b13dd9ed7a24c6017f7f0f` | dict(keys=['supported', 'evidence', 'why']) |
| `fa1efca6e31bb0bddb595edb` | dict(keys=['supported', 'evidence', 'why']) |
| `e40276ea89d54f5c6d6506fa` | dict(keys=['supported', 'evidence', 'why']) |
| `03728dbc25f0cbb337019686` | dict(keys=['supported', 'evidence', 'why']) |
| `37ac00fa54205e9fa90ec0b6` | dict(keys=['supported', 'evidence', 'why']) |
| `dc55d57f06dd15122ebf7415` | dict(keys=['supported', 'evidence', 'why']) |
| `141d6fa9b0bef7dd3678d562` | dict(keys=['supported', 'evidence', 'why']) |
| `d449131e75d3c58ac38bf8d3` | dict(keys=['supported', 'evidence', 'why']) |
| `519112594e2acff899bcde73` | dict(keys=['supported', 'evidence', 'why']) |
| `7e67653099002c20eadc9ce9` | dict(keys=['supported', 'evidence', 'why']) |
| `c29011e67159e48fcddd12f3` | dict(keys=['supported', 'evidence', 'why']) |
| `49bae67ee37b313a40b7e3af` | dict(keys=['verdict', 'why']) |
| `29c820d665f646b939993149` | dict(keys=['supported', 'evidence', 'why']) |
| `468d9282c1518ff94cdf2b00` | dict(keys=['supported', 'evidence', 'why']) |
| `7d589e16ba7bcb624860a270` | dict(keys=['verdict', 'why']) |
| `5ac70adf6e6d3cd981549cf0` | dict(keys=['supported', 'evidence', 'why']) |
| `e1a2fe2fcd976d66ea0441f6` | dict(keys=['supported', 'evidence', 'why']) |
| `e2076036a962b1c86efd0a2f` | dict(keys=['supported', 'evidence', 'why']) |
| `4946753f065d035516dc8334` | dict(keys=['verdict', 'why']) |
| `c3ce73fd4181af73c573f7f6` | dict(keys=['supported', 'evidence', 'why']) |
| `5120866c6d88180a05216e2d` | dict(keys=['verdict', 'why']) |
| `669f0958f4f68a3ae02a4bb8` | dict(keys=['supported', 'evidence', 'why']) |
| `eb999c51371175bf271a54b7` | dict(keys=['supported', 'evidence', 'why']) |
| `7300b7c68e4d3f1ace584e96` | dict(keys=['supported', 'evidence', 'why']) |
| `a1734e9da265687225ebe67d` | dict(keys=['verdict', 'why']) |
| `6a45c0daec8e87730dab5467` | dict(keys=['supported', 'evidence', 'why']) |
| `6a7028c5191b41833daa461c` | dict(keys=['supported', 'evidence', 'why']) |
| `553776cc2e12263215e1c50e` | dict(keys=['verdict', 'why']) |
| `c9adfe45a688091e6ecf018d` | dict(keys=['supported', 'evidence', 'why']) |
| `3c30acfe38277418914cef77` | dict(keys=['supported', 'evidence', 'why']) |
| `c6f36213d38f07193fc5f12d` | dict(keys=['verdict', 'why']) |
| `8436d0304ccf9b84688d7f05` | dict(keys=['supported', 'evidence', 'why']) |
| `73654d71d61c3b5435bd87e5` | dict(keys=['supported', 'evidence', 'why']) |
| `70aca9d3632159c58a1242e3` | dict(keys=['supported', 'evidence', 'why']) |
| `a153d9cf3a6d1dcbd7b7fdb2` | dict(keys=['verdict', 'why']) |
| `a0c2b7433a9a1a162e9d4882` | dict(keys=['supported', 'evidence', 'why']) |
| `c4f47ae09eb6277d2140a1fc` | dict(keys=['supported', 'evidence', 'why']) |
| `1ab56ff3746b7cdc074b2470` | dict(keys=['verdict', 'why']) |
| `2077c8e66e53f4ad83d0380c` | dict(keys=['supported', 'evidence', 'why']) |
| `6f8e4c8f21a5bbfc70ba657f` | dict(keys=['supported', 'evidence', 'why']) |
| `530bed42842eee8bb72ba4d2` | dict(keys=['supported', 'evidence', 'why']) |
| `b5c65024ccdb57ecafef020d` | dict(keys=['verdict', 'why']) |
| `de6baf5600e236eb49d57790` | dict(keys=['supported', 'evidence', 'why']) |
| `c50e3a38bb795ebad7ca185d` | dict(keys=['supported', 'evidence', 'why']) |
| `1fb267c5a31f94eee422bc30` | dict(keys=['supported', 'evidence', 'why']) |
| `50c15e0f8728cd4703589680` | dict(keys=['verdict', 'why']) |
| `eedf56f7a7c39a289cdadd5e` | dict(keys=['supported', 'evidence', 'why']) |
| `bd33c50e02564d80597b1dab` | dict(keys=['supported', 'evidence', 'why']) |
| `737c186d23e0142a79774e0b` | dict(keys=['supported', 'evidence', 'why']) |
| `acd8e105e86c622b1c1a74db` | dict(keys=['verdict', 'why']) |
| `f0434dc8bb6112939a02f4d1` | dict(keys=['supported', 'evidence', 'why']) |
| `559239a249a3d429a3318915` | dict(keys=['supported', 'evidence', 'why']) |
| `62edf26b174698b566634b40` | dict(keys=['verdict', 'why']) |
| `27d31933964ac0d81342d18e` | dict(keys=['supported', 'evidence', 'why']) |
| `91e8448aa2757fe5e5a62633` | dict(keys=['supported', 'evidence', 'why']) |
| `c544bfcb3b1d9ac7a7a845ef` | dict(keys=['supported', 'evidence', 'why']) |
| `4446b3a1cca3a5f02a2b686a` | dict(keys=['verdict', 'why']) |
| `64611a7b843816a25c8d180c` | dict(keys=['supported', 'evidence', 'why']) |
| `68a4753912c0cffab1e29d33` | dict(keys=['supported', 'evidence', 'why']) |
| `a2fd21e2deaf9c44969cbb69` | dict(keys=['supported', 'evidence', 'why']) |
| `c660e2e3cdd682cda8616782` | dict(keys=['verdict', 'why']) |
| `da4a381a8d8c485ba8fd728d` | dict(keys=['supported', 'evidence', 'why']) |
| `5298a05e82bfed8ea23ea6c9` | dict(keys=['verdict', 'why']) |
| `115e72b8602bf9e9ff70cc55` | dict(keys=['supported', 'evidence', 'why']) |
| `0c92307d53c2684a41884ad8` | dict(keys=['supported', 'evidence', 'why']) |
| `2f75b62505ca4ccac63bf047` | dict(keys=['supported', 'evidence', 'why']) |
| `09cfe48aa2ded98272365a58` | dict(keys=['verdict', 'why']) |
| `6f78caae98138515aba894e3` | dict(keys=['supported', 'evidence', 'why']) |
| `1709da56b62c088f7c2b593a` | dict(keys=['supported', 'evidence', 'why']) |
| `c9f335f639fde53d26513964` | dict(keys=['supported', 'evidence', 'why']) |
| `7f1d5e124300ed4b8f52200d` | dict(keys=['verdict', 'why']) |
| `7187ddbd430da4e7786f338b` | dict(keys=['supported', 'evidence', 'why']) |
| `b307999c9c9d0891c09aaf01` | dict(keys=['supported', 'evidence', 'why']) |
| `ff4d248c8529b93e53de98c6` | dict(keys=['supported', 'evidence', 'why']) |
| `5de45a4e44bf58da09e170fd` | dict(keys=['verdict', 'why']) |
| `31e2c9321720a39e96bae100` | dict(keys=['supported', 'evidence', 'why']) |
| `20c72fbc744d80ca45908efc` | dict(keys=['supported', 'evidence', 'why']) |
| `33997fc6198dadc95e9fa983` | dict(keys=['supported', 'evidence', 'why']) |
| `37bbf3fe5e1f6c74db81ce3a` | dict(keys=['verdict', 'why']) |
| `d044b9e796014c8094ebb5e3` | dict(keys=['supported', 'evidence', 'why']) |
| `e34a52696d833273de7a79f9` | dict(keys=['supported', 'evidence', 'why']) |
| `851e013a929794a68212a050` | dict(keys=['supported', 'evidence', 'why']) |
| `177a345accdc30545ca6990f` | dict(keys=['verdict', 'why']) |
| `321716bc6deb636c87be7400` | dict(keys=['supported', 'evidence', 'why']) |
| `49a6b80535042a789f0c49f7` | dict(keys=['supported', 'evidence', 'why']) |
| `1ff5f1843979264c83d0c39f` | dict(keys=['supported', 'evidence', 'why']) |
| `af704b609fdb11c342f033f9` | dict(keys=['verdict', 'why']) |
| `ac28d38d9cdf1620d80cf49f` | dict(keys=['supported', 'evidence', 'why']) |
| `d1cd7fd5b09dc20aed6f85f7` | dict(keys=['supported', 'evidence', 'why']) |
| `5452f8b64dc998a6a0f7fbf6` | dict(keys=['supported', 'evidence', 'why']) |
| `e42f044c6f150f945c17fdbb` | dict(keys=['verdict', 'why']) |
| `96e27977976e25fd36cfa199` | dict(keys=['supported', 'evidence', 'why']) |
| `2cf9153402d67f3ef7d232a1` | dict(keys=['supported', 'evidence', 'why']) |
| `f72dc45351fff697aceaab11` | dict(keys=['supported', 'evidence', 'why']) |
| `7ecf94d3da0b8c7d47d536b7` | dict(keys=['verdict', 'why']) |
| `f10d19f2db12294d66aefc71` | dict(keys=['supported', 'evidence', 'why']) |
| `e482bf74bdcdbacaea361942` | dict(keys=['supported', 'evidence', 'why']) |
| `1e6253cfd34ce55ab3b3cb5b` | dict(keys=['verdict', 'why']) |
| `58ace1acc7ce7c0d6ed544e6` | dict(keys=['verdict', 'why']) |
| `94f3eb0c3c6a946b4568b67d` | dict(keys=['supported', 'evidence', 'why']) |
| `384601cba7a2a518d6094faa` | dict(keys=['supported', 'evidence', 'why']) |
| `a410114de70d72ee37c7f8bf` | dict(keys=['supported', 'evidence', 'why']) |
| `b23a270aeab2721e8b5963c9` | dict(keys=['verdict', 'why']) |
| `b6d8f2d726a141137736183a` | dict(keys=['supported', 'evidence', 'why']) |
| `1cde33aa88901eeef0eba8db` | dict(keys=['supported', 'evidence', 'why']) |
| `f4770d3557a7fbac7e0ccae0` | dict(keys=['supported', 'evidence', 'why']) |
| `ec7db56f973ab294006125e9` | dict(keys=['verdict', 'why']) |
| `b850cdbd7967f46a59f16924` | dict(keys=['supported', 'evidence', 'why']) |
| `25df702ab040fdd374e40cf8` | dict(keys=['supported', 'evidence', 'why']) |
| `b447ef4b9765a28dfc643d4b` | dict(keys=['supported', 'evidence', 'why']) |
| `c2fdacf5e1f485ad12831861` | dict(keys=['verdict', 'why']) |
| `bfa93f4ec374bcbb5a19951c` | dict(keys=['supported', 'evidence', 'why']) |
| `2439a4fdeacf5c7e0a7d80a5` | dict(keys=['supported', 'evidence', 'why']) |
| `2700f73972f9e8c5e1a04322` | dict(keys=['supported', 'evidence', 'why']) |
| `f39b1957b83e5d2f0f22f777` | dict(keys=['verdict', 'why']) |
| `fcaf62baddb702eeb144944a` | dict(keys=['supported', 'evidence', 'why']) |
| `e5dc059ac51600389362ac73` | dict(keys=['supported', 'evidence', 'why']) |
| `e7e3a4bb481e52e526146a0b` | dict(keys=['supported', 'evidence', 'why']) |
| `82ce187c334012095442f797` | dict(keys=['supported', 'evidence', 'why']) |
| `8716a9fd147dbecf35bb1ac1` | dict(keys=['verdict', 'why']) |
| `9aabe9563ac39ca373a29240` | dict(keys=['supported', 'evidence', 'why']) |
| `aae175252457ac00fd6615db` | dict(keys=['verdict', 'why']) |
| `fdaeac63dc17d22a4263795d` | dict(keys=['supported', 'evidence', 'why']) |
| `f0ec89b417bc56361d49fcf9` | dict(keys=['supported', 'evidence', 'why']) |
| `3c1c1ab336b9dd60acc87ee8` | dict(keys=['verdict', 'why']) |
| `e6eb2c470a56de396080382d` | dict(keys=['supported', 'evidence', 'why']) |
| `6753423d37f35c14ba26c58a` | dict(keys=['supported', 'evidence', 'why']) |
| `112c6a8a5a2bee2153b14793` | dict(keys=['supported', 'evidence', 'why']) |
| `84bbe901572af6477b875c13` | dict(keys=['supported', 'evidence', 'why']) |
| `59c919decd7b0b3ae3ad5fd7` | dict(keys=['verdict', 'why']) |
| `07e3dbdc378de7fe0a8e26dc` | dict(keys=['supported', 'evidence', 'why']) |
| `75b0042b729ce10bff980bd8` | dict(keys=['verdict', 'why']) |
| `2d559699a323b914e7424bcd` | dict(keys=['supported', 'evidence', 'why']) |
| `e4b3cf7ddb6a9266ef62e167` | dict(keys=['supported', 'evidence', 'why']) |
| `7f859cfb4fb6f91e4eb4ff62` | dict(keys=['supported', 'evidence', 'why']) |
| `bb327128288b31272dbd8933` | dict(keys=['verdict', 'why']) |
| `563168fddfb1b6514c982187` | dict(keys=['supported', 'evidence', 'why']) |
| `c547f5639081c3313302ea38` | dict(keys=['supported', 'evidence', 'why']) |
| `c4d049543ea13da0ec42a7a2` | dict(keys=['supported', 'evidence', 'why']) |
| `a453e4cbb809861550f611e0` | dict(keys=['verdict', 'why']) |
| `65793133563284d12fc12004` | dict(keys=['supported', 'evidence', 'why']) |
| `af00d40a01db9632ad1cb1b0` | dict(keys=['supported', 'evidence', 'why']) |
| `e2dd019c886485bc180e9bfc` | dict(keys=['verdict', 'why']) |
| `8b2faa9c78ccc1d5b4304765` | dict(keys=['supported', 'evidence', 'why']) |
| `ba4a4961e0b770221d9c7e14` | dict(keys=['supported', 'evidence', 'why']) |
| `c4f5e0a805c839032dfb3b1e` | dict(keys=['supported', 'evidence', 'why']) |
| `d109366aff7c4c91d144599a` | dict(keys=['verdict', 'why']) |
| `fcaa57d2d9e12b018fcc69ce` | dict(keys=['supported', 'evidence', 'why']) |
| `6f610e5aa930a1c69951ab42` | dict(keys=['supported', 'evidence', 'why']) |
| `c9c9f8ea787b207cff6774ef` | dict(keys=['supported', 'evidence', 'why']) |
| `f5856c40d08299cff051342c` | dict(keys=['verdict', 'why']) |
| `b8f9a2e073d11b27bc7e2f96` | dict(keys=['supported', 'evidence', 'why']) |
| `2bc3398c19b08b76ef40ac06` | dict(keys=['verdict', 'why']) |
| `79fd711bcbcf10a26e0f9af4` | dict(keys=['supported', 'evidence', 'why']) |
| `14cb87a2651ae73c258c58c4` | dict(keys=['supported', 'evidence', 'why']) |
| `2df3f58a55980df29d82acc6` | dict(keys=['supported', 'evidence', 'why']) |
| `6e526b26c74487e3e4900d2f` | dict(keys=['verdict', 'why']) |
| `befe50682144a9f85a4567ac` | dict(keys=['supported', 'evidence', 'why']) |
| `56d9c31132e2203c704d7758` | dict(keys=['supported', 'evidence', 'why']) |
| `23b1570e94d263e3170cb9ef` | dict(keys=['supported', 'evidence', 'why']) |
| `44e7d86c0a9ac6be9504047d` | dict(keys=['verdict', 'why']) |
| `ab4d70c056d7f9b89e0c2049` | dict(keys=['supported', 'evidence', 'why']) |
| `620287be1941db183eb6cc77` | dict(keys=['supported', 'evidence', 'why']) |
| `820ac9c2dbf764fde5c3e537` | dict(keys=['supported', 'evidence', 'why']) |
| `dd38111f4a53731c7f4c0a63` | dict(keys=['verdict', 'why']) |
| `bc22c3da18165df320333d49` | dict(keys=['supported', 'evidence', 'why']) |
| `8cdd4ca162ccacc0996d726d` | dict(keys=['supported', 'evidence', 'why']) |
| `0f8b3e25aa459f2129c34cf9` | dict(keys=['verdict', 'why']) |
| `9f27db28414f259d779ed2d6` | dict(keys=['supported', 'evidence', 'why']) |
| `a4d348fb9978838d2dc9334c` | dict(keys=['supported', 'evidence', 'why']) |
| `920394f7e3530d6bb3141eda` | dict(keys=['supported', 'evidence', 'why']) |
| `e360dba6b55773ad498785e7` | dict(keys=['verdict', 'why']) |
| `4970624fa89ac3bad003f941` | dict(keys=['supported', 'evidence', 'why']) |
| `f0d2d724f760d369d17d2e8b` | dict(keys=['supported', 'evidence', 'why']) |
| `f2ea78f454217eff654e2feb` | dict(keys=['supported', 'evidence', 'why']) |
| `5d240a5f159f9aed652c1577` | dict(keys=['verdict', 'why']) |
| `dc6280088269ae9948c7fe0a` | dict(keys=['supported', 'evidence', 'why']) |
| `449021038a13706207798c91` | dict(keys=['supported', 'evidence', 'why']) |
| `dbf9e6f9cf82665f55fa4d88` | dict(keys=['verdict', 'why']) |
| `29d4b4e3c6f58788e1a08414` | dict(keys=['supported', 'evidence', 'why']) |
| `68a6b2ade123bf328c42af2e` | dict(keys=['supported', 'evidence', 'why']) |
| `daa835a7dadfc3dd17a777c4` | dict(keys=['supported', 'evidence', 'why']) |
| `2820e9fe198ed0c6e29d23a5` | dict(keys=['verdict', 'why']) |
| `30d5e8a7d99f3b5117bb601c` | dict(keys=['supported', 'evidence', 'why']) |
| `41992cf5e0688ad999b07336` | dict(keys=['supported', 'evidence', 'why']) |
| `064e25a7e9efbc400f344cca` | dict(keys=['verdict', 'why']) |
| `0f7f4f87f5651fd639f39642` | dict(keys=['supported', 'evidence', 'why']) |
| `6a21bc50d14d165306287360` | dict(keys=['supported', 'evidence', 'why']) |
| `4ff0ce98e6e09f59d36eb6cf` | dict(keys=['verdict', 'why']) |
| `2752c9d31cbd4280ae879435` | dict(keys=['supported', 'evidence', 'why']) |
| `79ab7e28ef88d76cba09f458` | dict(keys=['supported', 'evidence', 'why']) |
| `60ae3069113cfe907b57bbc7` | dict(keys=['supported', 'evidence', 'why']) |
| `3f8a62adadae466c9d5b17d7` | dict(keys=['verdict', 'why']) |
| `3e80b25b9784e05bcdbe173d` | dict(keys=['verdict', 'why']) |
| `442c795e130e85b839580f91` | dict(keys=['supported', 'evidence', 'why']) |
| `7fa6fd06e609bc084fa7e5ac` | dict(keys=['supported', 'evidence', 'why']) |
| `b0f05dafa970c44668920406` | dict(keys=['verdict', 'why']) |
| `0bd19c46aa73f4fa5595c007` | dict(keys=['verdict', 'why']) |
| `e6e444b6a6e6f61ab1c2957d` | dict(keys=['supported', 'evidence', 'why']) |
| `01203f047e4a24a598f8add1` | dict(keys=['supported', 'evidence', 'why']) |
| `12de6d7a861b89a27aa15e50` | dict(keys=['verdict', 'why']) |
| `a2d45fcba68a5e4bdd167474` | dict(keys=['supported', 'evidence', 'why']) |
| `e53e224be9aea523e5df7c8d` | dict(keys=['supported', 'evidence', 'why']) |
| `d77de35031d29ee58046605c` | dict(keys=['verdict', 'why']) |
| `283a996f73ac5065e386038d` | dict(keys=['supported', 'evidence', 'why']) |
| `cd4bfad4552c33c702eb9ed3` | dict(keys=['supported', 'evidence', 'why']) |
| `d6ac78a40debf2496107fdbc` | dict(keys=['supported', 'evidence', 'why']) |
| `8d72aa52e63e99016feb7a88` | dict(keys=['verdict', 'why']) |
| `f5de15bbce21308005f72ae7` | dict(keys=['supported', 'evidence', 'why']) |
| `7619bc05e5297cd854220266` | dict(keys=['verdict', 'why']) |
| `f14b7563847c74a27b797dfc` | dict(keys=['supported', 'evidence', 'why']) |
| `f43d177dffb231a6200aaf5c` | dict(keys=['supported', 'evidence', 'why']) |
| `6917b0357de63ea074d92e96` | dict(keys=['verdict', 'why']) |
| `9ac32b057ef056282615db62` | dict(keys=['supported', 'evidence', 'why']) |
| `5d84c5d01bbcfe3e1bbef8b0` | dict(keys=['supported', 'evidence', 'why']) |
| `b096beac8364c81426b11516` | dict(keys=['supported', 'evidence', 'why']) |
| `1e4f725b6ebe5117a43088aa` | dict(keys=['verdict', 'why']) |
| `65c201f06c05b097d3bb0f1c` | dict(keys=['supported', 'evidence', 'why']) |
| `c876f9afbbd876e0ccc0247e` | dict(keys=['verdict', 'why']) |
| `60925013c22fb3d21ca71cac` | dict(keys=['supported', 'evidence', 'why']) |
| `eb44654e1da80dc6c077c9c8` | dict(keys=['supported', 'evidence', 'why']) |
| `6479530bb96c2039d40dc203` | dict(keys=['verdict', 'why']) |
| `8b052125bb9b15f68654a3c4` | dict(keys=['supported', 'evidence', 'why']) |
| `8a3975c70084bbee626e7962` | dict(keys=['supported', 'evidence', 'why']) |
| `335a231455046c6206b3f323` | dict(keys=['verdict', 'why']) |
| `159308d1b1d5c4ee197b76d3` | dict(keys=['supported', 'evidence', 'why']) |
| `2f5ef63e2fcab3a211791217` | dict(keys=['supported', 'evidence', 'why']) |
| `4567f8c51b6f2f572252644a` | dict(keys=['verdict', 'why']) |
| `9812d6efa129954f6999f8b5` | dict(keys=['supported', 'evidence', 'why']) |
| `e4357d08ba9041b6253c7aaf` | dict(keys=['supported', 'evidence', 'why']) |
| `24d4d8e92bc6f94e67dd15b0` | dict(keys=['supported', 'evidence', 'why']) |
| `28b84d8222f07d3fa377a207` | dict(keys=['verdict', 'why']) |
| `4aa793750826f187ea163f16` | dict(keys=['supported', 'evidence', 'why']) |
| `5f6a7747c78d52e76222e78a` | dict(keys=['supported', 'evidence', 'why']) |
| `f9ac1abc1aa53d01315a22d1` | dict(keys=['supported', 'evidence', 'why']) |
| `70c79c111cdeb53cbd340a55` | dict(keys=['verdict', 'why']) |
| `0a66bacdee6e1f0011aada8c` | dict(keys=['supported', 'evidence', 'why']) |
| `e85f0ee0c40971a85ae77725` | dict(keys=['supported', 'evidence', 'why']) |
| `29fb15b2f87a31d8705f43d6` | dict(keys=['verdict', 'why']) |
| `3ba8b3f2c1ac4845d20736b3` | dict(keys=['verdict', 'why']) |
| `e1252a4d779eab6e48f5402a` | dict(keys=['supported', 'evidence', 'why']) |
| `34f6948d9072a2b986e81517` | dict(keys=['supported', 'evidence', 'why']) |
| `080b237acbaef855860890b8` | dict(keys=['supported', 'evidence', 'why']) |
| `cbed64b001928d282cea6f8b` | dict(keys=['supported', 'evidence', 'why']) |
| `0373fa63c5d28709203b9f56` | dict(keys=['verdict', 'why']) |
| `e6956fd02924d7128f90db0b` | dict(keys=['supported', 'evidence', 'why']) |
| `ad855729dd67a60ea58d690d` | dict(keys=['supported', 'evidence', 'why']) |
| `e792f5d61c556553719d3b75` | dict(keys=['supported', 'evidence', 'why']) |
| `783fa874455697209b15996e` | dict(keys=['verdict', 'why']) |
| `5d11e9e3aed7b71975a41c71` | dict(keys=['supported', 'evidence', 'why']) |
| `5ab751511c3d5f5854861c82` | dict(keys=['supported', 'evidence', 'why']) |
| `fae3e3f642282c0a089ef856` | dict(keys=['verdict', 'why']) |
| `f302812dcbd33248073ffc1f` | dict(keys=['supported', 'evidence', 'why']) |
| `3561548089a28403f493e52a` | dict(keys=['supported', 'evidence', 'why']) |
| `37999a2d9d2d5d02fc579d3e` | dict(keys=['supported', 'evidence', 'why']) |
| `ca55d6f9d1fac6036f61d703` | dict(keys=['supported', 'evidence', 'why']) |
| `accb9d46c8bc5dbbe99dbd96` | dict(keys=['verdict', 'why']) |
| `8dd3cd5c41d5118f8c829cb9` | dict(keys=['supported', 'evidence', 'why']) |
| `1148180b15017af36a3f0cc4` | dict(keys=['verdict', 'why']) |
| `118a64ed855b7288f0bc05f7` | dict(keys=['supported', 'evidence', 'why']) |
| `428d9ecb7cf5566b826f146d` | dict(keys=['supported', 'evidence', 'why']) |
| `8423a276769c2fe1054ab899` | dict(keys=['supported', 'evidence', 'why']) |
| `9fa88d50d275be205105aa6e` | dict(keys=['verdict', 'why']) |
| `b5964d1c5cbf0c2ce27bd14f` | dict(keys=['supported', 'evidence', 'why']) |
| `e2839cf317f338aee7fc995d` | dict(keys=['supported', 'evidence', 'why']) |
| `2d01d788d24af4efa8c98ee5` | dict(keys=['verdict', 'why']) |
| `106dced029cf8386efec5ee3` | dict(keys=['supported', 'evidence', 'why']) |
| `aed990fa793cd5e20afebe48` | dict(keys=['supported', 'evidence', 'why']) |
| `c100a17bb8dac6fc3e50137c` | dict(keys=['verdict', 'why']) |
| `6e053c9ee7d6b377f9b3dd86` | dict(keys=['supported', 'evidence', 'why']) |
| `bd56b23567915968cdf87278` | dict(keys=['supported', 'evidence', 'why']) |
| `ac3208859896f87c463fb555` | dict(keys=['supported', 'evidence', 'why']) |
| `381136b3a44edeb4cfedf0d6` | dict(keys=['verdict', 'why']) |
| `6a7ef23c1b97e7fb48ac60bc` | dict(keys=['supported', 'evidence', 'why']) |
| `c6768781d3d34e71033bd9e9` | dict(keys=['supported', 'evidence', 'why']) |
| `44fa30332c5ac1daad68ffdc` | dict(keys=['supported', 'evidence', 'why']) |
| `6b1e121af700cc85f6f27f0f` | dict(keys=['verdict', 'why']) |
| `4f8f067b10306b82d6b2d5f9` | dict(keys=['supported', 'evidence', 'why']) |
| `8df7dcce557bee81b5dd0dc5` | dict(keys=['supported', 'evidence', 'why']) |
| `3fd694505f7c02747e4c3f74` | dict(keys=['supported', 'evidence', 'why']) |
| `7893006b57282943c24a4e06` | dict(keys=['supported', 'evidence', 'why']) |
| `0c04a9a46452392af8d62b8c` | dict(keys=['verdict', 'why']) |
| `212107dfd8f4cda898cab16a` | dict(keys=['supported', 'evidence', 'why']) |
| `b9d49c23ba260afffed3463d` | dict(keys=['verdict', 'why']) |
| `f8d70b7fbaf10642804b65b3` | dict(keys=['verdict', 'why']) |
| `ba6782204c99234df45a6612` | dict(keys=['supported', 'evidence', 'why']) |
| `0caca71226e76025f9f8d483` | dict(keys=['supported', 'evidence', 'why']) |
| `a553a5d58b6749b47dd8c96f` | dict(keys=['verdict', 'why']) |
| `db53de30fa44adc868947ef5` | dict(keys=['supported', 'evidence', 'why']) |
| `a0998d07cdac14b73bae89a2` | dict(keys=['supported', 'evidence', 'why']) |
| `929c2df88dfdd775f9baadeb` | dict(keys=['supported', 'evidence', 'why']) |
| `75ecff964dc4178ca2156b26` | dict(keys=['verdict', 'why']) |
| `1f97da982ecb7fb013ab8c6e` | dict(keys=['supported', 'evidence', 'why']) |
| `1818bb2f5ba1056d78259879` | dict(keys=['supported', 'evidence', 'why']) |
| `fda1eb074234a1687d168e4e` | dict(keys=['supported', 'evidence', 'why']) |
| `66531d45da44b1ee05fc18a3` | dict(keys=['supported', 'evidence', 'why']) |
| `bd992f3b6041977620800c5d` | dict(keys=['verdict', 'why']) |
| `3c1a26048f19d57707b8769d` | dict(keys=['supported', 'evidence', 'why']) |
| `eecabd15b213895ec0cc260e` | dict(keys=['supported', 'evidence', 'why']) |
| `133af09392aa841d5354ea98` | dict(keys=['supported', 'evidence', 'why']) |
| `20096a499c7c9cfc232a08e8` | dict(keys=['verdict', 'why']) |
| `c4e2b70185680839ff8f9510` | dict(keys=['supported', 'evidence', 'why']) |
| `de6d912297c01ef3a541d347` | dict(keys=['supported', 'evidence', 'why']) |
| `bc249fb17a50ce1dd1821406` | dict(keys=['verdict', 'why']) |
| `e0c5dc03305a9be9ed81adc8` | dict(keys=['supported', 'evidence', 'why']) |
| `67edc5d13cc49c3ff9f1b1be` | dict(keys=['verdict', 'why']) |
| `15aeff86b82ca4c094d149a1` | dict(keys=['supported', 'evidence', 'why']) |
| `d8687e568a5977e20e3748bc` | dict(keys=['supported', 'evidence', 'why']) |
| `082abda15da6735dcac6dac0` | dict(keys=['verdict', 'why']) |
| `c4cded53fcbbeb687ce39da2` | dict(keys=['supported', 'evidence', 'why']) |
| `e2b8d704f0d8b5c8a2b8bc4c` | dict(keys=['supported', 'evidence', 'why']) |
| `ff487b5cf88ecf2750449836` | dict(keys=['verdict', 'why']) |
| `4b528c1f03360a5acf553c2c` | dict(keys=['supported', 'evidence', 'why']) |
| `77fafa2e605629403f0bc8f2` | dict(keys=['supported', 'evidence', 'why']) |
| `7a10c278d9e25cd749b93017` | dict(keys=['verdict', 'why']) |
| `84a8848905576572f2280994` | dict(keys=['verdict', 'why']) |
| `3888bdf500b03c638a36902b` | dict(keys=['supported', 'evidence', 'why']) |
| `1eb3b82f9cab768f85c01986` | dict(keys=['verdict', 'why']) |
| `325ef8d6dd30b93029fadb38` | dict(keys=['supported', 'evidence', 'why']) |
| `e83dec155b72df07065aeb92` | dict(keys=['verdict', 'why']) |
| `3ace936855258fa060bce1f1` | dict(keys=['supported', 'evidence', 'why']) |
| `de76bc76b62d3bdf31b3c848` | dict(keys=['supported', 'evidence', 'why']) |
| `3beee8787f859c2cf816dcb8` | dict(keys=['supported', 'evidence', 'why']) |
| `3e65862ccc44fcd43488e354` | dict(keys=['verdict', 'why']) |
| `2a162e27b159ab769121a47b` | dict(keys=['supported', 'evidence', 'why']) |
| `3b743b02a598493ab10d9f0f` | dict(keys=['supported', 'evidence', 'why']) |
| `baa206924f58f07ae1283150` | dict(keys=['supported', 'evidence', 'why']) |
| `afb042d8ba21fb02284cfb4e` | dict(keys=['verdict', 'why']) |
| `dc5dd4c37da6971d130709e4` | dict(keys=['supported', 'evidence', 'why']) |
| `dc8d360fb40befbf644404d2` | dict(keys=['supported', 'evidence', 'why']) |
| `ac223cef5707fc24c68eb817` | dict(keys=['verdict', 'why']) |
| `71c529ad888d2fc86d5add1b` | dict(keys=['supported', 'evidence', 'why']) |
| `f13d4f492eccf147e7650a0b` | dict(keys=['verdict', 'why']) |
| `4f0f66834758eb9a565ee082` | dict(keys=['supported', 'evidence', 'why']) |
| `f73549bb5281279d1b99c3b8` | dict(keys=['supported', 'evidence', 'why']) |
| `6e69693ea51cf57dce0cdf50` | dict(keys=['verdict', 'why']) |
| `81e091a9ecc792aa74737b5f` | dict(keys=['supported', 'evidence', 'why']) |
| `48eef96d2fdae6448159688c` | dict(keys=['verdict', 'why']) |
| `bd29b06a34947ae955ba874b` | dict(keys=['supported', 'evidence', 'why']) |
| `13822c40269e708b1189c010` | dict(keys=['verdict', 'why']) |
| `4e2ed72bfb7a84dafb0b0227` | dict(keys=['supported', 'evidence', 'why']) |
| `3f322ff82d64dda5284fdec1` | dict(keys=['verdict', 'why']) |
| `d60cf643d9aa0642f542dee1` | dict(keys=['supported', 'evidence', 'why']) |
| `8422684b02aee58c454701b2` | dict(keys=['verdict', 'why']) |
| `a877215b12cf774b7fcbed4c` | dict(keys=['supported', 'evidence', 'why']) |
| `c6610c6cae520c022e231f2c` | dict(keys=['supported', 'evidence', 'why']) |
| `77c315495eba1a17e2471d07` | dict(keys=['verdict', 'why']) |
| `579fc0465cb0dea4b346dd5f` | dict(keys=['supported', 'evidence', 'why']) |
| `8d3a3ef65074c7681f996e7d` | dict(keys=['supported', 'evidence', 'why']) |
| `f65de95b46290b905e49b50b` | dict(keys=['supported', 'evidence', 'why']) |
| `9777807270e20794cbcb7fa5` | dict(keys=['verdict', 'why']) |
| `77f45ddf194d1660c6881a81` | dict(keys=['supported', 'evidence', 'why']) |
| `ff7b9ef5564adbaf6cf955f6` | dict(keys=['supported', 'evidence', 'why']) |
| `3637b604fadaadfc28d46894` | dict(keys=['verdict', 'why']) |
| `ca090bc3f45131ac184ef12f` | dict(keys=['supported', 'evidence', 'why']) |
| `19a93effb31cf14a54d57a7c` | dict(keys=['verdict', 'why']) |
| `bc01bdce4ebd59057f4d67ab` | dict(keys=['supported', 'evidence', 'why']) |
| `c7d181f2cb012242937937a1` | dict(keys=['supported', 'evidence', 'why']) |
| `0dfb4295b3a8c00303a2994a` | dict(keys=['verdict', 'why']) |
| `8c061cc25889f2eeded163af` | dict(keys=['supported', 'evidence', 'why']) |
| `0eda5c1bbba5389cdb97ae1f` | dict(keys=['verdict', 'why']) |
| `c75dec007174d20f6e1981b6` | dict(keys=['supported', 'evidence', 'why']) |
| `71abd908f6745881152a5800` | dict(keys=['verdict', 'why']) |
| `4c4a9a4ee6af2e909b358063` | dict(keys=['supported', 'evidence', 'why']) |
| `44b817890a2b634ee1f97f7c` | dict(keys=['supported', 'evidence', 'why']) |
| `0f5f9efa0ef3d65e4a7e5db8` | dict(keys=['supported', 'evidence', 'why']) |
| `369ffc90319babc7df1ab422` | dict(keys=['verdict', 'why']) |
| `66c96fa3f43d03be7f0b95f8` | dict(keys=['supported', 'evidence', 'why']) |
| `1c2263ba0f84fe4a27b28dee` | dict(keys=['supported', 'evidence', 'why']) |
| `7a97bbf765e1144ed6e58290` | dict(keys=['verdict', 'why']) |
| `61ad470ea7037a928fdfac48` | dict(keys=['supported', 'evidence', 'why']) |
| `67b1830ffd1f3f1dac9b6dcf` | dict(keys=['supported', 'evidence', 'why']) |
| `34a7e3a56c0a0b3b64542699` | dict(keys=['verdict', 'why']) |
| `524d3525820f1933b0743463` | dict(keys=['supported', 'evidence', 'why']) |
| `2b8a91e1243a8a085ac740b4` | dict(keys=['supported', 'evidence', 'why']) |
| `bd36a445d324757cf0df0573` | dict(keys=['verdict', 'why']) |
| `b4ec30d29d1f93df510b8490` | dict(keys=['supported', 'evidence', 'why']) |
| `fea25cef631ae80e7b239ddb` | dict(keys=['supported', 'evidence', 'why']) |
| `dcf9df2236f5be9bad87431e` | dict(keys=['verdict', 'why']) |
| `179df87c6d16eee71b69e358` | dict(keys=['supported', 'evidence', 'why']) |
| `2a7762aa8859d665f5985410` | dict(keys=['supported', 'evidence', 'why']) |
| `a3d10de006c542073fe4a374` | dict(keys=['supported', 'evidence', 'why']) |
| `4b1d44fa35d8ca404f1e4feb` | dict(keys=['verdict', 'why']) |
| `40f7d6f7a2e51d2831c943fd` | dict(keys=['supported', 'evidence', 'why']) |
| `46564f75c72079551fc58738` | dict(keys=['verdict', 'why']) |
| `4139ca6563a457c2e8decc16` | dict(keys=['supported', 'evidence', 'why']) |
| `4e91bb0904e11dd914260806` | dict(keys=['verdict', 'why']) |
| `318f333ddd859e18bb761a22` | dict(keys=['supported', 'evidence', 'why']) |
| `7af2144895132b880b4e2756` | dict(keys=['verdict', 'why']) |
| `b081a5eab90ba5f6babc61b8` | dict(keys=['supported', 'evidence', 'why']) |
| `a8314c9803cba6a6be840069` | dict(keys=['verdict', 'why']) |
| `ae279de20abca2ebb85276a2` | dict(keys=['supported', 'evidence', 'why']) |
| `37bd9f462de9ddb13abf6b23` | dict(keys=['verdict', 'why']) |
| `d58aefc1fbc9f94b888bc730` | dict(keys=['supported', 'evidence', 'why']) |
| `db788d3576633a3b3f9e73d6` | dict(keys=['supported', 'evidence', 'why']) |
| `ef3b9168367664453b4533bf` | dict(keys=['supported', 'evidence', 'why']) |
| `14c66e634c1b6544f0c64f7c` | dict(keys=['verdict', 'why']) |
| `f1b8b046951059e2c4829413` | dict(keys=['supported', 'evidence', 'why']) |
| `9e8ae7ec7d8c0e708ec397a4` | dict(keys=['supported', 'evidence', 'why']) |
| `27fde941ed80a15f559af125` | dict(keys=['verdict', 'why']) |
| `51716a7ab88cb330600276e9` | dict(keys=['supported', 'evidence', 'why']) |
| `106c2bcc371c613fe7aff240` | dict(keys=['supported', 'evidence', 'why']) |
| `25fcd44c019be3f3f2983ca8` | dict(keys=['verdict', 'why']) |
| `b2f9a39c91db1dd6c5b5e4be` | dict(keys=['supported', 'evidence', 'why']) |
| `314eb131a67abbd5267d91d6` | dict(keys=['supported', 'evidence', 'why']) |
| `52ff6a809ea4e617a1f9d20a` | dict(keys=['verdict', 'why']) |
| `a42d053913cef3de7d5ba70a` | dict(keys=['supported', 'evidence', 'why']) |
| `501131cec4875587d2e812ef` | dict(keys=['verdict', 'why']) |
| `a65888033070f5b1da5c2803` | dict(keys=['supported', 'evidence', 'why']) |
| `a1b06d7979085fa9e4140446` | dict(keys=['supported', 'evidence', 'why']) |
| `068843e261797e9b717de53c` | dict(keys=['verdict', 'why']) |
| `865792a5447027cd13291ef9` | dict(keys=['supported', 'evidence', 'why']) |
| `9849e8f3d03bb7f5a2c9a2de` | dict(keys=['supported', 'evidence', 'why']) |
| `a041014fec965d05e6e7ebca` | dict(keys=['verdict', 'why']) |
| `05bc8d82a1d18ba33d6da0a0` | dict(keys=['supported', 'evidence', 'why']) |
| `5d609bff6b4bf80f6b02fedc` | dict(keys=['verdict', 'why']) |
| `08538f240bea90d4e62c55b8` | dict(keys=['supported', 'evidence', 'why']) |
| `0af610e81f39580e27d5961c` | dict(keys=['supported', 'evidence', 'why']) |
| `949e3d6750a7cf6c8605017c` | dict(keys=['verdict', 'why']) |
| `94ee88745bc7e27a76c39c7b` | dict(keys=['supported', 'evidence', 'why']) |
| `ce2c3d7c2df7f65a75026225` | dict(keys=['supported', 'evidence', 'why']) |
| `12cc6d0bb604968e2deee19a` | dict(keys=['supported', 'evidence', 'why']) |
| `28779964d06e58dfc8c35309` | dict(keys=['verdict', 'why']) |
| `edb700db77172bd3496eb16a` | dict(keys=['supported', 'evidence', 'why']) |
| `ceb281fd1c94e0e200ddde8c` | dict(keys=['supported', 'evidence', 'why']) |
| `22953c0540bea91dd501d241` | dict(keys=['verdict', 'why']) |
| `4a03a618d4b9a9ddaab4d86e` | dict(keys=['supported', 'evidence', 'why']) |
| `5095fee04000530cc928609a` | dict(keys=['verdict', 'why']) |
| `5ba22f2c8e26d711e31f9513` | dict(keys=['supported', 'evidence', 'why']) |
| `fed87abd57f93b1412b8c72a` | dict(keys=['verdict', 'why']) |
| `a8df2333836a1705b5c56db6` | dict(keys=['supported', 'evidence', 'why']) |
| `90c84e590dd7c7087262f451` | dict(keys=['supported', 'evidence', 'why']) |
| `3d01fc48dd70fdf9941e7837` | dict(keys=['verdict', 'why']) |
| `8dda6294e9db767d76915cf9` | dict(keys=['supported', 'evidence', 'why']) |
| `7bfb356c7c19f8b749a58416` | dict(keys=['supported', 'evidence', 'why']) |
| `0d46f2890a9cdeee0899a7de` | dict(keys=['supported', 'evidence', 'why']) |
| `e85f23cade2df81b11e1384a` | dict(keys=['verdict', 'why']) |
| `4305e630e96baa17149778d3` | dict(keys=['supported', 'evidence', 'why']) |
| `ece2e1ff5426b6168cb6969e` | dict(keys=['supported', 'evidence', 'why']) |
| `55c8db857c3d1256238095aa` | dict(keys=['supported', 'evidence', 'why']) |
| `a779e47bb138f87fcee53508` | dict(keys=['verdict', 'why']) |

### `data/interim/g_full_answers.json`

- 크기 434.9KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-23 03:42
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 96개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `14살 미만 애들 현금IC카드 본인이 직접 받을 수 있어?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 입출금 예금 이율은 바뀌면 언제부터 적용돼요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 환전지갑에서 외화 찾을 때 신분증 꼭 가져가야 해요? 대리인은 어떻게 되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `주택청약종합저축 가입하고 청약 1순위 되려면 어느 정도 기간과 납입횟수가 필요한가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `Range Forward 상품은 만기 환율이 정해진 범위 밖으로 나가면 어떻게 처리되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `소득공제 장기펀드를 5년 안에 중간에 해지하면 세금이 얼마나 나와요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 디폴트옵션 안정투자형 포트폴리오 1 상품이 위험등급이 바뀌거나 승인이 취소되면 어떻게 알려주나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `휴대폰 번호는 안 바꾸고 기기만 바꿨는데 금융인증서 클라우드 계속 쓸 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 가계 대출받을 때 이자율은 중간에 바꿀 수 있어?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 CD 수익률은 누가 어떻게 정하는 거예요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 기업 인터넷뱅킹에서 급여 이체는 언제까지 할 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 디폴트옵션 중립투자형 포트폴리오 1 상품 가입 후 따로 운용 방법 지시 안 하면 어떻게 돼요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 디폴트옵션 중립투자형 포트폴리오 2는 중간에 해지하면 원금 손실 가능성이 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나원큐에서 특정 계좌 조회 못하게 하려면 어떻게 해?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나원큐 로그인할 때 기본 로그인 방법을 바꾸려면 어떻게 해야 해요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `기업 대출할 때 원화 금리는 어떤 기준으로 정해지나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `일임형 개인종합자산관리계좌(ISA) 가입 시 투자 가능한 위험등급 모델포트폴리오 제시 기준은 무엇인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `기업대출 자금용도별 구분에는 어떤 항목들이 포함되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `Enhanced Forward라는 상품은 만기 환율이 예상과 다르게 떨어지면 어떤 위험이 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `이자율 스왑(고객 고정금리 지급) 중도해지는 어떻게 가능한가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `구 외환은행 프라임등급 계속 유지하려면 어떻게 해야 해요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 이자율 스왑 상품의 위험등급과 손실범위는 어떻게 되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `청년 주택드림 청약통장 가입할 때 소득 증명서류로는 어떤 게 필요한가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나일임형ISA 저위험포커스 모델포트폴리오가 포함한 금융상품은 원금보장이 되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행에서 전자금융 거래 내역을 못 받을 때 은행이 어떻게 알려줘야 해?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `해외 출국 전 하나은행 전자금융사기예방서비스 설정은 어떻게 해야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 예금 거래는 보통 어디에서 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `종합부동산세는 하나은행 자료에 따르면 국세인가 지방세인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `대출일과 월 납입일이 다르면 첫 이자는 언제 내야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나 합 서비스는 몇 살 이상부터 창구에서 바로 신청할 수 있어?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 ISA 모델포트폴리오(최고위험_밸런스) 상품에서 중도해지 시 과세특례 추징이 발생하는 조건은 무엇인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `골드바 살 때는 가격 계산할 때 수수료랑 세금 어떻게 붙어요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 대출이자 계산 시 '한편넣기'란 무엇을 의미하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `해외에서 돈 받을 때 해외 사람이 우리 은행에 보내려면 뭘 알려줘야 해?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `환전 신청을 했는데 아직 처리 안 됐어요. 이럴 때 더 신청할 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `일임형 개인종합자산관리계좌(ISA) 계약 만료 후 수수료는 어떻게 처리됩니까?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `ISA 계약을 중간에 해지할 때 어떤 특별한 사유가 있어야 해지할 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 브라우저인증서 서비스는 2020년 12월 10일 이후 신규 발급이 가능한가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 카드로 해외에서 현지 돈을 찾으려면 어떤 서비스가 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `은행이 돈을 다 갚으라고 통보를 늦게 하면 저는 언제부터 바로 갚아야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `마이데이터서비스 약관이 갑자기 바뀌면 은행이 어떻게 알려줘요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `전세자금대출에서 중도상환해약금 산정 기준은 무엇인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `은행에서 빌린 돈을 10일 넘게 못 갚으면 어떻게 되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `사업자 주거래 우대통장 가입 후 최초 다음달까지 제공되는 수수료 우대서비스는 무엇인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `Enhanced Forward 상품에서 환율 변동이 평가손익에 어떤 영향을 주나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `통장 없이 거래할 때 도장이나 서명은 꼭 제출해야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `예금 토큰 전환이 제한되는 경우는 어떤 상황인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `최고위험 밸런스 포트폴리오 어떤 사람한테 맞아요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 마이너스통장 대출 이자는 어떤 기준으로 출금되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `외화 송금한 후에 돈 보내는 걸 취소하거나 바꿀 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `전세자금대출 갚을 때 빚이 여러 개면 어떤 순서로 갚아요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `상속으로 집을 받았는데 그 집이 수도권 밖 85㎡ 이하 단독주택일 때, 다른 지역으로 이사 가면 어떤 상황이 되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `OTP 이용 중 '보정거래 필요' 오류가 발생하면 어떻게 해야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 대출 중간에 갚으면 수수료 어떻게 계산돼요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `제가 13살인데 해외에서 계좌를 이용하려면 ATM에서 돈을 뺄 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나 화상상담데스크 이용 가능시간은 어떻게 되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 예금 토큰은 모바일 앱에서 어떻게 가입해요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `해외에서 현지 돈을 뽑을 때 수수료가 어떻게 되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `해외 ATM 출금 서비스의 1회 출금 표준한도는 얼마인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 글로벌ATM에서 해외카드로 잔액 조회할 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `폰뱅킹 신청 시 관공서 서류 제출 기준은?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `저축예금 통장에 사기 의심이 생기면 거래가 어떻게 제한돼요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `월급이 7000만 원 이하이고 집이 없는데, 다음 해 2월 말까지 어떤 서류를 은행에 내야 소득공제를 받을 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `『하나은행 디폴트옵션 안정투자형 포트폴리오 3』 운용방법 변경 시 가입자가 선택할 수 있는 조치는 무엇인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `기업전자금융 서비스에서 개인사업자가 ERP랑 연결하면 기본수수료가 얼마나 나와?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `내가 가입할 때 다른 사람 정보를 써서 신청하면 계약을 바로 끊나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `주택청약저축에서 선납한 금액은 언제 인정회차로 산정되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 인터넷뱅킹에서 1천만원 넘게 이체하려면 어떻게 해야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `대출받은 뒤에 원금은 안 내고 이자만 낼 수 있는 기간이 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 심플이체에서 이체 내역을 수정할 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 가계 대출할 때 인지세는 누가 부담해요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `전세자금 대출할 때 배우자랑 합친 자산이 너무 많으면 우대금리 받을 수 없나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나 합(마이데이터 서비스) 이용 시 만 14세 미만 손님의 거래 제한 내용은 무엇인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `돈 빌린 사람이 약속을 안 지켜서 은행이 보증인에게 알릴 때, 은행은 얼마나 빨리 알려야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 환전지갑 선물하기 서비스에서 하나머니 환급 절차는 어떻게 이루어지나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `내가 자급제폰에 안드로이드 10 쓰는데 스마트 간편인증이 안 돼요. 어떻게 해야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `제가 대출받을 때 알려준 집 보유 수가 사실과 다르면 대출에 문제가 생기나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `기업용 대출 연체했을 때 은행이 받는 지연이자는 어떻게 계산돼요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `부부 합산 소득이 7천만원 넘으면 어떤 대출을 못 받나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 내 계좌이체 시 보안매체나 공동인증서 사용 여부는?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `제휴사 해외 ATM 출금 서비스에서 특정 제휴사 환율 적용 방식은 무엇인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `외화를 은행 외화통장에 넣으면 수수료가 어떻게 되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 디폴트옵션 중립투자형 포트폴리오 1 상품, 만기 전에 돈 찾으면 이자가 어떻게 돼요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 통해 마이데이터 정보 보내달라 할 땐 뭘 알려줘야 해?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `개인형IRP에서 만기 자금 운용 지시가 없을 때 어떻게 운용되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `소득공제 장기펀드 소득공제 대상자 요건은 무엇인가요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `은행이 서류를 잃어버리거나 다치게 하면 제가 가진 다른 증서로 빚을 갚아야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나원큐기업 환전지갑 서비스에서 재환전 시 건당 금액이 1백만원 초과하면 어떻게 처리해야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 제휴 환전지갑으로 외화 받을 때 직접 가서 신분증 보여줘야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `내 계좌에 돈이 부족하면 인터넷으로 돈 보내기가 안 되나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 저축예금에서 날짜 없는 수표가 들어오면 어떻게 처리해줘?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 예금 토큰 해지 시 수수료나 위약금이 발생합니까?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `자유저축예금 이자는 언제 받을 수 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `대출금을 제때 못 갚으면 집 팔고 남은 돈 더 내야 하나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `하나은행 디폴트옵션 안정형 포트폴리오의 투자원금 손실 가능성은 어떻게 설명되어 있나요?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |
| `외국인도 하나은행 인터넷뱅킹으로 해외송금하려면 어떻게 해야 해?` | dict(keys=['question', 'decision', 'answer', 'not_found', 'refusal_reason', 'stage']) |

### `reports/refusal_eval_2026-08-23.csv`

- 크기 7.8KB / CSV, 총 53행 / 수정 2026-08-23 03:37
- **RAGAS 적합도: 부분** — 누락: contexts, answer
- 역할 추정:
  - question: `﻿query_id`, `question`
  - refusal: `refusal_type`
  - verdict: `score_top1`
- 키 11개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿query_id` | r001 |
| `question` | 제 연체 기록이 대출 연체 이자에 어떻게 영향을 미치나요? |
| `expected` | refuse |
| `refusal_type` | personalized |
| `predicted` | refuse |
| `stage` | pattern |
| `reason` | personalized |
| `score_top1` | 0.6025 |
| `gap_1_5` | 0.0399 |
| `blank_ratio` | 0.0 |
| `correct` | True |

### `reports/refusal_eval_2026-08-23.json`

- 크기 1.6KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-23 03:37
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 6개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `date` | 2026-08-23 |
| `eval_set` | dict(keys=['total', 'refuse', 'answer']) |
| `retrieval` | dict(keys=['model', 'alpha']) |
| `thresholds` | dict(keys=['hard_refuse_top1', 'hard_refuse_gap15', 'blank_ratio']) |
| `stages` | dict(keys=['0단계만 (질문 패턴)', '0+1단계 (패턴+검색)', '0+1+2단계 (LLM 포함)']) |
| `by_type` | dict(keys=['personalized', 'time_variant', 'blank_value', 'out_of_scope']) |

### `data/interim/refusal_verify_cache_e5-small_a05_gpt-41-mini.json`

- 크기 7.1KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-23 03:37
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 44개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `bb74150d489df9966169b9c5` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `0699809af9a0e4444e24c7e3` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `d8c1d1109ebbf28a75952d87` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `28cc2815279bfd693d61045c` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `bc6a0819b83a5432a1a38e10` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `cd46cd07050a56996801945a` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `5604608f325c5ffbb1cd1542` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `3972a21359c3225651594da6` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `0cfa21154dfedff84f106ae1` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `a9dcc15260f89b461bc53855` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `511e3a91aa4f4849838f2d51` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `d46a62b1912ee14fb4cceac7` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `dbdb895945b6298b37f6d7d0` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `7bb76f9f572750982f425d86` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `c6946376b1bc72b586ee684e` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `29df4bd2728cfb837da9d7d9` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `39e4d315b283badf1b635bed` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `78e054c4c9eac72fc2738838` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `03df878e8b914ce8593f488c` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `8f481a53121e73efebc1ef9b` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `e77f173577fb5fd52ca57ba3` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `0f96c62fa641ce631ce70094` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `4afe0617f346ae4cee19d786` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `de25dca6d625d19551f65791` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `4680502b69d0a8da7af1f414` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `b186e6c79556a43f6dd79a0f` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `32e6dcdc5ffac7af3f782fba` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `e0e208bb6fc1ce9c0c9a26bc` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `a4fd9e61fa0bd401784c8c75` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `daa8b37147cc0105606f934f` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `c933a167ad75ba845fb29c78` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `8814d4184f60205c18636741` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `c0ff2ce019a12fe1d3d4d2ee` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `19e703791f3dd18e7713327a` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `9aa5b9a332b19eb564fc6cfe` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `e813ad2611ae45a8bb74bf85` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `dff5a7ad6ad563e6d01a664e` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `774c3456c68c51879cfbaf7b` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `cedbe474e6e6fb1bf92a3f2b` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `9957d63a15281cc63218534b` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `59ae1de15428404d09274f09` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `4dd44b73197390fd93d6e30a` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `41efdb36750197cdd289bd80` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |
| `7312b745ee18711b3a2b9c9e` | dict(keys=['decision', 'stage', 'llm_reason', 'tokens']) |

### `reports/false_refusal_diag_2026-08-23.csv`

- 크기 3.8KB / CSV, 총 16행 / 수정 2026-08-23 03:15
- **RAGAS 적합도: 부분** — 누락: answer
- 역할 추정:
  - question: `﻿question`
  - refusal: `refusal_reason`
  - contexts: `n_evidence`
- 키 10개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿question` | 하나은행 디폴트옵션 안정투자형 포트폴리오 1 상품이 위험등급이 바뀌거나 승인이 취소되면 어떻게 알려주나요? |
| `stage` | retrieval |
| `difficulty` | medium |
| `doc_type` | 설명서 |
| `refusal_reason` | low_confidence |
| `llm_reason` |  |
| `n_evidence` | 3 |
| `top1` | 0.5917 |
| `top_citation` | 『하나은행 디폴트옵션 안정투자형 포트폴리오 1』 상품설명서 주 (2026-04-01 시행) |
| `분류` |  |

### `reports/mismatch_review_2026-08-23.csv`

- 크기 8.1KB / CSV, 총 20행 / 수정 2026-08-23 03:10
- **RAGAS 적합도: 부분** — 누락: contexts, answer
- 역할 추정:
  - question: `question`
- 키 9개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿id` | 01 |
| `question` | 하나은행 이자율 스왑 상품의 위험등급과 손실범위는 어떻게 되나요? |
| `difficulty` | easy |
| `doc_type` | 설명서 |
| `extracted_product` |  |
| `why` | 질문 상품 '이자율 스왑 상품' 과 일치하는 근거 문서 없음 |
| `class` | D |
| `is_real` | False |
| `memo` | 근거 문서가 이자율 스왑 설명서로 질문에 직접 답한다. mismatch가 아니라 Query Analyzer가 질문의 상품명/서비스명을 추출하지  … |

### `reports/g_eval_v2_2026-08-23.csv`

- 크기 9.2KB / CSV, 총 25행 / 수정 2026-08-23 02:51
- **RAGAS 적합도: 부분** — 누락: contexts, answer
- 역할 추정:
  - question: `question`
  - refusal: `false_refusal`
  - verdict: `v2_환각`, `v2_수치정확`, `why_환각`, `why_수치정확`
- 키 19개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿id` | 01 |
| `group` | 정상 답변 |
| `question` | 하나은행 내 계좌이체 시 보안매체나 공동인증서 사용 여부는? |
| `decision` | refuse |
| `stage` | pattern |
| `false_refusal` | True |
| `extracted_product` |  |
| `unsupported` | 0 |
| `n_sentences` | 0 |
| `v2_환각` | na |
| `v2_상품일치` | na |
| `v2_수치정확` | na |
| `v2_미확인신고` | manual |
| `v2_실무활용` | manual |
| `why_환각` | 거절 응답 — 답변 본문 없음 |
| `why_상품일치` | 거절 응답 — 답변 본문 없음 |
| `why_수치정확` | 거절 응답 — 답변 본문 없음 |
| `why_미확인신고` | 거절 응답 — 답변 본문 없음 |
| `why_실무활용` | 거절 응답 — 답변 본문 없음 |

### `reports/g_eval_v2_sentences_2026-08-23.csv`

- 크기 8.2KB / CSV, 총 37행 / 수정 2026-08-23 02:51
- **RAGAS 적합도: 부분** — 누락: question, answer
- 역할 추정:
  - contexts: `evidence`
- 키 6개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿id` | 02 |
| `sentence` | 특정 제휴사 해외 ATM 출금 서비스에서는 현지통화 출금금액을 먼저 실시간 ATM 출금 서비스용 매입율을 적용해 USD로 환전하고, 이후 USD … |
| `supported` | True |
| `evidence` | [1, 2] |
| `why` | 문장 내용이 근거 1,2에 명확히 있음 |
| `error` |  |

### `reports/g_eval_labels_2026-08-21.csv`

- 크기 7.3KB / CSV, 총 25행 / 수정 2026-08-21 21:59
- **RAGAS 적합도: 부분** — 누락: contexts, answer
- 역할 추정:
  - question: `question`
  - verdict: `human_환각`, `draft_환각`, `human_수치정확`, `draft_수치정확`
- 키 17개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿id` | 01 |
| `group` | 정상 답변 |
| `question` | 하나은행 내 계좌이체 시 보안매체나 공동인증서 사용 여부는? |
| `decision` | refuse |
| `stage` | pattern |
| `blind` | False |
| `memo` | 일반 계좌이체 보안수단 질문을 개인화 문의로 오거절. 개인정보 조회 필요 판단은 부적절. |
| `human_환각` | na |
| `draft_환각` | pass |
| `human_상품일치` | na |
| `draft_상품일치` | na |
| `human_수치정확` | na |
| `draft_수치정확` | na |
| `human_미확인신고` | fail |
| `draft_미확인신고` | fail |
| `human_실무활용` | fail |
| `draft_실무활용` | fail |

### `reports/judge_calibration_2026-08-21.csv`

- 크기 307B / CSV, 총 5행 / 수정 2026-08-21 21:59
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 9개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿criterion` | 환각 |
| `n` | 20 |
| `agree` | 11 |
| `agreement` | 0.55 |
| `kappa` | 0.0 |
| `missed_fail` | 3 |
| `over_flag` | 0 |
| `missed_ids` | 03/11/23 |
| `over_ids` |  |

### `data/interim/g_eval_answers.json`

- 크기 98.4KB / JSON 배열, 총 25건 / 수정 2026-08-15 18:27
- **RAGAS 적합도: 적합** — 질문·근거·답변 3필드 모두 확인
- 역할 추정:
  - question: `question`
  - answer: `answer`
  - refusal: `refusal_reason`
  - contexts: `evidences`
- 키 16개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `id` | 01 |
| `group` | 정상 답변 |
| `difficulty` | easy |
| `doc_type` | FAQ |
| `note` |  |
| `question` | 하나은행 내 계좌이체 시 보안매체나 공동인증서 사용 여부는? |
| `decision` | refuse |
| `answer` | 고객님의 개인 정보가 필요한 문의입니다. 고객 정보를 조회한 뒤 안내해 주시기 바랍니다. |
| `not_found` | list[0] 첫원소:  |
| `refusal_reason` | personalized |
| `stage` | pattern |
| `tokens` | 0 |
| `latency_ms` | 11836 |
| `evidences` | list[0] 첫원소:  |
| `blind` | False |
| `draft` | dict(keys=['환각', '상품일치', '수치정확', '미확인신고', '실무활용', '근거']) |

### `reports/coverage_measure_2026-08-15.csv`

- 크기 66.6KB / CSV, 총 319행 / 수정 2026-08-15 18:16
- **RAGAS 적합도: 부분** — 누락: answer
- 역할 추정:
  - question: `﻿question`
  - refusal: `gate_refuse`, `refusal_type`
  - contexts: `source`
- 키 16개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿question` | 제 연체 기록이 대출 연체 이자에 어떻게 영향을 미치나요? |
| `extracted_product` |  |
| `granularity` | none |
| `intent` | general_policy |
| `needs_specific` | False |
| `general_ok` | True |
| `confidence` | 0.9 |
| `match_status` | none |
| `matched_product` |  |
| `candidates` |  |
| `gate_refuse` | False |
| `gate_reason` |  |
| `error` |  |
| `source` | refusal_eval.csv |
| `expected` | refuse |
| `refusal_type` | personalized |

### `data/interim/product_catalog.json`

- 크기 33.1KB / JSON dict → 'covered_products' 배열, 총 36건 / 수정 2026-08-15 18:14
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 7개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `canonical_name` | 청년 주택드림 청약통장 |
| `doc_id` | hana_desc_deposit_002 |
| `doc_title` | 『청년 주택드림 청약통장』 요약 상품설명서 |
| `doc_type` | 설명서 |
| `category` | deposit |
| `granularity` | specific_product |
| `effective_date` | 2026-06-30 |

### `reports/refusal_eval_2026-08-15.csv`

- 크기 7.8KB / CSV, 총 53행 / 수정 2026-08-15 16:50
- **RAGAS 적합도: 부분** — 누락: contexts, answer
- 역할 추정:
  - question: `﻿query_id`, `question`
  - refusal: `refusal_type`
  - verdict: `score_top1`
- 키 11개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿query_id` | r001 |
| `question` | 제 연체 기록이 대출 연체 이자에 어떻게 영향을 미치나요? |
| `expected` | refuse |
| `refusal_type` | personalized |
| `predicted` | refuse |
| `stage` | pattern |
| `reason` | personalized |
| `score_top1` | 0.6025 |
| `gap_1_5` | 0.0399 |
| `blank_ratio` | 0.0 |
| `correct` | True |

### `reports/refusal_eval_2026-08-15.json`

- 크기 1.6KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-15 16:50
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 6개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `date` | 2026-08-15 |
| `eval_set` | dict(keys=['total', 'refuse', 'answer']) |
| `retrieval` | dict(keys=['model', 'alpha']) |
| `thresholds` | dict(keys=['hard_refuse_top1', 'hard_refuse_gap15', 'blank_ratio']) |
| `stages` | dict(keys=['0단계만 (질문 패턴)', '0+1단계 (패턴+검색)', '0+1+2단계 (LLM 포함)']) |
| `by_type` | dict(keys=['personalized', 'time_variant', 'blank_value', 'out_of_scope']) |

### `reports/refusal_tuning_v2_full_2026-08-15.csv`

- 크기 40.9KB / CSV, 총 504행 / 수정 2026-08-15 16:35
- **RAGAS 적합도: 부분** — 누락: question, contexts
- 역할 추정:
  - refusal: `refusal_accuracy`, `over_refusal_rate`
  - answer: `false_answer_rate`, `answer_accuracy`
- 키 17개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿hard_top1` | 0.75 |
| `hard_gap15` | 0.08 |
| `blank_ratio` | 0.05 |
| `to_llm` | 22 |
| `tp` | 32 |
| `fn` | 1 |
| `fp` | 2 |
| `tn` | 18 |
| `refusal_accuracy` | 0.9697 |
| `false_answer_rate` | 0.0303 |
| `over_refusal_rate` | 0.1 |
| `answer_accuracy` | 0.9 |
| `precision` | 0.9412 |
| `f1` | 0.9552 |
| `balanced_accuracy` | 0.9348 |
| `n` | 53 |
| `cost` | 0.20211 |

### `data/interim/refusal_llm_verdicts_e5-small_a05_gpt-41-mini.json`

- 크기 7.5KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-15 16:35
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 53개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `r001` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r002` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r003` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r005` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r006` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r008` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r009` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r010` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r012` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r013` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r015` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r016` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r017` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r018` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r020` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r021` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r022` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r023` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r024` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r026` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r027` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r028` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r029` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r030` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r031` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r032` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r033` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r034` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r035` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r036` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r038` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r039` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `r040` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c001` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c002` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c003` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c004` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c005` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c006` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c007` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c008` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c009` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c010` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c011` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c012` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c013` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c014` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c015` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c016` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c017` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c018` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c019` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |
| `c020` | dict(keys=['verdict', 'stage', 'llm_reason', 'tokens']) |

### `reports/refusal_tuning_2026-08-15.csv`

- 크기 7.8KB / CSV, 총 96행 / 수정 2026-08-15 10:55
- **RAGAS 적합도: 부분** — 누락: question, contexts
- 역할 추정:
  - refusal: `refusal_accuracy`, `over_refusal_rate`
  - answer: `false_answer_rate`, `answer_accuracy`
- 키 17개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿hard_top1` | 0.75 |
| `hard_gap15` | 0.08 |
| `blank_ratio` | 0.2 |
| `to_llm` | 22 |
| `tp` | 30 |
| `fn` | 3 |
| `fp` | 1 |
| `tn` | 19 |
| `refusal_accuracy` | 0.9091 |
| `false_answer_rate` | 0.0909 |
| `over_refusal_rate` | 0.05 |
| `answer_accuracy` | 0.95 |
| `precision` | 0.9677 |
| `f1` | 0.9375 |
| `balanced_accuracy` | 0.9295 |
| `n` | 53 |
| `cost` | 0.2318 |

### `reports/refusal_tuning_2026-08-14.csv`

- 크기 26.9KB / CSV, 총 320행 / 수정 2026-08-14 20:42
- **RAGAS 적합도: 부분** — 누락: question, contexts
- 역할 추정:
  - refusal: `﻿refuse_top1`, `refuse_gap`, `refusal_accuracy`, `over_refusal_rate`
  - answer: `answer_top1`, `answer_gap`, `false_answer_rate`, `answer_accuracy`
- 키 18개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿refuse_top1` | 0.84 |
| `refuse_gap` | 0.18 |
| `answer_top1` | 0.88 |
| `answer_gap` | 0.3 |
| `uncertain` | 14 |
| `tp` | 25 |
| `fn` | 8 |
| `fp` | 3 |
| `tn` | 17 |
| `refusal_accuracy` | 0.7576 |
| `false_answer_rate` | 0.2424 |
| `over_refusal_rate` | 0.15 |
| `answer_accuracy` | 0.85 |
| `precision` | 0.8929 |
| `f1` | 0.8197 |
| `balanced_accuracy` | 0.8038 |
| `n` | 53 |
| `cost` | 0.6348 |

### `reports/refusal_eval_2026-08-14.csv`

- 크기 7.8KB / CSV, 총 53행 / 수정 2026-08-14 20:41
- **RAGAS 적합도: 부분** — 누락: contexts, answer
- 역할 추정:
  - question: `﻿query_id`, `question`
  - refusal: `refusal_type`
  - verdict: `score_top1`
- 키 11개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿query_id` | r001 |
| `question` | 제 연체 기록이 대출 연체 이자에 어떻게 영향을 미치나요? |
| `expected` | refuse |
| `refusal_type` | personalized |
| `predicted` | refuse |
| `stage` | pattern |
| `reason` | personalized |
| `score_top1` | 0.6025 |
| `gap_1_5` | 0.0399 |
| `blank_ratio` | 0.0 |
| `correct` | True |

### `reports/refusal_eval_2026-08-14.json`

- 크기 1.7KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-14 20:41
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 6개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `date` | 2026-08-14 |
| `eval_set` | dict(keys=['total', 'refuse', 'answer']) |
| `retrieval` | dict(keys=['model', 'alpha']) |
| `thresholds` | dict(keys=['refuse', 'answer']) |
| `stages` | dict(keys=['0단계만 (질문 패턴)', '0+1단계 (패턴+검색)', '0+1+2단계 (LLM 포함)']) |
| `by_type` | dict(keys=['personalized', 'time_variant', 'blank_value', 'out_of_scope']) |

### `reports/refusal_signals_2026-08-14.csv`

- 크기 11.5KB / CSV, 총 53행 / 수정 2026-08-14 20:29
- **RAGAS 적합도: 부분** — 누락: answer
- 역할 추정:
  - question: `﻿query_id`, `question`
  - refusal: `refusal_type`
  - verdict: `score_top1`
  - contexts: `top1_chunk`
- 키 12개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿query_id` | r001 |
| `question` | 제 연체 기록이 대출 연체 이자에 어떻게 영향을 미치나요? |
| `expected` | refuse |
| `refusal_type` | personalized |
| `score_top1` | 0.6025 |
| `gap_1_2` | 0.0 |
| `gap_1_5` | 0.0399 |
| `std_top5` | 0.0161 |
| `blank_ratio` | 0.0 |
| `doc_concentration` | 3 |
| `top1_doc` | 가계대출 상품설명서 |
| `top1_chunk` | hana_desc_loan_006_c043 |

### `data/eval/refusal_eval.csv`

- 크기 10.3KB / CSV, 총 53행 / 수정 2026-08-14 20:29
- **RAGAS 적합도: 적합** — 질문·근거·답변 3필드 모두 확인
- 역할 추정:
  - question: `﻿query_id`, `question`
  - refusal: `refusal_type`
  - answer: `why_unanswerable`
  - contexts: `source_chunk_id`
- 키 9개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿query_id` | r001 |
| `question` | 제 연체 기록이 대출 연체 이자에 어떻게 영향을 미치나요? |
| `expected` | refuse |
| `refusal_type` | personalized |
| `why_unanswerable` | 개인 연체 기록 필요 |
| `source_chunk_id` | hana_desc_loan_006_c004 |
| `doc_display_name` | 가계대출 상품설명서 |
| `doc_type` | 설명서 |
| `category` | loan |

### `data/eval/refusal_eval_draft.csv`

- 크기 14.2KB / CSV, 총 60행 / 수정 2026-08-14 20:26
- **RAGAS 적합도: 적합** — 질문·근거·답변 3필드 모두 확인
- 역할 추정:
  - question: `﻿query_id`, `question`
  - refusal: `refusal_type`
  - answer: `why_unanswerable`
  - contexts: `source_chunk_id`
- 키 13개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿query_id` | r001 |
| `question` | 제 연체 기록이 대출 연체 이자에 어떻게 영향을 미치나요? |
| `expected` | refuse |
| `refusal_type` | personalized |
| `why_unanswerable` | 개인 연체 기록 필요 |
| `source_chunk_id` | hana_desc_loan_006_c004 |
| `doc_display_name` | 가계대출 상품설명서 |
| `doc_type` | 설명서 |
| `category` | loan |
| `verified` | Y |
| `keep` | Y |
| `review_note` |  |
| `reviewed` | Y |

### `reports/hybrid_tuning_final_doclevel_2026-08-14.json`

- 크기 25.9KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-14 20:06
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 8개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `date` | 2026-08-14 |
| `model` | e5-small |
| `index` | e5-small |
| `bm25_index` | default |
| `eval_set` | retrieval_eval.csv |
| `level` | doc |
| `candidate_k` | 100 |
| `results` | dict(keys=['dense 단독', 'sparse 단독', 'weighted α=0.3', 'weighted α=0.5', 'weighted α=0.7', 'weighted-z α=0.3']) |

### `data/indexes/faiss/e5-small/chunks_meta.jsonl`

- 크기 4.4MB / JSONL, 총 2999줄 / 수정 2026-08-14 19:59
- **RAGAS 적합도: 부분** — 누락: question, answer
- 역할 추정:
  - contexts: `chunk_id`, `source_url`
- 키 17개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `chunk_id` | hana_desc_deposit_002_c001 |
| `doc_id` | hana_desc_deposit_002 |
| `text` | 준법감시인심의필제2026-설명서-082호(2026.06.23) 『청년 주택드림 청약통장』 요약 상품설명서 [기본정보] (2026.06.30 현재 … |
| `order` | 1 |
| `section` |  |
| `page_start` | 0 |
| `page_end` | 0 |
| `doc_display_name` | 『청년 주택드림 청약통장』 요약 상품설명서 |
| `doc_type` | 설명서 |
| `category` | deposit |
| `effective_date` | 2026-06-30 |
| `source_url` |  |
| `char_count` | 345 |
| `metadata` | dict(keys=['structure', 'compliance_no', 'is_latest']) |
| `citation` | 『청년 주택드림 청약통장』 요약 상품설명서 (2026-06-30 시행) |
| `tokens_e5-small` | 253 |
| `tokens_bge-m3` | 253 |

### `data/indexes/faiss/e5-small/config.json`

- 크기 123B / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-14 19:59
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 5개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `model_key` | e5-small |
| `dim` | 384 |
| `n_vectors` | 2999 |
| `index_type` | IndexFlatIP |
| `normalized` | True |

### `data/interim/chunks.jsonl`

- 크기 4.4MB / JSONL, 총 2999줄 / 수정 2026-08-14 19:57
- **RAGAS 적합도: 부분** — 누락: question, answer
- 역할 추정:
  - contexts: `chunk_id`, `source_url`
- 키 17개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `chunk_id` | hana_desc_deposit_002_c001 |
| `doc_id` | hana_desc_deposit_002 |
| `text` | 준법감시인심의필제2026-설명서-082호(2026.06.23) 『청년 주택드림 청약통장』 요약 상품설명서 [기본정보] (2026.06.30 현재 … |
| `order` | 1 |
| `section` |  |
| `page_start` | 0 |
| `page_end` | 0 |
| `doc_display_name` | 『청년 주택드림 청약통장』 요약 상품설명서 |
| `doc_type` | 설명서 |
| `category` | deposit |
| `effective_date` | 2026-06-30 |
| `source_url` |  |
| `char_count` | 345 |
| `metadata` | dict(keys=['structure', 'compliance_no', 'is_latest']) |
| `citation` | 『청년 주택드림 청약통장』 요약 상품설명서 (2026-06-30 시행) |
| `tokens_e5-small` | 253 |
| `tokens_bge-m3` | 253 |

### `data/interim/chunk_stats.csv`

- 크기 33.0KB / CSV, 총 304행 / 수정 2026-08-14 19:57
- **RAGAS 적합도: 부분** — 누락: question, answer
- 역할 추정:
  - contexts: `n_chunks`, `avg_chunk_chars`, `max_chunk_chars`
- 키 8개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿doc_id` | hana_desc_deposit_002 |
| `doc_type` | 설명서 |
| `structure` | article |
| `title` | 『청년 주택드림 청약통장』 요약 상품설명서 |
| `doc_chars` | 14150 |
| `n_chunks` | 39 |
| `avg_chunk_chars` | 381 |
| `max_chunk_chars` | 895 |

### `reports/hybrid_tuning_baseline_doclevel_2026-08-14.json`

- 크기 25.9KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-14 19:46
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 8개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `date` | 2026-08-14 |
| `model` | e5-small |
| `index` | e5-small |
| `bm25_index` | default |
| `eval_set` | retrieval_eval.csv |
| `level` | doc |
| `candidate_k` | 100 |
| `results` | dict(keys=['dense 단독', 'sparse 단독', 'weighted α=0.3', 'weighted α=0.5', 'weighted α=0.7', 'weighted-z α=0.3']) |

### `reports/hybrid_tuning_structural_doclevel_2026-08-14.json`

- 크기 25.9KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-14 19:44
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 8개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `date` | 2026-08-14 |
| `model` | e5-small |
| `index` | e5-small_structural |
| `bm25_index` | structural |
| `eval_set` | retrieval_eval_structural.csv |
| `level` | doc |
| `candidate_k` | 100 |
| `results` | dict(keys=['dense 단독', 'sparse 단독', 'weighted α=0.3', 'weighted α=0.5', 'weighted α=0.7', 'weighted-z α=0.3']) |

### `reports/eval_baseline_doclevel_2026-08-14.json`

- 크기 6.7KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-14 19:41
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 13개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `date` | 2026-08-14 |
| `tag` | baseline |
| `model` | e5-small |
| `index` | e5-small |
| `eval_set` | retrieval_eval.csv |
| `level` | doc |
| `retrieval` | dense |
| `n_queries` | 96 |
| `overall` | dict(keys=['recall@1', 'recall@3', 'recall@5', 'recall@10', 'mrr', 'ndcg@10']) |
| `by_difficulty` | dict(keys=['easy', 'hard', 'medium']) |
| `by_doc_type` | dict(keys=['FAQ', '설명서', '약관']) |
| `by_category` | dict(keys=['auth_otp_security_card', 'bill_payment', 'cd_atm', 'certificate', 'deposit', 'deposit_trust']) |
| `failure_analysis` | dict(keys=['top5', 'rank_6_20', 'beyond_20']) |

### `reports/failures_baseline_doclevel_2026-08-14.csv`

- 크기 2.5KB / CSV, 총 9행 / 수정 2026-08-14 19:41
- **RAGAS 적합도: 부분** — 누락: answer
- 역할 추정:
  - question: `﻿query_id`, `question`
  - contexts: `relevant_chunk_ids`, `top5_retrieved`
  - verdict: `top1_score`, `score_gap`
- 키 11개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿query_id` | q002 |
| `question` | 하나은행 입출금 예금 이율은 바뀌면 언제부터 적용돼요? |
| `difficulty` | medium |
| `doc_type` | 약관 |
| `category` | deposit |
| `hit_rank` | 미발견 |
| `failure_type` | 20위 밖 |
| `relevant_chunk_ids` | hana_terms_deposit_002 |
| `top5_retrieved` | hana_terms_loan_011/hana_desc_deposit_006/hana_terms_loan_017/hana_terms_loan_00 … |
| `top1_score` | 0.8981 |
| `score_gap` | 0.0069 |

### `reports/hybrid_tuning_structural_v2_2026-08-14.json`

- 크기 26.0KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-14 19:38
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 7개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `date` | 2026-08-14 |
| `model` | e5-small |
| `index` | e5-small_structural |
| `bm25_index` | structural |
| `eval_set` | retrieval_eval_structural.csv |
| `candidate_k` | 100 |
| `results` | dict(keys=['dense 단독', 'sparse 단독', 'weighted α=0.3', 'weighted α=0.5', 'weighted α=0.7', 'weighted-z α=0.3']) |

### `reports/failures_structural_v2_2026-08-14.csv`

- 크기 7.7KB / CSV, 총 21행 / 수정 2026-08-14 19:36
- **RAGAS 적합도: 부분** — 누락: answer
- 역할 추정:
  - question: `﻿query_id`, `question`
  - contexts: `relevant_chunk_ids`, `top5_retrieved`
  - verdict: `top1_score`, `score_gap`
- 키 11개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `﻿query_id` | q002 |
| `question` | 하나은행 입출금 예금 이율은 바뀌면 언제부터 적용돼요? |
| `difficulty` | medium |
| `doc_type` | 약관 |
| `category` | deposit |
| `hit_rank` | 18 |
| `failure_type` | 순위 낮음 |
| `relevant_chunk_ids` | hana_terms_deposit_002_c009/hana_terms_deposit_002_c010 |
| `top5_retrieved` | hana_terms_loan_011_c057/hana_desc_deposit_006_c011/hana_desc_retirement_pension … |
| `top1_score` | 0.8929 |
| `score_gap` | 0.0018 |

### `reports/eval_structural_v2_2026-08-14.json`

- 크기 6.7KB / JSON 단일 dict (요약 파일로 추정) / 수정 2026-08-14 19:36
- **RAGAS 적합도: 부적합** — 3필드 모두 없음
- 키 12개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `date` | 2026-08-14 |
| `tag` | structural_v2 |
| `model` | e5-small |
| `index` | e5-small_structural |
| `eval_set` | retrieval_eval_structural.csv |
| `retrieval` | dense |
| `n_queries` | 96 |
| `overall` | dict(keys=['recall@1', 'recall@3', 'recall@5', 'recall@10', 'mrr', 'ndcg@10']) |
| `by_difficulty` | dict(keys=['easy', 'hard', 'medium']) |
| `by_doc_type` | dict(keys=['FAQ', '설명서', '약관']) |
| `by_category` | dict(keys=['auth_otp_security_card', 'bill_payment', 'cd_atm', 'certificate', 'deposit', 'deposit_trust']) |
| `failure_analysis` | dict(keys=['top5', 'rank_6_20', 'beyond_20']) |

### `data/eval/remap_log.json`

- 크기 31.4KB / JSON dict → 'mappings' 배열, 총 96건 / 수정 2026-08-14 19:36
- **RAGAS 적합도: 부분** — 누락: contexts, answer
- 역할 추정:
  - question: `query_id`
- 키 4개와 첫 레코드 값(잘림):

| 키 | 값 미리보기 |
|---|---|
| `query_id` | q001 |
| `old` | list[1] 첫원소: hana_faq_016_c001 |
| `new` | list[1] 첫원소: hana_faq_016_c001 |
| `sims` | list[1] 첫원소: 1.0 |

## 3. 요약 — RAGAS 입력 후보

| 파일 | 적합도 | 사유 | 형식 |
|---|---|---|---|
| `data/interim/g_eval_answers.json` | 적합 | 질문·근거·답변 3필드 모두 확인 | JSON 배열, 총 25건 |
| `data/eval/refusal_eval.csv` | 적합 | 질문·근거·답변 3필드 모두 확인 | CSV, 총 53행 |
| `data/eval/refusal_eval_draft.csv` | 적합 | 질문·근거·답변 3필드 모두 확인 | CSV, 총 60행 |
| `reports/g_eval_full_2026-08-23.csv` | 부분 | 누락: answer | CSV, 총 96행 |
| `reports/refusal_eval_2026-08-23.csv` | 부분 | 누락: contexts, answer | CSV, 총 53행 |
| `reports/false_refusal_diag_2026-08-23.csv` | 부분 | 누락: answer | CSV, 총 16행 |
| `reports/mismatch_review_2026-08-23.csv` | 부분 | 누락: contexts, answer | CSV, 총 20행 |
| `reports/g_eval_v2_2026-08-23.csv` | 부분 | 누락: contexts, answer | CSV, 총 25행 |
| `reports/g_eval_v2_sentences_2026-08-23.csv` | 부분 | 누락: question, answer | CSV, 총 37행 |
| `reports/g_eval_labels_2026-08-21.csv` | 부분 | 누락: contexts, answer | CSV, 총 25행 |
| `reports/coverage_measure_2026-08-15.csv` | 부분 | 누락: answer | CSV, 총 319행 |
| `reports/refusal_eval_2026-08-15.csv` | 부분 | 누락: contexts, answer | CSV, 총 53행 |
| `reports/refusal_tuning_v2_full_2026-08-15.csv` | 부분 | 누락: question, contexts | CSV, 총 504행 |
| `reports/refusal_tuning_2026-08-15.csv` | 부분 | 누락: question, contexts | CSV, 총 96행 |
| `reports/refusal_tuning_2026-08-14.csv` | 부분 | 누락: question, contexts | CSV, 총 320행 |
| `reports/refusal_eval_2026-08-14.csv` | 부분 | 누락: contexts, answer | CSV, 총 53행 |
| `reports/refusal_signals_2026-08-14.csv` | 부분 | 누락: answer | CSV, 총 53행 |
| `data/indexes/faiss/e5-small/chunks_meta.jsonl` | 부분 | 누락: question, answer | JSONL, 총 2999줄 |
| `data/interim/chunks.jsonl` | 부분 | 누락: question, answer | JSONL, 총 2999줄 |
| `data/interim/chunk_stats.csv` | 부분 | 누락: question, answer | CSV, 총 304행 |
| `reports/failures_baseline_doclevel_2026-08-14.csv` | 부분 | 누락: answer | CSV, 총 9행 |
| `reports/failures_structural_v2_2026-08-14.csv` | 부분 | 누락: answer | CSV, 총 21행 |
| `data/eval/remap_log.json` | 부분 | 누락: contexts, answer | JSON dict → 'mappings' 배열, 총 96건 |
| `data/interim/query_analysis_cache.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `data/interim/g_judge_v2_cache.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `data/interim/g_full_answers.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `reports/refusal_eval_2026-08-23.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `data/interim/refusal_verify_cache_e5-small_a05_gpt-41-mini.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `reports/judge_calibration_2026-08-21.csv` | 부적합 | 3필드 모두 없음 | CSV, 총 5행 |
| `data/interim/product_catalog.json` | 부적합 | 3필드 모두 없음 | JSON dict → 'covered_products' 배열, 총 36건 |
| `reports/refusal_eval_2026-08-15.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `data/interim/refusal_llm_verdicts_e5-small_a05_gpt-41-mini.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `reports/refusal_eval_2026-08-14.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `reports/hybrid_tuning_final_doclevel_2026-08-14.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `data/indexes/faiss/e5-small/config.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `reports/hybrid_tuning_baseline_doclevel_2026-08-14.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `reports/hybrid_tuning_structural_doclevel_2026-08-14.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `reports/eval_baseline_doclevel_2026-08-14.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `reports/hybrid_tuning_structural_v2_2026-08-14.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |
| `reports/eval_structural_v2_2026-08-14.json` | 부적합 | 3필드 모두 없음 | JSON 단일 dict (요약 파일로 추정) |