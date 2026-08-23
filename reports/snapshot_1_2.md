## 1. 디렉터리 트리

```
financial_rag_project/
├── _backup/
│   ├── refusal.py.bak  [18.8KB]
│   ├── refusal_pre_docfix.py.bak  [19.3KB]
│   └── refusal_v2.py.bak  [24.1KB]
├── app/
├── configs/
│   └── banks/
├── data/ (파일 167개, 내용 생략)
├── docs/
├── notebooks/
├── outputs/ (파일 0개, 내용 생략)
├── reports/
│   ├── coverage_measure_2026-08-15.csv  [66.6KB]
│   ├── eval_baseline_dense_2026-08-13.json  [6.7KB]
│   ├── eval_baseline_dense_2026-08-13.md  [2.5KB]
│   ├── eval_baseline_doclevel_2026-08-14.json  [6.7KB]
│   ├── eval_baseline_doclevel_2026-08-14.md  [2.5KB]
│   ├── eval_set_summary.md  [2.0KB]
│   ├── eval_structural_2026-08-14.json  [6.7KB]
│   ├── eval_structural_2026-08-14.md  [2.5KB]
│   ├── eval_structural_v2_2026-08-14.json  [6.7KB]
│   ├── eval_structural_v2_2026-08-14.md  [2.5KB]
│   ├── failures_baseline_dense_2026-08-13.csv  [7.6KB]
│   ├── failures_baseline_doclevel_2026-08-14.csv  [2.5KB]
│   ├── failures_structural_2026-08-14.csv  [9.3KB]
│   ├── failures_structural_v2_2026-08-14.csv  [7.7KB]
│   ├── false_refusal_diag_2026-08-23.csv  [3.8KB]
│   ├── g_eval_full_2026-08-23.csv  [19.7KB]
│   ├── g_eval_labels_2026-08-21.csv  [7.3KB]
│   ├── g_eval_v2_2026-08-23.csv  [9.2KB]
│   ├── g_eval_v2_sentences_2026-08-23.csv  [8.2KB]
│   ├── hybrid_tuning_2026-08-13.json  [18.9KB]
│   ├── hybrid_tuning_2026-08-13.md  [3.3KB]
│   ├── hybrid_tuning_baseline_doclevel_2026-08-14.json  [25.9KB]
│   ├── hybrid_tuning_baseline_doclevel_2026-08-14.md  [3.9KB]
│   ├── hybrid_tuning_final_doclevel_2026-08-14.json  [25.9KB]
│   ├── hybrid_tuning_final_doclevel_2026-08-14.md  [3.9KB]
│   ├── hybrid_tuning_structural_2026-08-14.json  [26.0KB]
│   ├── hybrid_tuning_structural_2026-08-14.md  [3.9KB]
│   ├── hybrid_tuning_structural_doclevel_2026-08-14.json  [25.9KB]
│   ├── hybrid_tuning_structural_doclevel_2026-08-14.md  [3.9KB]
│   ├── hybrid_tuning_structural_v2_2026-08-14.json  [26.0KB]
│   ├── hybrid_tuning_structural_v2_2026-08-14.md  [3.9KB]
│   ├── judge_calibration_2026-08-21.csv  [307B]
│   ├── mismatch_review_2026-08-23.csv  [8.1KB]
│   ├── refusal_eval_2026-08-14.csv  [7.8KB]
│   ├── refusal_eval_2026-08-14.json  [1.7KB]
│   ├── refusal_eval_2026-08-15.csv  [7.8KB]
│   ├── refusal_eval_2026-08-15.json  [1.6KB]
│   ├── refusal_eval_2026-08-23.csv  [7.8KB]
│   ├── refusal_eval_2026-08-23.json  [1.6KB]
│   ├── refusal_signals_2026-08-14.csv  [11.5KB]
│   └── … (항목 수 제한으로 일부 생략)
├── scripts/
│   ├── 01_crawl_hana_faq.py  [6.4KB]
│   ├── 02_validate_faq.py  [7.2KB]
│   ├── 03_build_registry.py  [5.9KB]
│   ├── 04_inspect_pdfs.py  [7.6KB]
│   ├── 05_triage_pdfs.py  [12.2KB]
│   ├── 06_build_documents.py  [15.1KB]
│   ├── 07_inspect_issues.py  [3.6KB]
│   ├── 08_benchmark_embedding.py  [6.4KB]
│   ├── 09_build_chunks.py  [12.2KB]
│   ├── 10_inspect_chunks.py  [4.2KB]
│   ├── 11_build_index.py  [7.1KB]
│   ├── 12_search.py  [4.6KB]
│   ├── 13_audit_titles.py  [8.7KB]
│   ├── 14_generate_eval.py  [32.7KB]
│   ├── 15_review_eval.py  [10.1KB]
│   ├── 16_finalize_eval.py  [16.6KB]
│   ├── 17_evaluate.py  [15.8KB]
│   ├── 18_analyze_failures.py  [8.0KB]
│   ├── 19_build_bm25.py  [6.7KB]
│   ├── 20_tune_hybrid.py  [16.4KB]
│   ├── 21_analyze_vocab.py  [7.8KB]
│   ├── 22_remap_eval.py  [12.1KB]
│   ├── 23_list_indexes.py  [4.9KB]
│   ├── 24_generate_refusal_set.py  [16.4KB]
│   ├── 25_review_refusal.py  [9.4KB]
│   ├── 26_analyze_refusal_signals.py  [15.1KB]
│   ├── 27_evaluate_refusal.py  [22.0KB]
│   ├── 28_tune_refusal_v2.py  [26.0KB]
│   ├── 29_diagnose_spacing.py  [9.1KB]
│   ├── 30_answer_cli.py  [4.4KB]
│   ├── 31_measure_coverage.py  [10.0KB]
│   ├── 32_build_g_evalset.py  [18.0KB]
│   ├── 33_calibrate_judge.py  [12.0KB]
│   ├── 34_evaluate_g_v2.py  [27.2KB]
│   ├── 35_apply_review.py  [8.7KB]
│   ├── 36_evaluate_g_full.py  [18.1KB]
│   ├── 37_readme_snapshot.py  [13.9KB]
│   ├── 37_review_mismatch.py  [10.6KB]
│   ├── 38_diagnose_refusal.py  [10.0KB]
│   ├── 39_check_patterns.py  [10.9KB]
│   └── … (항목 수 제한으로 일부 생략)
├── src/
│   └── finguide_rag/
│       ├── chunking/
│       │   ├── strategies/
│       │   │   └── … (깊이 제한)
│       │   ├── __init__.py  [493B]
│       │   ├── base.py  [5.9KB]
│       │   ├── factory.py  [2.4KB]
│       │   ├── flat_chunker.py  [3.9KB]
│       │   └── structural_chunker.py  [10.9KB]
│       ├── embedding/
│       │   ├── __init__.py  [270B]
│       │   ├── embedder.py  [4.9KB]
│       │   └── store.py  [7.0KB]
│       ├── evaluation/
│       │   ├── __init__.py  [462B]
│       │   └── metrics.py  [6.9KB]
│       ├── generation/
│       │   ├── prompt_templates/
│       │   │   └── … (깊이 제한)
│       │   ├── __init__.py  [633B]
│       │   ├── catalog_matcher.py  [6.4KB]
│       │   ├── generator.py  [14.5KB]
│       │   ├── pipeline.py  [5.7KB]
│       │   ├── product_catalog.py  [11.4KB]
│       │   ├── product_match.py  [7.7KB]
│       │   ├── query_analyzer.py  [7.8KB]
│       │   └── refusal.py  [29.4KB]
│       ├── ingestion/
│       │   ├── crawlers/
│       │   │   └── … (깊이 제한)
│       │   └── parsers/
│       │       └── … (깊이 제한)
│       ├── parsing/
│       │   ├── __init__.py  [361B]
│       │   ├── base.py  [1.6KB]
│       │   ├── factory.py  [1.6KB]
│       │   ├── faq_parser.py  [3.8KB]
│       │   └── pdf_parser.py  [14.1KB]
│       ├── retrieval/
│       │   ├── __init__.py  [472B]
│       │   ├── hybrid_retriever.py  [11.3KB]
│       │   └── sparse_retriever.py  [8.6KB]
│       ├── schemas/
│       ├── __init__.py  [0B]
│       └── schema.py  [10.0KB]
├── tests/
├── .env  [179B]
├── .gitattributes  [467B]
├── .gitignore  [2.1KB]
├── README.md  [23.3KB]
├── requirements.lock.txt  [2.4KB]
├── requirements.txt  [1.4KB]
└── setup_repo.ps1  [4.2KB]
```

## 2. 현재 README.md 전문

- 크기: 23.3KB / 수정일: 2026-08-14 19:53

````markdown
# FinGuide-RAG

> 은행 영업점·고객센터 직원이 **상품설명서·약관·FAQ의 원문 근거**를 바탕으로
> 고객 문의에 정확히 답변하도록 돕는 금융 특화 RAG 시스템 (B2E)

![status](https://img.shields.io/badge/status-검색_파이프라인_완료-green)
![python](https://img.shields.io/badge/python-3.14-blue)
![recall](https://img.shields.io/badge/Recall@5-0.979-brightgreen)
![mrr](https://img.shields.io/badge/MRR-0.847-brightgreen)

하나은행 공개 문서 **308건**을 정규화해 **2,999개 청크**로 색인하고,
Dense + Sparse 하이브리드 검색으로 **문서 단위 Recall@5 0.979**를 달성했습니다.
직접 구축한 **96건 평가셋**으로 모든 개선을 정량 측정했으며,
청킹 전략·결합 방식·가중치를 실험으로 결정했습니다.

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| **하이브리드 검색** | Dense(FAISS) + Sparse(BM25·Kiwi) 결합. 약관 Recall@5 0.739 → 0.957 |
| **구조별 청킹** | 약관은 제N조, 항 구조는 ①②③, 목록형은 번호, FAQ는 Q&A 쌍 (Strategy 패턴) |
| **개정 문서 버전 관리** | `content_hash`, `is_latest`, `superseded_by`로 구버전 답변 차단 |
| **데이터 무결성 자동 검증** | 스캔본·중복·크롤링 오류를 코드가 탐지 (실제로 6건 발견) |
| **정량 평가 체계** | Recall@k, MRR, NDCG@10을 난이도별·문서유형별로 측정 |
| **근거 부족 시 거절** | 검색 신뢰도가 낮으면 답변 생성 대신 거절 *(구현 예정)* |

---

## 1. 문제 정의

은행 영업점 직원은 수백 종의 상품설명서와 약관을 상대로 고객 문의에 답해야 합니다.
문서는 **시행일 단위로 개정**되며, 구버전 약관을 근거로 안내하면 불완전판매 리스크로 직결됩니다.

### 실제로 확인한 사례

수집 과정에서 마이데이터 서비스 설명서의 두 버전을 확보했습니다.

| 항목 | 2022년판 | 2026년판 |
|---|---|---|
| 심의필 | 제2022-상품-170호 | 제2026-설명서-078호 |
| 이용 연령 | 만 19세 미만 제한 | 만 14세 미만 / 14~19세 / 19세 이상 **3단계** |
| 수수료 변경 통지 | 모바일 앱 확인 | **이메일·문자 개별 통지** |

17세 고객이 마이데이터에 가입할 수 있는지 물었을 때, 구버전을 근거로 답하면
"불가"라고 잘못 안내하게 됩니다. 실제로는 비대면 신청이 가능합니다.

**이 프로젝트가 해결하려는 것**

- 직원이 자연어로 질문 → 최신 버전 문서의 해당 조항을 근거와 함께 제시
- 근거가 불충분하면 답변을 생성하지 않고 거절

---

## 2. 왜 B2C가 아니라 B2E인가

| 구분 | B2C: 고객 대상 챗봇 | **B2E: 직원 대상 보조** |
|---|---|---|
| 오답 리스크 | 불완전판매로 법적 책임 발생 | 직원이 1차 검증 |
| 요구 정확도 | 사실상 100% | 근거 제시로 보완 가능 |
| 실무 도입 가능성 | 낮음 | 높음 |
| 설계 방향 | 답변 자동화 | 의사결정 보조 |

환각 리스크를 **사람이 흡수하는 구조**로 설계했습니다.

> **한계 고지**: 실제 은행에는 내부 상품 조회 시스템이 존재합니다. 다만 대부분
> 키워드·메뉴 기반이라 "이 조건에 해당하나요" 같은 자연어 질의에 약합니다.
> 본 프로젝트는 **공개 문서로 만든 프로토타입**이며, 실제 도입 시에는 코퍼스를
> 행내 문서로 교체하는 구조로 설계했습니다.

---


