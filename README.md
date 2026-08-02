# FinGuide-RAG

> 은행 영업점·고객센터 직원이 **상품설명서·약관·FAQ의 원문 근거**를 바탕으로
> 고객 문의에 정확히 답변하도록 돕는 금융 특화 RAG 시스템 (B2E)

![status](https://img.shields.io/badge/status-in%20development-yellow)
![python](https://img.shields.io/badge/python-3.14-blue)

---

## 1. 문제 정의

<!-- TODO: 불완전판매 관련 통계나 금감원 자료로 근거를 보강할 것 -->

은행 영업점 직원은 수백 종의 상품설명서와 약관을 상대로 고객 문의에 답해야 한다.
문서는 **시행일 단위로 개정**되며, 구버전 약관을 근거로 안내하면 불완전판매 리스크로 직결된다.

**해결하려는 것**

- 직원이 자연어로 질문하면 → 최신 버전 문서의 해당 조항을 근거와 함께 제시
- 근거가 불충분하면 **답변을 생성하지 않고 거절**한다

## 2. 왜 B2C가 아니라 B2E인가

| 구분 | B2C (고객 대상 챗봇) | **B2E (직원 대상 보조)** |
|---|---|---|
| 오답 리스크 | 불완전판매 → 법적 책임 | 직원이 1차 검증 |
| 요구 정확도 | 사실상 100% | 근거 제시로 보완 가능 |
| 실무 도입 가능성 | 낮음 | 높음 |

환각 리스크를 **사람이 흡수하는 구조**로 설계하여, 금융권에서 실제 도입 가능한 형태를 택했다.

## 3. 시스템 아키텍처

<!-- TODO: mermaid 다이어그램으로 교체 -->

```
문서 수집 → 파싱(유형별) → 청킹(유형별) → 임베딩 → 하이브리드 검색 → 재순위 → 생성 / 거절
                                                   ├─ FAISS (Dense)
                                                   └─ BM25  (Sparse, Kiwi 토크나이저)
```

## 4. 데이터

| 유형 | 건수 | 형식 | 청킹 단위 |
|---|---|---|---|
| FAQ | 200 | JSONL (13개 카테고리) | Q&A 쌍 |
| 상품설명서 | 80 | PDF | 표 보존 + 섹션 |
| 약관 | 28 | PDF | 조항 (제N조) |

- 문서 대장: `data/registry/document_registry.csv` (121행)
- 파일명 규칙: `hana_{유형}_{카테고리}_{상품명}_{YYYYMMDD}.pdf`

## 5. 핵심 설계 결정

<!-- TODO: 면접에서 가장 많이 묻는 부분. 각 근거를 실제 실험 결과로 채울 것. -->

| 결정 | 선택 | 근거 |
|---|---|---|
| 검색 방식 | Dense + Sparse 하이브리드 | 금융 상품명·용어는 정확 매칭이 중요 → BM25로 보완 |
| 한국어 토크나이저 | Kiwi | Java 의존성 없음, Windows 환경 재현 용이 |
| 청킹 전략 | 문서 유형별 분기 (Strategy 패턴) | 약관/설명서/FAQ의 구조가 근본적으로 다름 |
| 버전 관리 | `content_hash`, `is_latest`, `superseded_by` | 개정 문서로 인한 오답 방지 |

## 6. 평가

### 6.1 검색 성능

| 지표 | 베이스라인 | 개선 후 |
|---|---|---|
| Recall@5 | TBD | TBD |
| MRR | TBD | TBD |
| NDCG@10 | TBD | TBD |

### 6.2 거절(Refusal) 평가

문서에 근거가 없는 질문에 대해 시스템이 답변을 거절하는지 측정한다.

| 지표 | 값 |
|---|---|
| Refusal Accuracy | TBD |
| False Answer Rate | TBD |

평가셋: `data/eval/refusal_test_set.csv`

## 7. 실행 방법

```powershell
# 1) 가상환경 (Python 3.14)
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2) 원본 문서 배치
#    data/raw/hana/ 하위에 문서를 배치한다 (저작권상 저장소에 미포함)

# 3) 인덱스 구축
python scripts/10_build_index.py

# 4) 데모 실행
streamlit run app.py
```

## 8. 프로젝트 구조

```
financial_rag_project/
├─ src/finguide_rag/
│  ├─ parsing/        # PDF·JSONL → 공통 Document 스키마
│  ├─ chunking/       # 문서 유형별 청킹 전략
│  ├─ embedding/
│  ├─ retrieval/      # hybrid_retriever.py
│  └─ evaluation/
├─ scripts/           # 실행 스크립트 (번호 순)
├─ data/
│  ├─ raw/            # 원본 (git 제외)
│  ├─ registry/       # 문서 대장 (git 포함)
│  ├─ eval/           # 평가셋 (git 포함)
│  ├─ processed/      # 중간 산출물 (git 제외)
│  └─ indexes/        # 벡터 인덱스 (git 제외)
└─ reports/           # 평가 리포트
```

## 9. 진행 현황

- [x] FAQ 200건 크롤링 및 무결성 검증
- [x] 상품설명서 80건 / 약관 28건 확보
- [x] 문서 대장 구축 (121행)
- [x] 저장소 구조화 및 개발 환경 구성
- [ ] PDF 파싱 및 공통 스키마 정규화
- [ ] 문서 유형별 청킹
- [ ] 임베딩 및 벡터 인덱스 구축
- [ ] 하이브리드 검색기
- [ ] 검색 성능 평가 (Recall@5 / MRR)
- [ ] 거절 평가셋 구축 및 측정
- [ ] Streamlit 데모

## 10. 데이터 고지

본 저장소는 하나은행이 공개한 문서를 **분석 목적으로만** 활용하며,
원본 문서 파일은 저작권상 저장소에 포함하지 않는다.
수집한 문서의 목록과 시점은 `data/registry/document_registry.csv`에서 확인할 수 있다.
