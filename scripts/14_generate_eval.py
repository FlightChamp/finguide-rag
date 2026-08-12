"""
14_generate_eval.py

목적
----
청크에서 평가용 질문을 생성해 검색 성능 평가셋의 초안을 만든다.

핵심 설계
--------
1. 난이도 3단계로 나눠 생성한다
   easy   : 상품명 + 정식 용어. 검색이 쉬운 기준선
   medium : 상품명은 있으나 구어체
   hard   : 상품명 없이 상황만 설명

   hard가 중요하다. 실제 관찰에서 "예금 만기 전에 해지하면 이율이..." 같은
   질의가 실패했다. easy만으로 평가하면 Recall이 0.95쯤 나와 개선 여지가
   없어 보이고, 하이브리드 검색을 도입해도 지표가 움직이지 않는다.

2. 청크 표현 베끼기를 막는다
   LLM에게 청크를 주고 질문을 만들라고 하면 청크의 단어를 그대로 쓴다.
   그러면 검색이 지나치게 쉬워져 지표가 거짓말을 한다. 프롬프트로 억제하고,
   생성 후 어절 겹침률을 계산해 과도한 것은 표시한다.

3. 정답 후보를 자동으로 넓힌다
   질문을 만든 청크만 정답으로 두면, 같은 내용이 담긴 다른 청크를 찾아온
   경우가 오답으로 집계된다. 실제로 주택청약 계열 상품설명서처럼 내용이
   유사한 문서가 많다. 따라서 생성 직후 현재 인덱스로 검색해 상위 결과를
   함께 기록하고, 사람이 검수하며 정답을 추가할 수 있게 한다.

사용법
-----
    python scripts/14_generate_eval.py --n 120
    python scripts/14_generate_eval.py --n 20 --dry-run   # 비용 없이 표본만 확인
    python scripts/14_generate_eval.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHUNKS_PATH = PROJECT_ROOT / "data" / "interim" / "chunks.jsonl"
OUT_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_draft.csv"
INDEX_ROOT = PROJECT_ROOT / "data" / "indexes" / "faiss"

# 난이도별 비율. hard 비중을 낮추면 개선 여지가 보이지 않는다.
DIFFICULTY_MIX = {"easy": 0.30, "medium": 0.45, "hard": 0.25}

# 문서 유형별 표본 비중.
# 실제 청크 비중은 설명서 72% / 약관 20% / FAQ 8% 이지만,
# FAQ는 쉬운 질의의 기준선 역할이 필요하므로 20%로 올려 잡는다.
# 반대로 FAQ가 과반을 넘으면 지표가 천장에 붙어 개선폭을 볼 수 없다.
DOC_TYPE_MIX = {"설명서": 0.50, "약관": 0.30, "FAQ": 0.20}

# 질문 생성에 부적합한 청크를 걸러내는 기준
MIN_CHUNK_CHARS = 150   # 너무 짧으면 물어볼 내용이 없다
MAX_CHUNK_CHARS = 900

# 반복 청크 탐지에 쓰는 지문 길이(한글 문자 수).
# 짧으면 서로 다른 내용까지 같은 것으로 묶여 표본이 과도하게 줄어든다.
FINGERPRINT_LEN = 150

# 여러 문서에 반복되는 정형 문구.
# 이런 청크로 질문을 만들면 정답이 수십 개가 되어 평가가 무의미해진다.
# 실제로 장외파생상품설명서 14건에는 동일한 면책 조항이 들어 있고,
# 대출 상품설명서에는 금융소비자보호법 관련 고지가 공통으로 실려 있다.
BOILERPLATE = re.compile(
    # 민원·분쟁 안내
    r"금융감독원|분쟁조정을 신청|고객상담센터|1599-1111|1588-1111"
    # 금융소비자보호법 정형 문구
    r"|자료열람요구권|위법계약해지|청약철회권|위법사실을 안 날"
    # 면책 조항
    r"|책임을 지지 않습니다|보증하지 않습니다|정확성을 보증"
    r"|참고용으로 제공되는|모든 요인을 설명한 것이 아"
    r"|어떠한 손실에 대해서도"
    # 서식·확인란
    r"|서명 또는 인|준법감시인|보존년한"
    # 설명서 표지 안내
    r"|이해를 돕고|참고자료이며|반드시 읽어보시기"
)


# ==================================================================
# 프롬프트
# ==================================================================

SYSTEM_PROMPT = """당신은 은행 영업점 직원용 문서 검색 시스템의 평가셋을 만드는 전문가입니다.
주어진 문서 조각을 근거로 답할 수 있는 질문을 만듭니다.

반드시 지킬 것:
- 질문은 이 문서 조각의 내용으로 답할 수 있어야 합니다.
- 문서에 없는 내용을 묻지 마십시오.
- 이 문서를 찾아야만 답할 수 있는 질문이어야 합니다.
  일반 상식이나 법률 지식으로 답할 수 있는 질문은 만들지 마십시오.
  (나쁜 예: "종합부동산세는 국세인가요 지방세인가요?" - 문서 없이도 답할 수 있음)
- 하나은행의 구체적 상품·서비스·절차에 관한 질문이어야 합니다.
- 답변이 아니라 질문만 만드십시오.
- JSON 외의 텍스트를 출력하지 마십시오."""

DIFFICULTY_PROMPTS = {
    "easy": """난이도: 쉬움
- 상품명이나 제도명을 명시하십시오.
- 문서에 쓰인 정식 용어를 사용하십시오. (예: 중도해지, 우대금리, 가입대상)
- 실무자가 정확한 용어로 검색하는 상황을 가정합니다.
- 다만 문서의 문장을 그대로 복사하지는 마십시오.""",

    "medium": """난이도: 보통
- 상품명은 언급하되, 나머지는 창구에서 고객이 말하듯 구어체로 쓰십시오.
- 정식 용어 대신 일상 표현을 쓰십시오. (중도해지 -> 중간에 해지, 만기 전에 찾으면)
- 문서에 나온 문장을 그대로 옮기지 마십시오.""",

    "hard": """난이도: 어려움 (가장 중요한 조건이니 반드시 지키십시오)

절대 사용하면 안 되는 것:
- 상품명, 서비스명, 제도명 (예: 주택청약종합저축, 환전지갑, 잔액증명서)
- 문서에 등장하는 전문 용어 (예: 중도해지, 기한의 이익, 우대금리)
- 문서의 소제목이나 항목명

반드시 지킬 것:
- 고객이 처한 구체적 상황만 서술하고, 그로부터 알고 싶은 것을 물으십시오.
- 전문 용어를 일상어로 바꾸십시오. (중도해지 -> 만기 전에 찾으면 / 예금주 -> 통장 주인)
- 은행 용어를 모르는 사람이 창구에서 하는 말처럼 쓰십시오.

좋은 예:
- "적금을 3개월 만에 깼는데 이자를 거의 못 받았습니다. 원래 이런가요?"
- "월 납입일에 돈을 못 넣으면 어떤 불이익이 있나요?"

나쁜 예 (용어를 그대로 씀):
- "잔액증명서 발급 당일 거래 제한이 있나요?"  -> 잔액증명서는 문서 용어입니다
- "중도해지 시 이율은 어떻게 되나요?"  -> 중도해지는 문서 용어입니다""",
}

USER_TEMPLATE = """다음은 하나은행 {doc_type}의 일부입니다.

[문서명] {doc_name}
[내용]
{text}

{difficulty_guide}

위 조건에 맞는 질문 1개를 만들어 아래 JSON 형식으로만 답하십시오.
{{"question": "질문 내용", "answer_hint": "문서에서 답이 되는 부분을 20자 이내로 요약"}}"""


# ==================================================================
# 청크 선정
# ==================================================================


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} 없음. 먼저 09_build_chunks.py 를 실행하세요.")
    return [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]


def is_suitable(chunk: dict) -> bool:
    """질문 생성에 적합한 청크인지 판단한다."""
    text = chunk.get("text", "")
    if not (MIN_CHUNK_CHARS <= len(text) <= MAX_CHUNK_CHARS):
        return False
    if BOILERPLATE.search(text):
        return False
    # 표가 뭉개진 청크는 질문을 만들어도 부자연스럽다.
    # 숫자와 기호 비율이 지나치게 높으면 제외한다.
    non_hangul = len(re.sub(r"[가-힣\s]", "", text))
    if non_hangul / len(text) > 0.45:
        return False
    return True


def find_repeated_chunks(chunks: list[dict], min_docs: int = 3) -> set[str]:
    """여러 문서에 반복 등장하는 내용의 청크 ID를 찾는다.

    정규식 목록만으로는 모든 정형 문구를 잡을 수 없다. 실제 데이터에서
    "서로 다른 문서에 거의 같은 내용이 들어 있는" 청크를 직접 찾아낸다.

    이런 청크로 질문을 만들면 정답이 여러 문서에 흩어져 있어, 검색이
    그중 하나를 정확히 찾아와도 오답으로 집계된다.

    비교는 어절 단위 집합의 자카드 유사도가 아니라, 정규화한 앞부분
    지문(fingerprint)으로 한다. 2,700개를 모두 대조하면 느리기 때문이다.
    """
    fingerprints: dict[str, set[str]] = defaultdict(set)   # 지문 -> 문서 ID 집합
    chunk_fp: dict[str, str] = {}

    for c in chunks:
        # 공백·숫자·기호를 제거해 서식 차이를 흡수한다
        norm = re.sub(r"[^가-힣]", "", c.get("text", ""))
        if len(norm) < FINGERPRINT_LEN:
            continue
        # 앞부분이 우연히 겹칠 수 있으므로 지문을 충분히 길게 잡는다.
        # 짧으면 서로 다른 내용까지 같은 것으로 묶여 표본이 과도하게 줄어든다.
        fp = norm[:FINGERPRINT_LEN]
        fingerprints[fp].add(c["doc_id"])
        chunk_fp[c["chunk_id"]] = fp

    repeated_fps = {fp for fp, docs in fingerprints.items() if len(docs) >= min_docs}
    return {cid for cid, fp in chunk_fp.items() if fp in repeated_fps}


def sample_chunks(chunks: list[dict], n: int, seed: int = 42) -> list[dict]:
    """문서 유형별 목표 비중에 맞춰 표본을 뽑는다.

    (문서유형, 카테고리) 조합마다 균등 배분하면 FAQ가 과대 대표된다.
    FAQ는 카테고리가 13개로 잘게 나뉘어 있어 슬롯을 그만큼 차지하는데,
    실제 청크 비중은 8%에 불과하다.

    FAQ는 질문-답변 쌍이라 검색이 압도적으로 쉽다. FAQ가 표본의 절반을
    넘으면 Recall@5가 0.9 이상으로 나오는데, 그것은 FAQ를 잘 찾는다는
    뜻이지 약관 조항을 찾는 능력과는 무관하다. 그런 평가셋으로는
    하이브리드 검색이나 리랭커를 도입해도 개선폭이 보이지 않는다.

    따라서 유형별 목표 비중을 먼저 정하고, 그 안에서 카테고리를 고르게
    섞는다. FAQ 비중을 청크 비중(8%)보다 높은 20%로 두는 것은,
    쉬운 질의의 기준선 역할이 필요하기 때문이다.
    """
    rng = random.Random(seed)
    repeated = find_repeated_chunks(chunks)
    pool = [c for c in chunks if is_suitable(c) and c["chunk_id"] not in repeated]

    # 유형별 목표 개수
    targets = {
        dt: round(n * ratio) for dt, ratio in DOC_TYPE_MIX.items()
    }

    selected: list[dict] = []
    per_doc: Counter = Counter()

    for doc_type, quota in targets.items():
        # 해당 유형 안에서 카테고리별로 묶는다
        groups: dict[str, list[dict]] = defaultdict(list)
        for c in pool:
            if c["doc_type"] == doc_type:
                groups[c["category"]].append(c)

        for g in groups.values():
            rng.shuffle(g)

        if not groups:
            continue

        keys = sorted(groups)
        picked = 0
        # 카테고리를 순회하며 하나씩 뽑아 고르게 섞는다
        while picked < quota:
            added = False
            for key in keys:
                if not groups[key] or picked >= quota:
                    continue
                c = groups[key].pop()
                # 같은 문서에서 질문이 몰리면 평가가 편향된다
                limit = 1 if doc_type == "FAQ" else 3
                if per_doc[c["doc_id"]] >= limit:
                    continue
                per_doc[c["doc_id"]] += 1
                selected.append(c)
                picked += 1
                added = True
            if not added:
                break

    rng.shuffle(selected)
    return selected[:n]


def assign_difficulties(selected: list[dict]) -> list[str]:
    """각 청크에 난이도를 배정한다.

    문서 유형을 고려한다. FAQ는 질문-답변 쌍이라 "상품명 없이 상황만
    설명하는" hard 질문을 만들기 어색하다. 반대로 약관 조항은 hard를
    만들기 좋다. 유형과 무관하게 기계적으로 배분하면 부자연스러운
    질문이 생긴다.
    """
    # 유형별로 허용할 난이도
    allowed = {
        "FAQ": ["easy", "medium"],
        "설명서": ["easy", "medium", "hard"],
        "약관": ["medium", "hard"],
    }

    rng = random.Random(0)
    out: list[str] = []
    counts: Counter = Counter()
    total = len(selected)

    for c in selected:
        options = allowed.get(c["doc_type"], ["medium"])
        # 목표 비율 대비 가장 부족한 난이도를 고른다
        best = min(
            options,
            key=lambda lv: counts[lv] - DIFFICULTY_MIX[lv] * total,
        )
        out.append(best)
        counts[best] += 1

    return out


# ==================================================================
# 생성
# ==================================================================


# 겹침 계산에서 제외할 어휘.
# 금융 문서를 다루는 이상 쓸 수밖에 없는 용어들이다. 이것까지 베낌으로
# 세면 정상적인 질문이 오탐된다.
# 예: "입출금이 자유로운 예금은 언제부터 새 이율이 적용되나요?" 는
#     공식 상품 분류명을 쓴 것이지 문장을 베낀 것이 아니다.
COMMON_TERMS = {
    "예금", "적금", "대출", "은행", "고객", "계좌", "통장", "이자", "이율",
    "금리", "신청", "가입", "해지", "거래", "서비스", "상품", "경우", "기간",
    "지급", "발급", "이용", "확인", "제공", "하나", "하나은행", "본인",
    "가능", "불가", "해당", "관련", "내용", "사항", "방법", "필요",
}


def overlap_ratio(question: str, chunk_text: str) -> float:
    """질문의 어절 중 청크에 그대로 등장하는 비율.

    높을수록 청크를 베낀 질문이라 검색이 지나치게 쉬워진다.

    다만 금융 도메인 필수 용어는 제외한다. "예금", "이율" 같은 단어는
    질문에 안 쓸 수가 없어서, 포함시키면 정상 질문까지 베낌으로 잡힌다.

    또한 2자 어간 비교는 오탐이 많아 3자 이상 어절만 비교한다.
    """
    words = [w for w in re.findall(r"[가-힣]{3,}", question)]
    words = [w for w in words if w not in COMMON_TERMS and w[:2] not in COMMON_TERMS]
    if len(words) < 3:
        return 0.0
    hit = sum(1 for w in words if w[:3] in chunk_text)
    return hit / len(words)


def check_hard_compliance(question: str, chunk: dict) -> str:
    """hard 난이도가 지시를 지켰는지 검사한다.

    LLM은 "용어를 쓰지 말라"는 지시를 자주 무시한다. 실제로 hard로
    지정했는데 "잔액증명서 발급 당일..." 처럼 문서 용어를 그대로 쓴
    사례가 나왔다. 이런 질문은 실질적으로 medium이므로 표시해 둔다.

    반환값이 비어 있으면 준수, 아니면 위반 사유다.
    """
    violations: list[str] = []

    # 1) 문서명에 등장하는 고유 명칭을 썼는가
    doc_name = chunk.get("doc_display_name", "")
    for token in re.findall(r"[가-힣A-Za-z]{3,}", doc_name):
        if token in COMMON_TERMS or token[:2] in COMMON_TERMS:
            continue
        if token in question:
            violations.append(f"문서명 용어 '{token}'")
            break

    # 2) 청크에만 등장하는 전문 용어를 그대로 썼는가
    #    조사가 붙은 형태를 그대로 비교하면 '납입일에' 같은 평범한 말도
    #    걸리므로, 조사를 떼고 명사 부분만 대조한다.
    #    또한 4자 이상 복합명사만 본다. 짧은 말은 일상어일 가능성이 높다.
    for raw in re.findall(r"[가-힣]{4,}", question):
        token = re.sub(
            r"(으로|에서|에게|까지|부터|이나|라도|든지|에는|은|는|이|가|을|를|의|에|도|만|과|와|로)$",
            "",
            raw,
        )
        if len(token) < 4 or token in COMMON_TERMS:
            continue
        if token in chunk.get("text", ""):
            violations.append(f"문서 용어 '{token}'")
            break

    return "; ".join(violations)


# ==================================================================
# 정답 판정
# ==================================================================

JUDGE_SYSTEM = """당신은 검색 시스템 평가셋의 정답을 판정하는 전문가입니다.
질문과 문서 조각이 주어지면, 그 조각만으로 질문에 답할 수 있는지 판정합니다.

YES로 판정할 조건 (모두 충족해야 함):
1. 질문이 묻는 구체적 정보가 이 조각에 실제로 들어 있다.
2. 질문이 전제한 상품·서비스·제도와 이 조각이 다루는 대상이 같다.

NO로 판정해야 하는 경우:
- 다른 상품이나 다른 제도에 관한 내용
  (예: 질문은 적금 납입에 관한 것인데 조각은 대출 연체에 관한 것)
- 주제나 표현은 비슷하지만 질문이 묻는 정보 자체는 없는 경우
- 일반적인 안내만 있고 구체적 답이 없는 경우

특히 주의: "비슷한 상황"과 "같은 상품"은 다릅니다.
청약저축 미납의 불이익과 대출 연체의 불이익은 전혀 다른 내용이므로,
서로의 답이 될 수 없습니다.

엄격하게 판정하십시오. 애매하면 NO입니다.

JSON으로만 답하십시오: {"verdict": "YES" 또는 "NO", "reason": "20자 이내"}"""

JUDGE_TEMPLATE = """[질문]
{question}

[이 질문이 전제한 문서]
{source_doc}

[판정 대상 문서 조각]
출처: {cand_doc}
내용: {text}

판정 대상 조각으로 위 질문에 답할 수 있습니까?
출처가 다른 상품·제도라면 NO입니다."""


def judge_relevance(client, model: str, question: str, source_doc: str,
                    cand_doc: str, chunk_text: str):
    """후보 청크가 질문의 정답인지 LLM으로 판정한다.

    질문을 생성한 청크만 정답으로 두면 평가가 왜곡된다. 예컨대
    이자율 스왑 중도해지 정산 조항은 파생상품 설명서 여러 건에 동일하게
    들어 있어, 검색이 그중 하나를 찾아와도 라벨에 없으면 오답이 된다.

    다만 판정이 관대하면 반대 방향으로 왜곡된다. 실제로 초기 시험에서
    "청약저축 월 납입 미이행" 질문에 "대출 연체이자" 조각을 정답으로
    인정한 사례가 있었다. 표면 구조는 비슷하지만 전혀 다른 내용이다.
    그래서 출처 문서명을 함께 제공해 상품·제도가 같은지 확인하게 한다.

    사람이 110건 x 후보 4개를 모두 검토하는 것은 현실적이지 않으므로
    LLM에게 1차 판정을 맡기고, 사람은 결과만 검수한다.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_TEMPLATE.format(
                    question=question,
                    source_doc=source_doc,
                    cand_doc=cand_doc,
                    text=chunk_text[:900],
                )},
            ],
            temperature=0,   # 판정은 일관성이 중요하므로 무작위성을 없앤다
            max_tokens=60,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        verdict = str(data.get("verdict", "")).upper().startswith("Y")
        return verdict, str(data.get("reason", ""))[:30], resp.usage
    except Exception:
        # 판정 실패 시 정답으로 인정하지 않는다. 잘못 넣는 것보다 낫다.
        return False, "판정실패", None


def generate_one(client, model: str, chunk: dict, difficulty: str) -> dict | None:
    prompt = USER_TEMPLATE.format(
        doc_type=chunk.get("doc_type", "문서"),
        doc_name=chunk.get("doc_display_name", ""),
        text=chunk["text"][:900],
        difficulty_guide=DIFFICULTY_PROMPTS[difficulty],
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,   # 다양성을 위해 높게 잡는다
            max_tokens=200,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        print(f"    API 오류: {type(exc).__name__}: {str(exc)[:80]}")
        return None

    try:
        data = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return None

    question = str(data.get("question", "")).strip()
    if len(question) < 8:
        return None

    return {
        "question": question,
        "answer_hint": str(data.get("answer_hint", "")).strip(),
        "usage": resp.usage,
    }


# ==================================================================
# 정답 후보 확장
# ==================================================================


def find_similar(store, embedder, question: str, top_k: int = 20) -> list[tuple[str, float]]:
    """현재 인덱스로 검색해 상위 결과를 반환한다.

    20위까지 보는 이유는 두 가지다.
    1. 정답 후보를 넓게 확보하기 위해. 5위까지만 보면 실제로 답이 있는
       청크를 놓쳐 라벨이 부족해진다.
    2. 5위 내 미발견의 원인을 구별하기 위해. 20위 안에 있으면 "순위가
       낮은 것"이고, 20위 밖이면 "검색이 완전히 실패한 것"이다.
       전자는 리랭커로 개선 가능하고, 후자는 질의-문서 표현 격차 문제다.
    """
    qv = embedder.encode_queries([question])
    hits = store.search(qv, top_k=top_k)
    return [(h.chunk_id, h.score) for h in hits]


# ==================================================================
# 메인
# ==================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="생성할 질문 수")
    ap.add_argument("--model", default="gpt-4.1-mini", help="사용할 LLM")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 표본만 확인")
    ap.add_argument("--no-search", action="store_true", help="정답 후보 확장 생략")
    ap.add_argument("--judge-top", type=int, default=4,
                    help="LLM으로 정답 판정할 상위 후보 수 (0이면 판정 생략)")
    args = ap.parse_args()

    print("=" * 70)
    print("검색 평가셋 생성")
    print("=" * 70)

    chunks = load_chunks()
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    suitable = [c for c in chunks if is_suitable(c)]
    print(f"  전체 청크 {len(chunks):,}개 중 적합 {len(suitable):,}개")

    selected = sample_chunks(chunks, args.n, args.seed)
    difficulties = assign_difficulties(selected)
    print(f"  표본 {len(selected)}개 선정")

    print("\n  [문서 유형별 분포]")
    by_type: Counter = Counter(c["doc_type"] for c in selected)
    for dt, cnt in by_type.most_common():
        print(f"    {dt:<8} {cnt:>3}개 ({cnt / len(selected) * 100:.0f}%)")

    dist: Counter = Counter()
    for c in selected:
        dist[(c["doc_type"], c["category"])] += 1
    print("\n  [표본 분포]")
    for (dt, cat), cnt in dist.most_common(10):
        print(f"    {dt:<6} {cat:<20} {cnt:>3}개")

    print("\n  [난이도 x 문서유형]")
    cross: Counter = Counter(zip(difficulties, (c["doc_type"] for c in selected)))
    print(f"    {'':<8} {'설명서':>6} {'약관':>6} {'FAQ':>6}   계")
    for level in ("easy", "medium", "hard"):
        row = [cross[(level, dt)] for dt in ("설명서", "약관", "FAQ")]
        print(f"    {level:<8} {row[0]:>6} {row[1]:>6} {row[2]:>6}   {sum(row):>3}")

    if args.dry_run:
        print("\n  [표본 미리보기]")
        for c, d in list(zip(selected, difficulties))[:3]:
            print(f"\n    난이도 {d} / {c['chunk_id']}")
            print(f"    문서: {c['doc_display_name'][:50]}")
            print(f"    본문: {c['text'][:150]}...")
        print("\n  --dry-run 이므로 API를 호출하지 않았습니다.")
        return

    # --- LLM 준비 ---
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()

    # --- 검색기 준비 (정답 후보 확장용) ---
    store = embedder = None
    if not args.no_search:
        index_dir = INDEX_ROOT / "e5-small"
        if (index_dir / "config.json").exists():
            from finguide_rag.embedding import Embedder, FaissStore
            store = FaissStore.load(index_dir)
            embedder = Embedder("e5-small")
            print(f"\n  정답 후보 확장용 인덱스 로드: {store.size:,}개")
        else:
            print("\n  인덱스 없음 — 정답 후보 확장을 생략합니다.")

    # --- 생성 ---
    judge_top = args.judge_top if store is not None else 0
    if judge_top:
        print(f"  정답 판정: 상위 {judge_top}개 후보를 LLM으로 확인합니다")

    print(f"\n  생성 중... (모델: {args.model})")
    results: list[dict] = []
    total_in = total_out = 0
    n_judge_calls = [0]   # 클로저 없이 카운트하기 위해 리스트를 쓴다

    for i, (chunk, difficulty) in enumerate(zip(selected, difficulties), 1):
        gen = generate_one(client, args.model, chunk, difficulty)
        if gen is None:
            print(f"    [{i:>3}/{len(selected)}] 실패 — 건너뜀")
            continue

        total_in += gen["usage"].prompt_tokens
        total_out += gen["usage"].completion_tokens

        question = gen["question"]
        ratio = overlap_ratio(question, chunk["text"])

        hard_issue = check_hard_compliance(question, chunk) if difficulty == "hard" else ""

        similar: list[tuple[str, float]] = []
        if store is not None:
            similar = find_similar(store, embedder, question)

        # 생성 근거 청크가 몇 위로 검색되는지 확인한다.
        # 0이면 20위 안에도 없다는 뜻으로, 검색이 완전히 실패한 경우다.
        found_rank = next(
            (r for r, (cid, _) in enumerate(similar, 1) if cid == chunk["chunk_id"]),
            0,
        )

        # --- 정답 후보 판정 ---
        # 상위 후보를 LLM에게 물어 실제로 답이 들어 있는 것만 정답에 넣는다.
        # 생성 근거 청크는 판정 없이 정답으로 인정한다.
        relevant = [chunk["chunk_id"]]
        judged: list[str] = []

        # 근거 청크가 20위 안에도 없으면 판정을 건너뛴다.
        # 검색이 완전히 실패한 경우인데 대체 문서를 정답으로 인정하면
        # 실패를 성공으로 둔갑시켜 지표를 왜곡한다. 실제로 초기 시험에서
        # 청약저축 질문에 대출 연체 조각이 정답으로 인정된 사례가 있었다.
        if judge_top > 0 and similar and found_rank > 0:
            candidates = [
                cid for cid, _ in similar[:judge_top]
                if cid != chunk["chunk_id"]
            ]
            for cid in candidates:
                cand = chunk_by_id.get(cid, {})
                if not cand.get("text"):
                    continue
                ok, reason, usage = judge_relevance(
                    client, args.model, question,
                    chunk.get("doc_display_name", ""),
                    cand.get("doc_display_name", ""),
                    cand["text"],
                )
                n_judge_calls[0] += 1
                if usage is not None:
                    total_in += usage.prompt_tokens
                    total_out += usage.completion_tokens
                if ok:
                    relevant.append(cid)
                    judged.append(f"{cid}({reason})")

        results.append({
            "query_id": f"q{len(results) + 1:03d}",
            "question": question,
            "difficulty": difficulty,
            "source_chunk_id": chunk["chunk_id"],
            "relevant_chunk_ids": " | ".join(relevant),
            "n_relevant": len(relevant),
            "doc_display_name": chunk["doc_display_name"],
            "doc_type": chunk["doc_type"],
            "category": chunk["category"],
            "answer_hint": gen["answer_hint"],
            "overlap_ratio": round(ratio, 2),
            "hard_issue": hard_issue,
            "found_rank": found_rank,
            "judged_added": " | ".join(judged),
            "top20_chunk_ids": " | ".join(cid for cid, _ in similar),
            "top20_scores": " | ".join(f"{s:.3f}" for _, s in similar),
            "keep": "Y",  # 검수하며 N으로 바꾸면 제외된다
            "review_note": "",
        })

        mark = ""
        if hard_issue:
            mark = f"  <-- 용어 회피 실패({hard_issue[:18]})"
        elif ratio > 0.6:
            mark = "  <-- 베낌 의심"
        elif found_rank == 0:
            mark = "  <-- 20위 밖"
        elif found_rank > 5:
            mark = f"  <-- {found_rank}위"
        if len(relevant) > 1:
            mark += f"  [정답 +{len(relevant) - 1}]"
        print(f"    [{i:>3}/{len(selected)}] {difficulty:<6} {question[:42]}{mark}")

        time.sleep(0.15)  # 레이트 리밋 여유

    if not results:
        sys.exit("생성된 질문이 없습니다.")

    # --- 저장 ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    # --- 요약 ---
    print("\n" + "=" * 70)
    print("요약")
    print("=" * 70)
    print(f"  생성 성공      : {len(results)}/{len(selected)}건")

    copied = [r for r in results if r["overlap_ratio"] > 0.6]
    print(f"  베낌 의심      : {len(copied)}건 (어절 겹침 60% 초과)")

    hard_bad = [r for r in results if r.get("hard_issue")]
    n_hard = sum(1 for r in results if r["difficulty"] == "hard")
    if n_hard:
        print(f"  hard 용어 회피 : {n_hard - len(hard_bad)}/{n_hard}건 준수")

    expanded = [r for r in results if r["n_relevant"] > 1]
    print(f"  정답 추가      : {len(expanded)}건 (LLM 판정, 총 {n_judge_calls[0]}회 호출)")

    print("\n  [난이도별 예비 Recall@5]")
    print("  현재 인덱스(e5-small) 기준 참고치다. 검수 후 확정된다.")
    by_diff: dict[str, list[int]] = defaultdict(list)
    for r in results:
        # 정답 중 하나라도 5위 안에 있으면 성공으로 본다
        top5 = r["top20_chunk_ids"].split(" | ")[:5]
        rel = set(r["relevant_chunk_ids"].split(" | "))
        by_diff[r["difficulty"]].append(1 if rel & set(top5) else 0)
    for level in ("easy", "medium", "hard"):
        vals = by_diff.get(level, [])
        if vals:
            print(f"    {level:<8} {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.2f}")

    # 검색 실패의 원인을 나눠서 본다.
    # 20위 안에 있으면 리랭커로 개선할 여지가 있고,
    # 20위 밖이면 질의와 문서의 표현 격차가 근본 원인이다.
    outside = [r for r in results if r["found_rank"] == 0]
    low_rank = [r for r in results if 5 < r["found_rank"] <= 20]
    print(f"\n  [검색 실패 분석]")
    print(f"    6~20위 (리랭커로 개선 여지) : {len(low_rank)}건")
    print(f"    20위 밖 (표현 격차 문제)     : {len(outside)}건")

    # --- 비용 ---
    # 대략적인 추정치다. 정확한 금액은 OpenAI 대시보드에서 확인한다.
    price = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4o-mini": (0.15, 0.60)}
    pin, pout = price.get(args.model, (0.40, 1.60))
    cost = total_in / 1e6 * pin + total_out / 1e6 * pout
    print(f"\n  토큰 사용      : 입력 {total_in:,} / 출력 {total_out:,}")
    print(f"  예상 비용      : 약 ${cost:.3f}")
    print("  (생성 + 정답 판정 합계. 정확한 금액은 OpenAI 대시보드에서 확인)")

    print(f"\n저장 → {OUT_CSV.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 70)
    print("다음: 검수")
    print("=" * 70)
    print("  정답 후보는 LLM이 1차 판정했으므로, 사람은 아래만 확인하면 된다.")
    print()
    print("  1. CSV를 Excel로 연다")
    print("  2. 아래에 해당하면 keep 을 N 으로 바꾼다")
    print("     - hard_issue 가 채워진 행 (난이도 지시 위반)")
    print("     - overlap_ratio 0.6 초과 (청크를 베낀 질문)")
    print("     - 문서 없이도 답할 수 있는 일반 상식 질문")
    print("     - 질문 자체가 어색하거나 답이 없는 경우")
    print("  3. judged_added 열을 훑어 LLM 판정이 이상한 것만 손본다")
    print("  4. 저장 후 15_finalize_eval.py 를 실행한다")
    print("=" * 70)


if __name__ == "__main__":
    main()
