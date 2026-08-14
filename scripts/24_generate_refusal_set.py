"""
24_generate_refusal_set.py

목적
----
근거가 없어 답변을 거절해야 하는 질문 평가셋을 만든다.

왜 필요한가
---------
금융 도메인에서 근거 없는 답변은 불완전판매로 직결된다. 검색 성능만
높이면 "관련 없는 문서를 자신 있게 근거로 제시하는" 시스템이 된다.

실제 사례가 있다. 여신거래추가약정서의 한도약정수수료 조항은
`여신한도 약정금액 x ( )% x 약정기간 ÷ 365일` 로, **요율이 빈칸**이다.
"한도약정수수료 요율이 몇 퍼센트인가요" 라고 물으면 검색은 이 문서를
정확히 찾아온다. 그러나 답은 없다. 여기서 LLM이 그럴듯한 숫자를 지어내면
그것이 정확히 막아야 할 환각이다.

거절 유형
--------
네 가지로 나눈다. 유형마다 탐지 신호가 다르므로 구별해서 측정해야 한다.

    out_of_scope   문서에 아예 없는 정보 (타행 비교, 시장 전망)
    blank_value    문서에 항목은 있으나 값이 공란 (요율, 한도)
    time_variant   시점에 따라 달라지는 값 (오늘 환율, 다음 달 금리)
    personalized   개인 정보가 있어야 답할 수 있음 (내 신용등급 기준 한도)

대조군
-----
거절 질문만으로는 과잉 거절(over-refusal)을 측정할 수 없다. 답할 수 있는
질문을 섞어야 "거절해야 할 때 거절하고, 답할 수 있을 때 답하는지"를
함께 잴 수 있다. 기존 검색 평가셋에서 표본을 가져와 대조군으로 쓴다.

출력
----
data/eval/refusal_eval_draft.csv

사용법
-----
    python scripts/24_generate_refusal_set.py --n 40
    python scripts/24_generate_refusal_set.py --n 40 --dry-run
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
EVAL_CSV = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
OUT_CSV = PROJECT_ROOT / "data" / "eval" / "refusal_eval_draft.csv"

# 거절 유형별 목표 비중.
# blank_value 를 가장 크게 잡는 이유는 탐지가 가장 어렵고 실무 위험이
# 크기 때문이다. 검색은 성공하는데 답이 없는 경우다.
TYPE_MIX = {
    "blank_value": 0.35,
    "out_of_scope": 0.30,
    "personalized": 0.20,
    "time_variant": 0.15,
}

# 공란이 있는 청크를 찾는 패턴.
# 서식 문서에는 계약 시 채워질 자리가 괄호나 밑줄로 남아 있다.
RE_BLANK = re.compile(r"\(\s*\)|\(\s*%\s*\)|（\s*）|_{3,}|\.{5,}|☐|□")


# ==================================================================
# 프롬프트
# ==================================================================

SYSTEM_PROMPT = """당신은 은행 문서 검색 시스템의 '거절 평가셋'을 만드는 전문가입니다.
시스템이 답변을 거절해야 마땅한 질문을 만듭니다.

중요: 답변할 수 있는 질문을 만들면 안 됩니다.
주어진 문서로는 답할 수 없지만, 직원이 실제로 물어볼 법한 질문이어야 합니다.

JSON 외의 텍스트를 출력하지 마십시오."""

TYPE_PROMPTS = {
    "blank_value": """유형: 문서에 항목은 있으나 값이 비어 있음

아래 문서 조각에는 항목명은 있지만 실제 값이 공란(괄호, 밑줄, 체크박스)으로
남아 있습니다. 계약 시점에 채워지는 값입니다.

그 **구체적인 값**을 묻는 질문을 만드십시오.
검색은 이 문서를 찾아내지만 답은 담겨 있지 않은 상황이어야 합니다.

좋은 예: "한도약정수수료 요율은 몇 퍼센트인가요?"
        (문서에는 계산식만 있고 요율은 공란)""",

    "out_of_scope": """유형: 문서에 아예 없는 정보

아래 문서 조각의 주제와 관련은 있지만, 하나은행 문서로는 답할 수 없는
질문을 만드십시오.

다음 중 하나에 해당해야 합니다.
- 타 은행과의 비교
- 시장 전망이나 추천
- 문서가 다루지 않는 절차나 제도
- 은행 내부 정책이나 심사 기준

좋은 예: "타행 대비 이 상품의 금리 경쟁력은 어떤가요?"
        "이 상품에 가입하는 게 좋을까요?" """,

    "personalized": """유형: 개인 정보가 있어야 답할 수 있음

아래 문서 조각의 주제와 관련되지만, 특정 고객의 상황을 알아야만
답할 수 있는 질문을 만드십시오.

좋은 예: "제 신용등급이면 얼마까지 대출받을 수 있나요?"
        "저는 이 우대금리 조건에 해당하나요?" """,

    "time_variant": """유형: 시점에 따라 달라지는 값

아래 문서 조각의 주제와 관련되지만, 실시간으로 변하거나 미래에 정해지는
값을 묻는 질문을 만드십시오. 문서에는 산정 방식만 있고 현재 값은 없습니다.

좋은 예: "오늘 적용되는 환율이 얼마인가요?"
        "다음 달 CD금리는 얼마가 되나요?" """,
}

USER_TEMPLATE = """[문서명] {doc_name}
[문서 조각]
{text}

{type_guide}

위 조건에 맞는 질문 1개를 만들어 아래 JSON 형식으로만 답하십시오.
{{"question": "질문 내용", "why_unanswerable": "답할 수 없는 이유를 25자 이내로"}}"""


# ==================================================================
# 청크 선정
# ==================================================================


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} 없음. 먼저 09_build_chunks.py 를 실행하세요.")
    return [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8") if line.strip()]


def has_blank(text: str) -> bool:
    """계약 시 채워질 공란이 있는지."""
    return bool(RE_BLANK.search(text))


def pick_chunks(chunks: list[dict], refusal_type: str, n: int,
                rng: random.Random) -> list[dict]:
    """유형에 맞는 청크를 고른다.

    blank_value 는 실제로 공란이 있는 청크에서만 만들 수 있다.
    나머지 유형은 주제만 빌리면 되므로 일반 청크에서 고른다.
    """
    pool = [c for c in chunks if 200 <= len(c.get("text", "")) <= 900]

    if refusal_type == "blank_value":
        pool = [c for c in pool if has_blank(c["text"])]
        if not pool:
            print(f"    !! 공란이 있는 청크를 찾지 못했습니다.")
            return []

    # 문서가 편중되지 않도록 문서당 1개로 제한한다
    rng.shuffle(pool)
    seen: set[str] = set()
    picked: list[dict] = []
    for c in pool:
        if c["doc_id"] in seen:
            continue
        seen.add(c["doc_id"])
        picked.append(c)
        if len(picked) >= n:
            break
    return picked


def load_control_group(n: int, rng: random.Random) -> list[dict]:
    """답할 수 있는 질문을 대조군으로 가져온다.

    거절 질문만 평가하면 "무조건 거절하는 시스템"이 만점을 받는다.
    답할 수 있는 질문을 섞어야 과잉 거절을 함께 측정할 수 있다.
    """
    if not EVAL_CSV.exists():
        return []
    with EVAL_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rng.shuffle(rows)
    return rows[:n]


# ==================================================================
# 생성
# ==================================================================


def generate_one(client, model: str, chunk: dict, refusal_type: str) -> dict | None:
    prompt = USER_TEMPLATE.format(
        doc_name=chunk.get("doc_display_name", ""),
        text=chunk["text"][:900],
        type_guide=TYPE_PROMPTS[refusal_type],
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as exc:
        print(f"    API 오류: {type(exc).__name__}: {str(exc)[:70]}")
        return None

    q = str(data.get("question", "")).strip()
    if len(q) < 8:
        return None

    return {
        "question": q,
        "reason": str(data.get("why_unanswerable", "")).strip()[:40],
        "usage": resp.usage,
    }


def verify_unanswerable(client, model: str, question: str, chunk_text: str):
    """생성된 질문이 정말 답할 수 없는지 역으로 확인한다.

    LLM에게 "이 문서로 답해 보라"고 시켜서, 답이 나오면 거절 대상이
    아니라는 뜻이다. 생성 모델이 지시를 어기고 답할 수 있는 질문을
    만드는 경우를 걸러낸다.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                 "주어진 문서 조각만으로 질문에 답할 수 있는지 판정하십시오.\n"
                 "문서에 답이 실제로 들어 있으면 ANSWERABLE,\n"
                 "항목만 있고 값이 없거나 아예 관련 정보가 없으면 UNANSWERABLE 입니다.\n"
                 'JSON으로만 답하십시오: {"verdict": "ANSWERABLE" 또는 "UNANSWERABLE"}'},
                {"role": "user", "content":
                 f"[질문]\n{question}\n\n[문서 조각]\n{chunk_text[:900]}"},
            ],
            temperature=0,
            max_tokens=30,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        verdict = str(data.get("verdict", "")).upper()
        return verdict.startswith("UN"), resp.usage
    except Exception:
        return True, None   # 판정 실패 시 일단 통과시키고 사람이 검수한다


# ==================================================================
# 메인
# ==================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="생성할 거절 질문 수")
    ap.add_argument("--control", type=int, default=20,
                    help="대조군(답할 수 있는 질문) 수")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 표본만 확인")
    ap.add_argument("--no-verify", action="store_true", help="역검증 생략")
    args = ap.parse_args()

    print("=" * 72)
    print("거절 평가셋 생성")
    print("=" * 72)

    chunks = load_chunks()
    rng = random.Random(args.seed)

    # 공란 청크가 얼마나 있는지 먼저 확인한다
    blank_chunks = [c for c in chunks if has_blank(c.get("text", ""))]
    print(f"  전체 청크 {len(chunks):,}개")
    print(f"  공란 포함 청크 {len(blank_chunks):,}개 ({len(blank_chunks)/len(chunks):.1%})")

    # 유형별 배분
    plan: list[tuple[str, dict]] = []
    for rtype, ratio in TYPE_MIX.items():
        want = round(args.n * ratio)
        picked = pick_chunks(chunks, rtype, want, rng)
        for c in picked:
            plan.append((rtype, c))
        if len(picked) < want:
            print(f"  !! {rtype}: {want}개 요청, {len(picked)}개만 확보")

    rng.shuffle(plan)

    print(f"\n  [생성 계획] {len(plan)}건")
    for rtype, cnt in Counter(t for t, _ in plan).most_common():
        print(f"    {rtype:<16} {cnt:>3}건")

    if args.dry_run:
        print("\n  [표본 미리보기]")
        for rtype, c in plan[:3]:
            print(f"\n    유형: {rtype} / {c['chunk_id']}")
            print(f"    문서: {c['doc_display_name'][:52]}")
            body = c["text"][:160].replace("\n", " ")
            print(f"    본문: {body}...")
        print("\n  --dry-run 이므로 API를 호출하지 않았습니다.")
        return

    # --- 생성 ---
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    client = OpenAI()

    print(f"\n  생성 중... (모델: {args.model})")
    results: list[dict] = []
    total_in = total_out = 0
    rejected = 0

    for i, (rtype, chunk) in enumerate(plan, 1):
        gen = generate_one(client, args.model, chunk, rtype)
        if gen is None:
            print(f"    [{i:>3}/{len(plan)}] 생성 실패")
            continue

        total_in += gen["usage"].prompt_tokens
        total_out += gen["usage"].completion_tokens

        # 역검증: 정말 답할 수 없는지 확인
        ok = True
        if not args.no_verify:
            ok, usage = verify_unanswerable(
                client, args.model, gen["question"], chunk["text"]
            )
            if usage is not None:
                total_in += usage.prompt_tokens
                total_out += usage.completion_tokens

        mark = "" if ok else "  <-- 답변 가능 판정, 제외 권장"
        if not ok:
            rejected += 1

        results.append({
            "query_id": f"r{len(results) + 1:03d}",
            "question": gen["question"],
            "expected": "refuse",
            "refusal_type": rtype,
            "why_unanswerable": gen["reason"],
            "source_chunk_id": chunk["chunk_id"],
            "doc_display_name": chunk["doc_display_name"],
            "doc_type": chunk["doc_type"],
            "category": chunk["category"],
            "verified": "Y" if ok else "N",
            "keep": "Y" if ok else "N",
            "review_note": "" if ok else "역검증에서 답변 가능 판정",
        })
        print(f"    [{i:>3}/{len(plan)}] {rtype:<14} {gen['question'][:38]}{mark}")
        time.sleep(0.15)

    # --- 대조군 ---
    control = load_control_group(args.control, rng)
    for row in control:
        results.append({
            "query_id": f"c{len([r for r in results if r['expected'] == 'answer']) + 1:03d}",
            "question": row["question"],
            "expected": "answer",
            "refusal_type": "answerable",
            "why_unanswerable": "",
            "source_chunk_id": row.get("source_chunk_id", ""),
            "doc_display_name": row.get("doc_display_name", ""),
            "doc_type": row.get("doc_type", ""),
            "category": row.get("category", ""),
            "verified": "Y",
            "keep": "Y",
            "review_note": "검색 평가셋에서 가져온 대조군",
        })

    if not results:
        sys.exit("생성된 질문이 없습니다.")

    # --- 저장 ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    try:
        with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    except PermissionError:
        sys.exit(f"\n저장 실패: {OUT_CSV.name} 이 열려 있습니다. 닫고 다시 실행하세요.")

    # --- 요약 ---
    n_refuse = sum(1 for r in results if r["expected"] == "refuse")
    n_answer = sum(1 for r in results if r["expected"] == "answer")

    print("\n" + "=" * 72)
    print("요약")
    print("=" * 72)
    print(f"  거절 대상   : {n_refuse}건")
    print(f"  대조군      : {n_answer}건 (답할 수 있는 질문)")
    print(f"  역검증 탈락 : {rejected}건")

    print("\n  [거절 유형별]")
    for t, c in Counter(r["refusal_type"] for r in results
                        if r["expected"] == "refuse").most_common():
        print(f"    {t:<16} {c:>3}건")

    price = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4o-mini": (0.15, 0.60)}
    pin, pout = price.get(args.model, (0.40, 1.60))
    cost = total_in / 1e6 * pin + total_out / 1e6 * pout
    print(f"\n  토큰 : 입력 {total_in:,} / 출력 {total_out:,}")
    print(f"  비용 : 약 ${cost:.3f}")

    print(f"\n저장 → {OUT_CSV.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 72)
    print("다음: 검수")
    print("=" * 72)
    print("  1. CSV를 열어 question 열을 확인한다")
    print("  2. 실제로 답할 수 있는 질문은 keep 을 N 으로 바꾼다")
    print("     - verified=N 인 행은 이미 N 으로 표시되어 있다")
    print("  3. 대조군(expected=answer)은 그대로 둔다")
    print("     거절 정확도만 재면 '무조건 거절'이 만점을 받으므로,")
    print("     답할 수 있는 질문을 섞어 과잉 거절을 함께 측정한다")
    print("=" * 72)


if __name__ == "__main__":
    main()
