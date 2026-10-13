"""50 — 정상 초안 30건을 '직원 구어체'로 변환하고, 의미 보존을 규칙으로 1차 점검한다.

왜 변환하는가
-----------
v1 답변은 '문서 요약체'다("…적용됩니다. 다만 …"). 실제 직원 초안은 구어체다
("…적용돼요. 그런데 …은 안 되세요."). 문체 차이를 그대로 두면 검증기 평가가
'문서체는 통과, 구어체는 경고' 같은 문체 효과를 측정하게 된다.

원칙
----
1. LLM 은 문체만 바꾼다. 수치·상품명·조건·한정어 추가/삭제 금지.
2. 변환 결과를 믿지 않는다. 규칙으로 바로 대조한다 (LLM 판사를 쓰지 않는다).
     - 수치: numeric.extract_numerics 결과의 (값, 단위, 범위, 비교, 수식어, 기간) 다중집합 비교
     - 강조어 추가 / 헤지 소실: rules.check_assertiveness (원문=v1 답변)
     - 기관명: rules.detect_institutions 집합 비교
     - 길이: 원문 대비 0.6 ~ 1.6 배
   하나라도 어긋나면 auto_check=REVIEW. 사람은 REVIEW 를 우선 본다.
3. 사람이 이미 채운 열은 절대 덮어쓰지 않는다. clean_drafts_v1.csv 가 있으면
   draft_id 기준으로 병합하고, draft_text 가 비어 있는 행만 새로 변환한다.
4. LLM 응답은 캐시한다. 키 = sha256(모델 | 프롬프트 버전 | 원문).
   프롬프트를 고치면 PROMPT_VERSION 을 올려 이전 결과와 섞이지 않게 한다.

실행
----
    python scripts/50_convert_draft_style.py --dry-run     # API 호출 없이 프롬프트 2건 확인
    python scripts/50_convert_draft_style.py               # 30건 변환 (캐시 없으면 30콜)
    python scripts/50_convert_draft_style.py --recheck     # API 없이 자동 점검만 다시 (사람이 draft_text 를 고친 뒤)

입출력
-----
    입력   data/eval/v2/clean_drafts_v0.csv   (49번 산출)
    출력   data/eval/v2/clean_drafts_v1.csv   (사람 검수 시트 — 이 파일에 Y/N 을 채운다)
    캐시   data/interim/v2_style_cache.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.verification.numeric import extract_numerics  # noqa: E402
from finguide_rag.verification.rules import (  # noqa: E402
    check_assertiveness,
    detect_institutions,
)

V0 = PROJECT_ROOT / "data" / "eval" / "v2" / "clean_drafts_v0.csv"
V1 = PROJECT_ROOT / "data" / "eval" / "v2" / "clean_drafts_v1.csv"
CACHE = PROJECT_ROOT / "data" / "interim" / "v2_style_cache.json"

MODEL = "gpt-4.1-mini"
PROMPT_VERSION = "style-v1"
LEN_RATIO = (0.6, 1.6)

HUMAN_COLS = ["human_is_correct", "human_style_ok", "note"]
OUT_COLS = [
    "draft_id", "query_id", "question", "doc_type", "difficulty", "not_found",
    "v1_answer", "draft_text", "auto_check", "auto_diff",
    "human_is_correct", "human_style_ok", "note",
    "v1_not_found", "evidence_doc_ids", "evidence_chunk_ids", "answer_hint",
    "style_model", "style_prompt_version",
]

SYSTEM = """너는 은행 영업점 직원의 말투를 재현하는 편집자다.
입력은 직원이 고객 문의에 답하려고 정리한 메모다. 이것을 같은 직원이 창구에서
고객에게 말로 안내하듯 자연스러운 구어체 존댓말로 바꿔라.

반드시 지킬 것
- 수치(금리·금액·기간·횟수·나이·날짜)와 단위, '이상/이하/초과/미만/이내', '최고/최대/기본/우대'는
  원문 표기 그대로 옮긴다. 숫자를 한글로 바꾸거나 반올림하지 않는다.
- 상품명·서비스명·기관명은 원문 그대로 쓴다. 줄이거나 다른 이름으로 바꾸지 않는다.
- 조건, 예외('다만', '제외', '~에 한해'), 가능성 표현('~될 수 있습니다')을 하나도 빼지 않는다.
- 원문에 없는 내용, 인사말, 맺음말, 권유를 더하지 않는다.
- '반드시', '무조건', '항상', '그대로' 같은 강조어를 원문에 없으면 추가하지 않는다.
- 문장 순서는 바꿔도 되지만 정보는 1:1로 대응해야 한다.

출력은 JSON 하나: {"draft": "<변환문>"}"""

USER_TMPL = "고객 질문: {question}\n\n직원 메모:\n{answer}"


# ------------------------------------------------------------------
# 입출력
# ------------------------------------------------------------------


def read_csv_any(path: Path) -> list[dict]:
    """엑셀이 CP949 로 다시 저장한 경우까지 읽는다."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            with path.open(encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    sys.exit(f"{path.name} 인코딩을 읽지 못했습니다. 엑셀에서 'CSV UTF-8' 로 다시 저장하세요.")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def cache_key(text: str) -> str:
    return hashlib.sha256(f"{MODEL}|{PROMPT_VERSION}|{text}".encode("utf-8")).hexdigest()[:24]


# ------------------------------------------------------------------
# 의미 보존 자동 점검 (규칙만)
# ------------------------------------------------------------------


def _num_sig(text: str) -> Counter:
    return Counter((s.unit, round(s.value, 6), s.value_hi, s.comparator, s.qualifier, s.period)
                   for s in extract_numerics(text))


def auto_check(src: str, dst: str) -> tuple[str, str]:
    if not dst.strip():
        return "EMPTY", ""
    diffs: list[str] = []
    a, b = _num_sig(src), _num_sig(dst)
    if a != b:
        lost, added = a - b, b - a
        if lost:
            diffs.append(f"수치 소실/변형 {sorted(lost.elements(), key=str)}")
        if added:
            diffs.append(f"수치 추가/변형 {sorted(added.elements(), key=str)}")
    hits, _ = check_assertiveness(dst, src)   # 원문=v1 답변을 '근거'로 본다
    diffs += [h.detail for h in hits]
    ia, ib = detect_institutions(src), detect_institutions(dst)
    if ia != ib:
        diffs.append(f"기관명 {sorted(ia)} → {sorted(ib)}")
    ratio = len(dst) / max(len(src), 1)
    if not (LEN_RATIO[0] <= ratio <= LEN_RATIO[1]):
        diffs.append(f"길이 비율 {ratio:.2f}")
    return ("REVIEW" if diffs else "OK"), " / ".join(diffs)


# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------


def make_client():
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as e:
        sys.exit(f"패키지가 없습니다: {e}. requirements-dev.txt 를 설치하세요.")
    load_dotenv(PROJECT_ROOT / ".env")
    return OpenAI()


def convert(client, question: str, answer: str) -> tuple[str, int]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": USER_TMPL.format(question=question, answer=answer)}],
        temperature=0,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    draft = str(data.get("draft", "")).strip()
    return draft, resp.usage.prompt_tokens + resp.usage.completion_tokens


# ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 프롬프트만 출력")
    ap.add_argument("--recheck", action="store_true", help="API 없이 자동 점검만 다시 계산")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만 (시험용)")
    args = ap.parse_args()

    if not V0.exists():
        sys.exit("clean_drafts_v0.csv 가 없습니다. 49번을 먼저 실행하세요.")
    base = read_csv_any(V0)
    existing = {r["draft_id"]: r for r in read_csv_any(V1)} if V1.exists() else {}

    rows: list[dict] = []
    for r in base:
        old = existing.get(r["draft_id"], {})
        row = dict(r)
        for c in HUMAN_COLS + ["draft_text", "style_model", "style_prompt_version"]:
            row[c] = old.get(c, "") or row.get(c, "")
        rows.append(row)
    targets = [r for r in rows if not r["draft_text"].strip()]
    if args.limit:
        targets = targets[: args.limit]

    if args.dry_run:
        for r in targets[:2]:
            print("=" * 60)
            print(SYSTEM)
            print("-" * 60)
            print(USER_TMPL.format(question=r["question"], answer=r["v1_answer"]))
        print(f"\n변환 대상 {len(targets)}건 (API 호출 없음)")
        return 0

    cache: dict = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    hits = calls = tokens = errors = 0

    if not args.recheck and targets:
        client = None
        for i, r in enumerate(targets, 1):
            k = cache_key(r["question"] + "\n" + r["v1_answer"])
            if k in cache:
                hits += 1
                r["draft_text"] = cache[k]["draft"]
            else:
                if client is None:
                    client = make_client()
                try:
                    draft, t = convert(client, r["question"], r["v1_answer"])
                except Exception as e:   # 한 건 실패로 전체를 멈추지 않는다
                    errors += 1
                    r["note"] = (r["note"] + f" [변환 실패: {str(e)[:60]}]").strip()
                    continue
                cache[k] = {"draft": draft, "tokens": t, "model": MODEL, "prompt": PROMPT_VERSION}
                calls += 1
                tokens += t
                r["draft_text"] = draft
                if calls % 5 == 0:   # 중간 저장 — 도중에 끊겨도 재실행 시 이어서 진행
                    CACHE.parent.mkdir(parents=True, exist_ok=True)
                    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            r["style_model"], r["style_prompt_version"] = MODEL, PROMPT_VERSION
            print(f"  [{i:>2}/{len(targets)}] {r['draft_id']} 변환")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in rows:
        r["auto_check"], r["auto_diff"] = auto_check(r["v1_answer"], r["draft_text"])
    write_csv(V1, rows)

    stat = Counter(r["auto_check"] for r in rows)
    print("=" * 60)
    print("정상 초안 구어체 변환")
    print("=" * 60)
    print(f"전체 {len(rows)}건 | 이번 변환 대상 {len(targets)}건 "
          f"(캐시 {hits} / API {calls}콜 / 실패 {errors})  토큰 {tokens:,}")
    print(f"자동 점검  OK {stat['OK']}  REVIEW {stat['REVIEW']}  EMPTY {stat['EMPTY']}")
    for r in rows:
        if r["auto_check"] == "REVIEW":
            print(f"  - {r['draft_id']}: {r['auto_diff'][:150]}")
    done = sum(1 for r in rows if r["human_is_correct"].strip() and r["human_style_ok"].strip())
    print(f"사람 검수 완료 {done}/{len(rows)}")
    print(f"→ {V1.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
