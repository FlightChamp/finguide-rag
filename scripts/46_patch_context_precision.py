# -*- coding: utf-8 -*-
"""
46_patch_context_precision.py — 42번 스크립트에 Context Precision 추가

목적
----
README 6.8 절의 RAGAs 계열 보조 지표 비교를 완결한다. 기존 Faithfulness,
Answer Relevancy 에 Context Precision 을 더해 세 지표를 한 리포트에서 본다.

왜 정답 라벨을 쓰지 않는가
------------------------
검색 평가셋 96 건에는 생성 평가에서 발견된 상품불일치 반례(id23, id24)가
없다. 두 건은 생성 평가셋을 만들 때 별도로 추가한 질문이기 때문이다.
따라서 gold chunk label 기반으로 계산하면 정작 검증하려는 사례가 대상에서
빠진다. 그래서 RAGAs 정의를 참고해 LLM judge 가 각 근거의 유용성을
판정하는 방식으로 재구현한다.

계산식
-----
    각 근거를 rank 순으로 본다.
    useful 로 판정된 rank 에서 precision@rank 를 구한다.
    그 값들의 평균이 context_precision 이다.
    useful 이 하나도 없으면 0.0.

    예) [O, X, O] -> (1/1 + 2/3) / 2 = 0.833

안전장치
-------
- 치환 지점을 먼저 전수 검사한다. 하나라도 어긋나면 파일을 바꾸지 않는다.
- 변경 전 .bak_<타임스탬프> 백업을 만든다.
- 이미 패치된 파일이면 아무 일도 하지 않는다.
- 치환 후 구문 검사에 실패하면 자동 원복한다.

캐시
----
cache_key 의 첫 인자를 "ctxprec" 로 두어 기존 stmt / verify / qgen 캐시를
건드리지 않는다. 이미 쌓인 판정은 그대로 재사용되고 신규 호출은 샘플당
1 회뿐이다.

사용법
-----
    python scripts/46_patch_context_precision.py --dry-run
    python scripts/46_patch_context_precision.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET = Path("scripts") / "42_ragas_measure.py"
ROOT_MARKERS = ("pyproject.toml", "src", "scripts", "reports")
SENTINEL = "def context_precision("


# ---------------------------------------------------------------------------
# 치환 정의
# ---------------------------------------------------------------------------

PATCHES: list[tuple[str, str, str]] = [
    # 1) 판정기에 useful 판정 메서드 추가 --------------------------------
    (
        "RagasJudge 에 근거 유용성 판정 추가",
        """    def generate_questions(self, answer: str) -> tuple[list[str], bool]:""",
        '''    def judge_contexts(self, question: str,
                       contexts: list[str]) -> list[dict[str, Any]]:
        """각 근거가 질문에 답하는 데 유용한지 rank 순으로 판정한다.

        정답 라벨을 쓰지 않는다. 검색 평가셋에 없는 질문(id23, id24)까지
        대상에 넣어야 상품불일치 반례를 검증할 수 있기 때문이다.
        """
        system = (
            "너는 검색 근거 평가기다. 각 근거가 주어진 질문에 답하는 데 "
            "실제로 유용한지 판정한다. 질문과 주제가 비슷하다는 이유만으로 "
            "유용하다고 하지 마라. 질문이 특정 상품을 가리키는데 근거가 "
            "다른 상품에 대한 것이면 유용하지 않다. "
            'JSON만 출력한다: {"verdicts": [{"rank": 1, "useful": true, '
            '"reason": "..."}]}'
        )
        block = "\\n\\n".join(
            f"[근거 {i}]\\n{c}" for i, c in enumerate(contexts, start=1))
        user = f"[질문]\\n{question}\\n\\n{block}"
        key = cache_key("ctxprec", self.model, question, block)
        result = self._chat_json(key, system, user)
        verdicts = [v for v in result.get("verdicts", []) if isinstance(v, dict)]

        # 판정이 누락된 rank 는 유용하지 않은 것으로 본다(보수적).
        by_rank = {}
        for i, v in enumerate(verdicts, start=1):
            try:
                rank = int(v.get("rank", i) or i)
            except (TypeError, ValueError):
                rank = i
            by_rank[rank] = v
        return [by_rank.get(i, {"rank": i, "useful": False,
                                "reason": "판정 없음"})
                for i in range(1, len(contexts) + 1)]

    def generate_questions(self, answer: str) -> tuple[list[str], bool]:''',
    ),

    # 2) 점수 계산 함수 추가 ----------------------------------------------
    (
        "context_precision 계산 함수 추가",
        """def answer_relevancy(embedder: Embedder, question: str,""",
        '''def context_precision(verdicts: list[dict[str, Any]]) -> float:
    """순위 가중 정밀도.

    useful 로 판정된 rank 마다 precision@rank 를 구해 평균낸다.
    유용한 근거가 상위에 있을수록 값이 커진다.

        [O, X, O] -> (1/1 + 2/3) / 2 = 0.833
        [X, O, X] -> (1/2) / 1       = 0.500
        [X, X, X] -> 0.0
    """
    if not verdicts:
        return 0.0
    hits = 0
    precisions: list[float] = []
    for rank, v in enumerate(verdicts, start=1):
        if v.get("useful") is True:
            hits += 1
            precisions.append(hits / rank)
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def answer_relevancy(embedder: Embedder, question: str,''',
    ),

    # 3) 측정 루프에서 호출 ------------------------------------------------
    (
        "측정 루프에 Context Precision 산출 추가",
        """            questions, noncommittal = judge.generate_questions(s["answer"])
            relevancy = answer_relevancy(
                embedder, s["question"], questions, noncommittal)
            error = ""
        except Exception as exc:  # 한 건 실패가 전체를 막지 않게 한다
            score, supported, judged, verdicts = None, 0, 0, []
            relevancy, noncommittal, error = None, False, str(exc)[:200]""",
        """            questions, noncommittal = judge.generate_questions(s["answer"])
            relevancy = answer_relevancy(
                embedder, s["question"], questions, noncommittal)
            ctx_verdicts = judge.judge_contexts(s["question"], s["contexts"])
            ctx_precision = context_precision(ctx_verdicts)
            error = ""
        except Exception as exc:  # 한 건 실패가 전체를 막지 않게 한다
            score, supported, judged, verdicts = None, 0, 0, []
            ctx_verdicts, ctx_precision = [], None
            relevancy, noncommittal, error = None, False, str(exc)[:200]""",
    ),

    # 4) 결과 행에 필드 추가 ------------------------------------------------
    (
        "결과 행에 Context Precision 필드 추가",
        """            "ragas_answer_relevancy": relevancy,
            "noncommittal": noncommittal,""",
        """            "ragas_answer_relevancy": relevancy,
            "ragas_context_precision": ctx_precision,
            "n_context": len(s.get("contexts") or []),
            "n_useful": sum(1 for v in ctx_verdicts
                            if v.get("useful") is True),
            "context_verdicts": " | ".join(
                f"{i}:{'useful' if v.get('useful') is True else 'not'}"
                f"({str(v.get('reason',''))[:50]})"
                for i, v in enumerate(ctx_verdicts, start=1)),
            "noncommittal": noncommittal,""",
    ),

    # 5) CSV 컬럼 추가 -------------------------------------------------------
    (
        "CSV 출력 컬럼 추가",
        """              "ragas_answer_relevancy", "noncommittal",""",
        """              "ragas_answer_relevancy", "noncommittal",
              "ragas_context_precision", "n_context", "n_useful",
              "context_verdicts",""",
    ),

    # 6) 리포트 지표 정의표 --------------------------------------------------
    (
        "리포트 지표 정의에 Context Precision 추가",
        """    add("| Answer Relevancy | 답변에서 역생성한 질문과 원 질문의 코사인 유사도 평균 | "
        "동일 |")""",
        """    add("| Answer Relevancy | 답변에서 역생성한 질문과 원 질문의 코사인 유사도 평균 | "
        "동일 |")
    add("| Context Precision | 유용한 근거가 상위에 있을수록 높은 순위 가중 정밀도 | "
        "동일 |")
    add("")
    add("> 세 지표 모두 RAGAs 라이브러리를 호출하지 않고 원논문 정의를 참고해 "
        "**RAGAs-style 로 재구현**한 값이다. 특히 Context Precision 은 정답 "
        "청크 라벨이 아니라 LLM judge 의 유용성 판정에 기반한다. 검색 평가셋 "
        "96건에 상품불일치 반례(id23, id24)가 포함돼 있지 않아, 정답 라벨 "
        "방식으로는 검증하려는 사례가 대상에서 빠지기 때문이다.")""",
    ),

    # 7) 전체 결과 요약 ------------------------------------------------------
    (
        "전체 평균에 Context Precision 추가",
        """    add(f"| Answer Relevancy | {fmt(sum(relev)/len(relev)) if relev else '-'} | {len(relev)} |")""",
        """    add(f"| Answer Relevancy | {fmt(sum(relev)/len(relev)) if relev else '-'} | {len(relev)} |")
    ctxp = [r.get("ragas_context_precision") for r in rows
            if isinstance(r.get("ragas_context_precision"), float)]
    add(f"| Context Precision | {fmt(sum(ctxp)/len(ctxp)) if ctxp else '-'} | {len(ctxp)} |")""",
    ),

    # 8) 항목별 표에 컬럼 추가 ------------------------------------------------
    (
        "항목별 표에 Context Precision 열 추가",
        """    add("| id | 유형 | Faithfulness | 지지/전체 | AnswerRel | 사람_환각 | "
        "사람_상품일치 | 사람_수치정확 |")
    add("|---|---|---|---|---|---|---|---|")
    for r in rows:
        add(f"| {r['id']} | {r.get('doc_type','')} | "
            f"{fmt(r['ragas_faithfulness'])} | "
            f"{r['supported']}/{r['judged']} | "
            f"{fmt(r['ragas_answer_relevancy'])} | "
            f"{r.get('human_환각','-') or '-'} | "
            f"{r.get('human_상품일치','-') or '-'} | "
            f"{r.get('human_수치정확','-') or '-'} |")""",
        """    add("| id | 유형 | Faithfulness | 지지/전체 | AnswerRel | CtxPrec | "
        "유용근거 | 사람_환각 | 사람_상품일치 | 사람_수치정확 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        mark = " **←**" if str(r["id"]) in ("23", "24") else ""
        add(f"| {r['id']}{mark} | {r.get('doc_type','')} | "
            f"{fmt(r['ragas_faithfulness'])} | "
            f"{r['supported']}/{r['judged']} | "
            f"{fmt(r['ragas_answer_relevancy'])} | "
            f"{fmt(r.get('ragas_context_precision'))} | "
            f"{r.get('n_useful','-')}/{r.get('n_context','-')} | "
            f"{r.get('human_환각','-') or '-'} | "
            f"{r.get('human_상품일치','-') or '-'} | "
            f"{r.get('human_수치정확','-') or '-'} |")
    add("\\n`←` 표시는 상품불일치 반례로 지목된 항목이다.")

    # 반례 상세
    focus = [r for r in rows if str(r["id"]) in ("23", "24")]
    if focus:
        add("\\n### 상품불일치 반례 상세\\n")
        for r in focus:
            add(f"**id {r['id']}** · {str(r.get('question',''))[:60]}\\n")
            add(f"- Faithfulness {fmt(r['ragas_faithfulness'])} / "
                f"Answer Relevancy {fmt(r['ragas_answer_relevancy'])} / "
                f"**Context Precision {fmt(r.get('ragas_context_precision'))}**")
            add(f"- 유용 판정 근거 {r.get('n_useful','?')}/{r.get('n_context','?')}건")
            add(f"- 사람 판정 상품일치: {r.get('human_상품일치','-') or '-'}")
            if r.get("context_verdicts"):
                add(f"- 근거별 판정: {r['context_verdicts']}")
            add("")""",
    ),

    # 9) 교차표에 Context Precision 판정 추가 -------------------------------
    (
        "핵심 교차표에 Context Precision 대조 추가",
        """    crosstab("human_상품일치", "상품불일치")""",
        """    def crosstab_ctx(label_key: str, title: str) -> None:
        \"\"\"Context Precision 이 사람 판정을 잡아내는지 본다.\"\"\"
        labeled = [r for r in rows
                   if r.get(label_key) in ("pass", "fail")
                   and isinstance(r.get("ragas_context_precision"), float)]
        if not labeled:
            add(f"\\n**{title} — Context Precision**: 라벨이 없어 계산 불가\\n")
            return
        fails = [r for r in labeled if r[label_key] == "fail"]
        passes = [r for r in labeled if r[label_key] == "pass"]
        caught = [r for r in fails if r["ragas_context_precision"] < 1.0]
        add(f"\\n**{title} — Context Precision** "
            f"(라벨 {len(labeled)}건 중 사람 fail {len(fails)}건)\\n")
        add("| 구분 | 건수 | 의미 |")
        add("|---|---|---|")
        add(f"| CtxPrec < 1.0 로 탐지 | {len(caught)} | 표준 지표가 포착 |")
        add(f"| CtxPrec = 1.0 으로 통과 | {len(fails) - len(caught)} | 놓친 오류 |")
        if fails:
            avg_f = sum(r["ragas_context_precision"] for r in fails) / len(fails)
            add(f"\\n- fail 항목 평균 CtxPrec **{avg_f:.3f}**")
        if passes:
            avg_p = sum(r["ragas_context_precision"] for r in passes) / len(passes)
            add(f"- pass 항목 평균 CtxPrec **{avg_p:.3f}**")

    crosstab("human_상품일치", "상품불일치")
    crosstab_ctx("human_상품일치", "상품불일치")""",
    ),

    # 10) 예상 호출 수 안내 --------------------------------------------------
    (
        "예상 호출 수 안내 문구 갱신",
        """    est_calls = len(targets) * 3
    print(f"      예상 LLM 호출: 최대 {est_calls}회 "
          f"(캐시 적중분은 제외됨, 항목당 진술추출 1 + 검증 1 + 역질문 1)")""",
        """    est_calls = len(targets) * 4
    print(f"      예상 LLM 호출: 최대 {est_calls}회 (캐시 적중분은 제외됨, "
          f"항목당 진술추출 1 + 검증 1 + 역질문 1 + 근거유용성 1)")""",
    ),
]


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def find_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if sum(1 for m in ROOT_MARKERS if (cand / m).exists()) >= 2:
            return cand
    return Path.cwd().resolve()


def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    sys.exit(f"[중단] 인코딩을 판별하지 못했습니다: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Precision 패치")
    parser.add_argument("--dry-run", action="store_true",
                        help="변경하지 않고 검사만 수행")
    args = parser.parse_args()

    root = find_root()
    path = root / TARGET
    if not path.exists():
        sys.exit(f"[중단] 대상 파일이 없습니다: {path}")

    print(f"대상: {path}")
    original = read_text(path)

    if SENTINEL in original:
        print("이미 패치된 파일입니다. 변경하지 않고 종료합니다.")
        return

    problems: list[str] = []
    for label, old, _ in PATCHES:
        count = original.count(old)
        print(f"  [{'OK' if count == 1 else '실패'}] {label} (일치 {count}회)")
        if count != 1:
            problems.append(f"{label}: {count}회")

    if problems:
        print("\n[중단] 아래 지점을 찾지 못했습니다. 파일을 변경하지 않았습니다.")
        for p in problems:
            print(f"  - {p}")
        print("\n42번 스크립트가 수정된 상태일 수 있습니다.")
        print("해당 부분 원문을 보내주시면 패치를 맞추겠습니다.")
        sys.exit(1)

    if args.dry_run:
        print(f"\n[dry-run] {len(PATCHES)}개 지점 모두 확인. 변경하지 않았습니다.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, backup)
    print(f"\n백업: {backup.name}")

    patched = original
    for _, old, new in PATCHES:
        patched = patched.replace(old, new, 1)
    path.write_text(patched, encoding="utf-8")
    print(f"패치 완료: {len(PATCHES)}개 지점")

    try:
        compile(patched, str(path), "exec")
        print("구문 검사 통과")
    except SyntaxError as exc:
        shutil.copy2(backup, path)
        sys.exit(f"[중단] 구문 오류가 발생해 원복했습니다: {exc}")

    print("\n다음 실행:")
    print("  python scripts/42_ragas_measure.py --dry-run")
    print("  python scripts/42_ragas_measure.py --yes")


if __name__ == "__main__":
    main()
