# -*- coding: utf-8 -*-
"""
39_ragas_measure.py — RAGAS 표준 지표 병행 측정 및 자체 지표와의 대조

목적
----
자체 정의 지표(환각/상품일치/수치정확)가 임의로 만든 것이 아님을 보이고,
동시에 "표준 지표만으로는 금융 도메인 오류를 잡지 못한다"를 수치로 증명한다.

측정 지표 (RAGAS 원논문 정의를 그대로 구현, 라이브러리 미사용)
------------------------------------------------------------
1) Faithfulness
   답변에서 원자적 진술(statement) 집합 S(A)를 추출하고, 각 진술이 근거 C에
   의해 지지되는지 검증한다. F = |지지된 진술| / |전체 진술|.

2) Answer Relevancy
   답변 A로부터 역으로 질문 q_1..q_m을 생성하고, 각 q_i와 원 질문 Q의
   임베딩 코사인 유사도를 평균낸다. AR = (1/m) * Σ sim(Q, q_i).
   답변이 회피성(noncommittal)이면 0으로 처리한다.

라이브러리(ragas)를 쓰지 않는 이유
--------------------------------
- ragas는 langchain/datasets/pydantic 의존이 무거워 Python 3.14 환경에서
  설치가 깨질 위험이 있다.
- 판정 캐싱·온도 고정 등 재현성 통제를 직접 넣어야 한다. 이미 프로젝트에서
  LLM 판정 흔들림 문제를 겪었으므로 구현을 통제 가능하게 두는 편이 낫다.
- 계산식은 논문 정의를 따르므로 값의 의미는 동일하다.

입력
----
- data/interim/g_eval_answers.json   (필수) 질문·답변·근거 25건
- reports/g_eval_labels_*.csv        (선택) 사람 라벨
- reports/g_eval_v2_*.csv            (선택) v2 자동 평가기 라벨
※ 위 셋은 `id` 컬럼으로 조인한다.

출력
----
- reports/ragas_measure_<날짜>.csv   항목별 RAGAS 점수 + 라벨 조인 결과
- reports/ragas_measure_<날짜>.md    요약 + 교차표(핵심 산출물)
- data/interim/ragas_cache_<모델>.json  LLM 판정 캐시(재실행 시 비용 0)

사용법
-----
    python scripts/39_ragas_measure.py                 # 실측
    python scripts/39_ragas_measure.py --dry-run       # 비용·대상만 확인
    python scripts/39_ragas_measure.py --yes           # 확인 프롬프트 생략
    python scripts/39_ragas_measure.py --embed openai  # 임베딩을 OpenAI로

의존성: openai, numpy, (기본 임베딩) sentence-transformers — 모두 설치돼 있음
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

CHAT_MODEL = "gpt-4.1-mini"
LOCAL_EMBED_MODEL = "intfloat/multilingual-e5-small"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

#: answer relevancy에서 역생성할 질문 수 (RAGAS 기본값과 동일한 취지)
N_GENERATED_QUESTIONS = 3

#: faithfulness가 이 값 이상이면 "표준 지표는 통과시켰다"로 간주
FAITHFUL_PASS_THRESHOLD = 0.8

ROOT_MARKERS = ("pyproject.toml", "src", "scripts", "reports")

#: 근거 리스트의 원소가 dict일 때 본문으로 쓸 키 후보
EVIDENCE_TEXT_KEYS = ("text", "chunk_text", "content", "body", "passage", "근거")

#: 답변이 비어 있거나 거절인지 판단할 decision 값
REFUSE_DECISIONS = {"refuse", "refused", "거절", "reject"}


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if sum(1 for m in ROOT_MARKERS if (cand / m).exists()) >= 2:
            return cand
    return Path.cwd().resolve()


def read_text(path: Path) -> str | None:
    # utf-8-sig 를 먼저 시도한다. BOM이 있으면 제거하고, 없으면 utf-8 과 동일하게
    # 동작하므로 첫 컬럼명에 \ufeff 가 붙는 문제를 원천 차단한다.
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    return None


def load_env(root: Path) -> None:
    """.env 를 직접 파싱해 환경변수에 넣는다(python-dotenv 의존 회피)."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    env_path = root / ".env"
    text = read_text(env_path)
    if text is None:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def newest(root: Path, pattern: str, exclude: str | None = None) -> Path | None:
    """패턴에 맞는 파일 중 최신 것을 고른다. exclude 문자열이 이름에 있으면 제외."""
    hits = [Path(p) for p in glob.glob(str(root / pattern))]
    if exclude:
        hits = [p for p in hits if exclude not in p.name]
    if not hits:
        return None
    # 파일명이 아니라 수정시각 기준으로 고른다(이름 정렬 함정 회피)
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def norm_id(value: Any) -> str:
    """'01', '1', ' 1 ', BOM 섞인 값을 같은 키로 맞춘다."""
    text = str(value).strip().lstrip("\ufeff").strip('"').strip("'")
    stripped = text.lstrip("0")
    return stripped if stripped else (text or "")


def row_id(row: dict[str, str]) -> str:
    """CSV 행에서 id 컬럼을 찾는다(컬럼명이 오염돼 있어도 동작)."""
    for key in row:
        if key is None:
            continue
        clean = str(key).strip().lstrip("\ufeff").lower()
        if clean == "id":
            return norm_id(row[key])
    return ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    text = read_text(path)
    if text is None:
        return []
    return list(csv.DictReader(text.splitlines()))


def cache_key(*parts: str) -> str:
    joined = "\u0000".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# 데이터 적재
# ---------------------------------------------------------------------------

def evidence_to_text(item: Any) -> str:
    """근거 원소를 본문 문자열로 정규화한다."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in EVIDENCE_TEXT_KEYS:
            if key in item and isinstance(item[key], str) and item[key].strip():
                return item[key].strip()
        # 본문 키를 못 찾으면 가장 긴 문자열 값을 쓴다
        strings = [v for v in item.values() if isinstance(v, str)]
        if strings:
            return max(strings, key=len).strip()
    return ""


def load_samples(path: Path) -> list[dict[str, Any]]:
    """g_eval_answers.json 을 표준 레코드 리스트로 만든다."""
    text = read_text(path)
    if text is None:
        sys.exit(f"[중단] 읽을 수 없습니다: {path}")
    obj = json.loads(text)

    raw: list[dict[str, Any]]
    if isinstance(obj, list):
        raw = [r for r in obj if isinstance(r, dict)]
    elif isinstance(obj, dict):
        raw = []
        for key, value in obj.items():
            if isinstance(value, dict):
                rec = dict(value)
                rec.setdefault("id", key)
                raw.append(rec)
    else:
        sys.exit(f"[중단] 예상치 못한 JSON 구조: {path}")

    out: list[dict[str, Any]] = []
    for i, rec in enumerate(raw, start=1):
        evidences = rec.get("evidences") or rec.get("contexts") or []
        if not isinstance(evidences, list):
            evidences = [evidences]
        contexts = [t for t in (evidence_to_text(e) for e in evidences) if t]
        out.append({
            "id": str(rec.get("id", f"{i:02d}")),
            "group": rec.get("group", ""),
            "difficulty": rec.get("difficulty", ""),
            "doc_type": rec.get("doc_type", ""),
            "question": (rec.get("question") or "").strip(),
            "answer": (rec.get("answer") or "").strip(),
            "decision": str(rec.get("decision", "")).strip().lower(),
            "refusal_reason": rec.get("refusal_reason", ""),
            "stage": rec.get("stage", ""),
            "contexts": contexts,
        })
    return out


def is_measurable(sample: dict[str, Any]) -> tuple[bool, str]:
    """RAGAS 계산 대상인지 판정한다."""
    if sample["decision"] in REFUSE_DECISIONS:
        return False, "거절 응답 (생성 지표 대상 아님)"
    if not sample["answer"]:
        return False, "답변 본문 없음"
    if not sample["contexts"]:
        return False, "근거 없음"
    if not sample["question"]:
        return False, "질문 없음"
    return True, ""


def join_labels(samples: list[dict[str, Any]], root: Path) -> list[str]:
    """사람 라벨·v2 라벨을 id로 조인한다. 조인 결과 설명을 돌려준다."""
    notes: list[str] = []

    def attach(path: Path | None, prefix: str, title: str) -> None:
        if path is None:
            notes.append(f"{title}: 파일 없음")
            return
        rows = read_csv_rows(path)
        table = {row_id(r): r for r in rows if row_id(r)}
        hit = 0
        matched_cols: set[str] = set()
        for s in samples:
            row = table.get(norm_id(s["id"]))
            if not row:
                continue
            hit += 1
            for key, value in row.items():
                clean = str(key).strip().lstrip("\ufeff")
                if clean.startswith(prefix):
                    s[clean] = (value or "").strip()
                    matched_cols.add(clean)
        notes.append(f"{title}: `{path.name}` → {hit}/{len(samples)}건 조인"
                     + (f", 컬럼 {len(matched_cols)}개" if matched_cols else ""))
        if hit == 0:
            # 조인 실패 원인을 즉시 알 수 있게 양쪽 키를 보여준다
            left = [norm_id(s["id"]) for s in samples[:5]]
            right = list(table)[:5]
            notes.append(f"  ↳ 진단: 샘플 id {left} / CSV id {right} "
                         f"/ CSV 컬럼 {list(rows[0])[:6] if rows else '없음'}")
        elif not matched_cols:
            notes.append(f"  ↳ 진단: id는 맞았으나 '{prefix}' 로 시작하는 컬럼이 없음 "
                         f"/ CSV 컬럼 {list(rows[0])[:8] if rows else '없음'}")

    attach(newest(root, "reports/g_eval_labels_*.csv"), "human_", "사람 라벨")
    attach(newest(root, "reports/g_eval_v2_*.csv", exclude="sentences"),
           "v2_", "v2 자동 라벨")
    return notes


# ---------------------------------------------------------------------------
# LLM / 임베딩 래퍼
# ---------------------------------------------------------------------------

class JudgeCache:
    """LLM 응답 캐시. 재실행 시 비용 0, 판정 흔들림 0."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            text = read_text(path)
            if text:
                try:
                    self.data = json.loads(text)
                except json.JSONDecodeError:
                    self.data = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        if key in self.data:
            self.hits += 1
            return self.data[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")


class RagasJudge:
    """RAGAS faithfulness / answer relevancy 계산기."""

    def __init__(self, cache: JudgeCache, model: str = CHAT_MODEL):
        from openai import OpenAI  # 지연 임포트

        self.client = OpenAI()
        self.model = model
        self.cache = cache
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # -- 내부 호출 -------------------------------------------------------
    def _chat_json(self, key: str, system: str, user: str) -> dict[str, Any]:
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        usage = getattr(response, "usage", None)
        if usage:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0

        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"_parse_error": content[:400]}
        self.cache.put(key, parsed)
        return parsed

    # -- 1) Faithfulness -------------------------------------------------
    def extract_statements(self, question: str, answer: str) -> list[str]:
        system = (
            "너는 문장 분해기다. 주어진 답변을 더 이상 쪼갤 수 없는 "
            "원자적 사실 진술(atomic statement)들로 나눈다. "
            "각 진술은 그 자체로 참·거짓을 판정할 수 있어야 하고, "
            "대명사는 실제 대상으로 바꿔 쓴다. 인사말·안내문구는 제외한다. "
            'JSON만 출력한다: {"statements": ["...", "..."]}'
        )
        user = f"[질문]\n{question}\n\n[답변]\n{answer}"
        key = cache_key("stmt", self.model, question, answer)
        result = self._chat_json(key, system, user)
        items = result.get("statements", [])
        return [s.strip() for s in items if isinstance(s, str) and s.strip()]

    def verify_statements(self, contexts: list[str],
                          statements: list[str]) -> list[dict[str, Any]]:
        system = (
            "너는 근거 검증기다. 각 진술이 제공된 근거만으로 지지되는지 판정한다. "
            "세상 상식으로는 참이더라도 근거에 없으면 지지되지 않음(false)이다. "
            'JSON만 출력한다: {"verdicts": [{"idx": 1, "supported": true, '
            '"reason": "..."}]}'
        )
        ctx_block = "\n\n".join(
            f"[근거 {i}]\n{c}" for i, c in enumerate(contexts, start=1))
        stmt_block = "\n".join(
            f"{i}. {s}" for i, s in enumerate(statements, start=1))
        user = f"{ctx_block}\n\n[검증할 진술]\n{stmt_block}"
        key = cache_key("verify", self.model, ctx_block, stmt_block)
        result = self._chat_json(key, system, user)
        verdicts = result.get("verdicts", [])
        return [v for v in verdicts if isinstance(v, dict)]

    def faithfulness(self, question: str, answer: str,
                     contexts: list[str]) -> tuple[float | None, int, int, list]:
        statements = self.extract_statements(question, answer)
        if not statements:
            return None, 0, 0, []
        verdicts = self.verify_statements(contexts, statements)
        supported = sum(1 for v in verdicts if v.get("supported") is True)
        total = len(statements)
        # 판정 수가 진술 수와 다르면 판정된 것만으로 계산하되 기록에 남긴다
        judged = len(verdicts) if verdicts else total
        score = supported / judged if judged else None
        return score, supported, judged, verdicts

    # -- 2) Answer relevancy --------------------------------------------
    def generate_questions(self, answer: str) -> tuple[list[str], bool]:
        system = (
            "너는 역질문 생성기다. 주어진 답변만 보고, 그 답변이 직접 답하고 있는 "
            f"질문을 정확히 {N_GENERATED_QUESTIONS}개 생성한다. "
            "답변이 회피적이거나(모른다·확인이 필요하다 등) 실질 정보가 없으면 "
            "noncommittal을 true로 둔다. "
            'JSON만 출력한다: {"questions": ["...", "..."], "noncommittal": false}'
        )
        key = cache_key("qgen", self.model, answer)
        result = self._chat_json(key, system, f"[답변]\n{answer}")
        questions = [q.strip() for q in result.get("questions", [])
                     if isinstance(q, str) and q.strip()]
        return questions, bool(result.get("noncommittal", False))


class Embedder:
    """질문 임베딩. 기본은 로컬 e5-small(비용 0), 대안은 OpenAI."""

    def __init__(self, backend: str = "local"):
        self.backend = backend
        if backend == "local":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(LOCAL_EMBED_MODEL)
        else:
            from openai import OpenAI
            self.client = OpenAI()

    def encode(self, texts: list[str]):
        import numpy as np

        if self.backend == "local":
            # e5 계열은 "query: " 접두사를 요구한다
            prefixed = [f"query: {t}" for t in texts]
            vecs = self.model.encode(prefixed, normalize_embeddings=True)
            return np.asarray(vecs, dtype=float)

        response = self.client.embeddings.create(
            model=OPENAI_EMBED_MODEL, input=texts)
        vecs = np.asarray([d.embedding for d in response.data])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def answer_relevancy(embedder: Embedder, question: str,
                     generated: list[str], noncommittal: bool) -> float | None:
    import numpy as np

    if noncommittal:
        return 0.0
    if not generated:
        return None
    vecs = embedder.encode([question, *generated])
    origin, others = vecs[0], vecs[1:]
    sims = others @ origin
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(rows: list[dict[str, Any]], skipped: list[dict[str, Any]],
                 notes: list[str], judge: RagasJudge,
                 source: Path) -> str:
    out: list[str] = []
    add = out.append

    add(f"# RAGAS 표준 지표 병행 측정 — {datetime.now():%Y-%m-%d %H:%M}\n")
    add(f"- 입력: `{source.name}`")
    add(f"- 판정 모델: `{judge.model}` (temperature=0, 판정 캐싱 적용)")
    add(f"- 측정 대상: {len(rows)}건 / 제외 {len(skipped)}건")
    for note in notes:
        add(f"- {note}")

    add("\n## 1. 지표 정의\n")
    add("| 지표 | 정의 | 출처 |")
    add("|---|---|---|")
    add("| Faithfulness | 답변의 원자적 진술 중 근거가 지지하는 비율 | "
        "RAGAs (Es et al., EACL 2024) |")
    add("| Answer Relevancy | 답변에서 역생성한 질문과 원 질문의 코사인 유사도 평균 | "
        "동일 |")

    # 전체 평균
    faith = [r["ragas_faithfulness"] for r in rows
             if isinstance(r["ragas_faithfulness"], float)]
    relev = [r["ragas_answer_relevancy"] for r in rows
             if isinstance(r["ragas_answer_relevancy"], float)]

    add("\n## 2. 전체 결과\n")
    add("| 지표 | 평균 | n |")
    add("|---|---|---|")
    add(f"| Faithfulness | {fmt(sum(faith)/len(faith)) if faith else '-'} | {len(faith)} |")
    add(f"| Answer Relevancy | {fmt(sum(relev)/len(relev)) if relev else '-'} | {len(relev)} |")

    # 항목별
    add("\n## 3. 항목별 결과\n")
    add("| id | 유형 | Faithfulness | 지지/전체 | AnswerRel | 사람_환각 | "
        "사람_상품일치 | 사람_수치정확 |")
    add("|---|---|---|---|---|---|---|---|")
    for r in rows:
        add(f"| {r['id']} | {r.get('doc_type','')} | "
            f"{fmt(r['ragas_faithfulness'])} | "
            f"{r['supported']}/{r['judged']} | "
            f"{fmt(r['ragas_answer_relevancy'])} | "
            f"{r.get('human_환각','-') or '-'} | "
            f"{r.get('human_상품일치','-') or '-'} | "
            f"{r.get('human_수치정확','-') or '-'} |")

    # 핵심 교차표
    add("\n## 4. 핵심 — 표준 지표가 놓친 도메인 오류\n")
    add(f"Faithfulness ≥ {FAITHFUL_PASS_THRESHOLD} 이면 표준 지표는 "
        "'문제 없음'으로 통과시킨 것으로 본다.\n")

    def crosstab(label_key: str, title: str) -> None:
        labeled = [r for r in rows
                   if r.get(label_key) in ("pass", "fail")
                   and isinstance(r["ragas_faithfulness"], float)]
        if not labeled:
            add(f"\n**{title}**: 조인된 라벨이 없어 계산 불가\n")
            return
        missed = [r for r in labeled
                  if r[label_key] == "fail"
                  and r["ragas_faithfulness"] >= FAITHFUL_PASS_THRESHOLD]
        caught = [r for r in labeled
                  if r[label_key] == "fail"
                  and r["ragas_faithfulness"] < FAITHFUL_PASS_THRESHOLD]
        n_fail = len(missed) + len(caught)
        add(f"\n**{title}** (라벨 있는 {len(labeled)}건 중 사람이 fail 판정한 "
            f"{n_fail}건)\n")
        add("| 구분 | 건수 | 의미 |")
        add("|---|---|---|")
        add(f"| Faithfulness가 통과시킴 | {len(missed)} | "
            "표준 지표가 놓친 오류 |")
        add(f"| Faithfulness도 잡아냄 | {len(caught)} | 표준 지표로도 탐지 |")
        if missed:
            ids = ", ".join(r["id"] for r in missed)
            add(f"\n놓친 항목 id: {ids}")

    crosstab("human_상품일치", "상품불일치")
    crosstab("human_환각", "환각")
    crosstab("human_수치정확", "수치오류")

    # 제외 목록
    if skipped:
        add("\n## 5. 측정 제외 항목\n")
        add("| id | 사유 |")
        add("|---|---|")
        for s in skipped:
            add(f"| {s['id']} | {s['skip_reason']} |")

    add("\n## 6. 비용 및 캐시\n")
    add(f"- 캐시 적중 {judge.cache.hits}회 / 신규 호출 {judge.cache.misses}회")
    add(f"- 토큰: prompt {judge.prompt_tokens:,} / "
        f"completion {judge.completion_tokens:,}")
    add("- 재실행 시 캐시가 모두 적중하므로 추가 비용은 발생하지 않는다.")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 표준 지표 병행 측정")
    parser.add_argument("--input", default="data/interim/g_eval_answers.json",
                        help="입력 JSON (프로젝트 루트 기준)")
    parser.add_argument("--embed", choices=["local", "openai"], default="local",
                        help="answer relevancy 임베딩 백엔드 (기본 local)")
    parser.add_argument("--model", default=CHAT_MODEL, help="판정 모델")
    parser.add_argument("--dry-run", action="store_true",
                        help="대상과 예상 호출 수만 확인")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="확인 프롬프트 생략")
    args = parser.parse_args()

    root = find_project_root()
    load_env(root)
    print(f"[1/5] 프로젝트 루트: {root}")

    source = root / args.input
    if not source.exists():
        sys.exit(f"[중단] 입력 파일이 없습니다: {source}")

    samples = load_samples(source)
    print(f"[2/5] 입력 {len(samples)}건 적재: {source.name}")

    notes = join_labels(samples, root)
    for note in notes:
        print(f"      {note}")

    targets, skipped = [], []
    for s in samples:
        ok, reason = is_measurable(s)
        if ok:
            targets.append(s)
        else:
            s["skip_reason"] = reason
            skipped.append(s)

    print(f"[3/5] 측정 대상 {len(targets)}건 / 제외 {len(skipped)}건")
    for s in skipped:
        print(f"      - id {s['id']}: {s['skip_reason']}")

    if not targets:
        sys.exit("[중단] 측정 가능한 항목이 없습니다. "
                 "거절 응답만 있거나 근거가 저장되지 않았습니다.")

    est_calls = len(targets) * 3
    print(f"      예상 LLM 호출: 최대 {est_calls}회 "
          f"(캐시 적중분은 제외됨, 항목당 진술추출 1 + 검증 1 + 역질문 1)")

    if args.dry_run:
        print("[dry-run] 실제 호출 없이 종료합니다.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[중단] OPENAI_API_KEY 를 찾지 못했습니다 (.env 확인).")

    if not args.yes:
        answer = input("측정을 진행할까요? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("취소했습니다.")
            return

    cache_path = (root / "data" / "interim"
                  / f"ragas_cache_{args.model.replace('.', '')}.json")
    cache = JudgeCache(cache_path)
    judge = RagasJudge(cache, model=args.model)

    print(f"[4/5] 임베딩 백엔드 준비 ({args.embed}) …")
    embedder = Embedder(args.embed)

    print("[5/5] 측정 시작")
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(targets, start=1):
        print(f"      ({i}/{len(targets)}) id {s['id']}", flush=True)
        try:
            score, supported, judged, verdicts = judge.faithfulness(
                s["question"], s["answer"], s["contexts"])
            questions, noncommittal = judge.generate_questions(s["answer"])
            relevancy = answer_relevancy(
                embedder, s["question"], questions, noncommittal)
            error = ""
        except Exception as exc:  # 한 건 실패가 전체를 막지 않게 한다
            score, supported, judged, verdicts = None, 0, 0, []
            relevancy, noncommittal, error = None, False, str(exc)[:200]
            print(f"        [오류] {error}")

        row = dict(s)
        row.pop("contexts", None)
        row.update({
            "ragas_faithfulness": score,
            "supported": supported,
            "judged": judged,
            "ragas_answer_relevancy": relevancy,
            "noncommittal": noncommittal,
            "unsupported_reasons": " | ".join(
                str(v.get("reason", ""))[:80] for v in verdicts
                if v.get("supported") is False),
            "error": error,
        })
        rows.append(row)
        cache.save()  # 중단되어도 진행분 보존

    cache.save()

    # 출력
    stamp = f"{datetime.now():%Y-%m-%d}"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    csv_path = reports / f"ragas_measure_{stamp}.csv"
    fields = ["id", "group", "difficulty", "doc_type", "question", "decision",
              "ragas_faithfulness", "supported", "judged",
              "ragas_answer_relevancy", "noncommittal",
              "human_환각", "human_상품일치", "human_수치정확",
              "v2_환각", "v2_상품일치", "v2_수치정확",
              "unsupported_reasons", "error"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = build_report(rows, skipped, notes, judge, source)
    md_path = reports / f"ragas_measure_{stamp}.md"
    md_path.write_text(report, encoding="utf-8")

    print()
    print(report)
    print()
    print("=" * 60)
    print(f"항목별 CSV : {csv_path}")
    print(f"요약 리포트: {md_path}")
    print(f"판정 캐시  : {cache_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
