"""52 — 정상 초안에 통제된 오류를 '규칙으로' 주입한다 (오류 주입 평가셋 v0).

목적
----
실제 직원 오답 데이터가 없으므로, 사람이 정상으로 확인한 초안에 오류를 하나씩
넣어 정답 라벨을 '구성으로' 확보한다. 그러면 검증기 탐지율을 LLM 판사 없이 잴 수 있다.

이 스크립트가 만드는 유형 (LLM 없음, 결정적)
----------------------------------------
    NUMERIC       수치 변경    value(값) / unit(%↔%p) / comparator(이상↔초과, 이하↔미만) / period(연↔월)
    PRODUCT       상품명 치환  같은 카탈로그 범주의 다른 상품 (hard negative)
    INSTITUTION   기관명 치환  하나은행 → 타행 (replace) / 없으면 문두에 삽입 (insert)
    OVERSTATED    단정 강화    강조어 삽입(intensifier) / '…될 수 있습니다' → '…됩니다'(hedge_drop)
    NONE          대조군       원본 그대로 (Clean False Alarm 용)

53번(예정)이 만드는 유형 — 의미 판단이 필요해 LLM 제안 + 사람 확정
    MISSING_CONDITION (claim 내 조건 삭제), NEEDS_CUSTOMER_DATA (확인 필요 항목의 확정화),
    OMITTED_REQUIRED_INFO (조건 문장 통째 삭제)

순환 측정을 피하는 장치
--------------------
주입기와 탐지기가 같은 단어 목록을 쓰면 '목록이 목록을 찾는다'를 재는 셈이다.
그래서 강조어 삽입 어휘(INJECT_INTENSIFIERS)에 탐지기 목록(rules.INTENSIFIERS)에
없는 표현을 일부러 섞었다('예외 없이', '언제나', '100프로'). 이 하위 유형의 탐지율이
낮게 나오는 것은 결함 은폐가 아니라 의도한 측정이다. 각 샘플의
`in_detector_vocab` 열로 나눠 보고한다.

유효성 검사 (주입이 정말 오류인지)
------------------------------
    수치 value  : 바꾼 값을 근거 청크 전체의 같은 단위 수치와 대조해
                  numeric.compare_slot 이 VALUE_MISMATCH 일 때만 채택
                  (근거 범위 안이나 근거의 다른 값과 같아지면 오류가 아니므로 폐기)
    상품        : ProductMatcher.same_product 가 False 인 상품으로만 치환
    그 외       : 원문에서 실제로 문자열이 바뀌었는지 확인
폐기 사유는 콘솔과 manifest 에 집계한다.

입출력
-----
    입력  data/eval/v2/clean_drafts_v1.csv     (50번 산출 + 사람 검수 열)
          data/interim/g_full_answers.json     (근거 청크 원문)
          data/interim/product_catalog.json
    출력  data/eval/v2/injected_v0.csv          샘플당 1행
          data/eval/v2/injected_v0_manifest.json

사람 검수 상태
------------
human_is_correct 또는 human_style_ok 가 'N' 인 초안은 제외한다.
둘 다 비어 있으면 포함하되 label_status=provisional 로 표시한다.
검수가 끝난 뒤 다시 실행하면 같은 시드로 같은 샘플이 재생성되고 상태만 confirmed 로 바뀐다.

실행
----
    python scripts/52_inject_errors.py
    python scripts/52_inject_errors.py --seed 2026 --show 3     # 유형별 예시 3개 출력
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.product_catalog import ProductCatalog, normalize  # noqa: E402
from finguide_rag.generation.product_match import ProductMatcher  # noqa: E402
from finguide_rag.verification.numeric import (  # noqa: E402
    MatchStatus,
    compare_slot,
    extract_numerics,
)
from finguide_rag.verification.rules import INTENSIFIERS  # noqa: E402

DRAFTS = PROJECT_ROOT / "data" / "eval" / "v2" / "clean_drafts_v1.csv"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
CATALOG = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"
OUT = PROJECT_ROOT / "data" / "eval" / "v2" / "injected_v0.csv"
MANIFEST = PROJECT_ROOT / "data" / "eval" / "v2" / "injected_v0_manifest.json"

GENERATOR_VERSION = "inject-rule-v0"

# 강조어 삽입 어휘 — 일부러 탐지기 목록 밖의 표현을 섞는다 (모듈 docstring)
INJECT_INTENSIFIERS = ("무조건", "항상", "예외 없이", "언제나", "100프로")
OTHER_BANKS = ("신한은행", "국민은행", "우리은행", "농협은행", "카카오뱅크")
COMPARATOR_FLIP = {"이상": "초과", "초과": "이상", "이하": "미만", "미만": "이하", "이내": "미만"}

EXPECT = {
    # error_type: (expected_codes, expected_min_verdict)
    "NUMERIC": (["NUMERIC_MISMATCH"], "PROHIBITED"),
    "PRODUCT": (["PRODUCT_MISMATCH"], "PROHIBITED"),
    "INSTITUTION": (["INSTITUTION_MISMATCH"], "PROHIBITED"),
    "OVERSTATED": (["OVERSTATED"], "CHECK"),
    "NONE": ([], "OK"),
}


@dataclass
class Sample:
    sample_id: str
    base_id: str
    query_id: str
    question: str
    doc_type: str
    error_type: str
    subtype: str
    text: str
    original_fragment: str
    injected_fragment: str
    span_start: int
    span_end: int
    expected_codes: str
    expected_min_verdict: str
    rule_judgeable: bool        # 규칙만으로 판정 가능한 구조인가 (예: 약관 근거 상품 치환은 False)
    in_detector_vocab: str      # OVERSTATED/intensifier 전용: 삽입어가 탐지기 목록에 있는가
    evidence_chunk_ids: str
    label_status: str
    generator: str = GENERATOR_VERSION


# ==================================================================
# 유틸
# ==================================================================


def read_csv_any(path: Path) -> list[dict]:
    for enc in ("utf-8-sig", "cp949"):
        try:
            with path.open(encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    sys.exit(f"{path.name} 를 읽지 못했습니다. 'CSV UTF-8' 로 다시 저장하세요.")


def sid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


_JOSA_PAIRS = {"을": ("을", "를"), "를": ("을", "를"), "이": ("이", "가"), "가": ("이", "가"),
               "은": ("은", "는"), "는": ("은", "는"), "과": ("과", "와"), "와": ("과", "와")}


def _has_batchim(word: str) -> bool:
    ch = word.rstrip()[-1:] if word.strip() else ""
    code = ord(ch) - 0xAC00 if ch else -1
    if 0 <= code < 11172:
        return code % 28 != 0
    return ch.isdigit() and ch in "0136789"   # 영·일·삼·육·칠·팔·구 (받침 있음)


def splice(text: str, start: int, end: int, new: str) -> tuple[str, int, int]:
    """치환하고, 바로 뒤 조사를 새 단어 받침에 맞춘다 ('예금 토큰을' → '화상상담데스크를')."""
    tail = text[end:]
    # 조사 한 글자 뒤가 한글이 아닐 때만 조사로 본다 ('토큰이며'의 '이'는 서술격이라 건드리지 않음)
    is_josa = tail[:1] in _JOSA_PAIRS and not ("가" <= tail[1:2] <= "힣")
    if new.strip() and not new.endswith(" ") and is_josa:
        with_b, without_b = _JOSA_PAIRS[tail[0]]
        tail = (with_b if _has_batchim(new) else without_b) + tail[1:]
    return text[:start] + new + tail, start, start + len(new)


_NUM_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _fmt_like(orig: str, value: float) -> str:
    """원래 토큰과 같은 형식(쉼표·소수 자리)으로 숫자를 쓴다."""
    decimals = len(orig.split(".")[1]) if "." in orig else 0
    s = f"{value:,.{decimals}f}" if "," in orig else f"{value:.{decimals}f}"
    return s


# ---------- 한글 받침 조작 (헤지 삭제용) ----------
_HANGUL_BASE, _JONG_RIEUL, _JONG_BIEUP = 0xAC00, 8, 17


def _set_jong(ch: str, jong: int) -> str | None:
    code = ord(ch) - _HANGUL_BASE
    if not (0 <= code < 11172):
        return None
    return chr(_HANGUL_BASE + (code // 28) * 28 + jong)


_HEDGE_RE = re.compile(r"(\S)\s*수\s*(?:도\s*)?있습니다")
_EVENT_STEMS = ("될", "생길", "발생할", "있을", "달라질", "변동될", "변경될", "부과될", "제한될",
                "않을", "감소할", "지연될", "하락할", "상승할", "거절될", "취소될", "해지될")


def drop_hedge(text: str) -> tuple[str, str, str, int] | None:
    """'부과될 수 있습니다' → '부과됩니다', '있을 수 있습니다' → '있습니다'.

    사건 동사(EVENT_STEMS)로 끝나는 가능성 표현만 바꾼다. 탐지기도 같은 범위만
    헤지로 보므로 이 하위 유형은 '탐지기 설계 범위 안'의 측정이다.
    """
    for m in _HEDGE_RE.finditer(text):
        before = text[: m.start(1) + 1]
        word = re.search(r"(\S+)$", before).group(1)
        if not any(word.endswith(s) for s in _EVENT_STEMS):
            continue
        last = m.group(1)
        if last == "을":                       # 있을 수 있습니다 → 있습니다
            new_tail, start = "습니다", m.start(1)
        elif ord(last) - _HANGUL_BASE >= 0 and (ord(last) - _HANGUL_BASE) % 28 == _JONG_RIEUL:
            new_tail, start = _set_jong(last, _JONG_BIEUP) + "니다", m.start(1)
        else:
            continue
        return text[start:m.end()], new_tail, start, m.end()
    return None


# ==================================================================
# 주입기
# ==================================================================


class Injector:
    def __init__(self, catalog_path: Path, seed: int):
        self.matcher = ProductMatcher.from_catalog(catalog_path)
        self.catalog = ProductCatalog.load(catalog_path)
        self.seed = seed
        self.dropped: Counter = Counter()

    # ---------- 수치 ----------
    def numeric(self, text: str, evidence: str) -> list[tuple]:
        slots = [s for s in extract_numerics(text) if s.unit != "date"]
        ev_slots = extract_numerics(evidence)
        out: dict[str, tuple] = {}
        for s in slots:
            a, b = s.span
            # value: 첫 숫자 토큰을 바꾸고, 근거와 대조해 '확실히 틀린 값'일 때만 채택
            if "value" not in out and not s.is_range:
                tok = _NUM_TOKEN.search(text, a, b)
                if tok:
                    v = float(tok.group(0).replace(",", ""))
                    cands = [v + 0.5, v + 0.3, v * 2] if "." in tok.group(0) or s.unit in ("%", "%p") \
                        else [v * 2, v + 1, v * 3]
                    for c in cands:
                        new_raw = text[a:tok.start()] + _fmt_like(tok.group(0), c) + text[tok.end():b]
                        new_slots = extract_numerics(new_raw)
                        if not new_slots:
                            continue
                        st, _ = compare_slot(new_slots[0], ev_slots)
                        if st == MatchStatus.VALUE_MISMATCH:
                            out["value"] = (text[a:b], new_raw, a, b)
                            break
                    else:
                        self.dropped["NUMERIC/value: 바꾼 값이 근거와 겹침"] += 1
            # unit: % ↔ %p
            if "unit" not in out and s.unit in ("%", "%p"):
                raw = text[a:b]
                new = raw.replace("%p", "%") if "%p" in raw else raw.replace("%", "%p", 1)
                if new != raw:
                    out["unit"] = (raw, new, a, b)
            # comparator: 수치 바로 뒤의 비교어
            if "comparator" not in out:
                m = re.match(r"\s*(을|를|이|가)?\s*(이상|초과|이하|미만|이내)", text[b:b + 8])
                if m:
                    w = m.group(2)
                    ws = b + m.start(2)
                    out["comparator"] = (w, COMPARATOR_FLIP[w], ws, ws + len(w))
            # period: 금리 앞의 '연'/'월'
            if "period" not in out and s.unit in ("%", "%p") and s.period:
                m = re.search(rf"(?<![가-힣])({s.period})(?=\s*\d|이율|금리)", text[max(0, a - 10):a])
                if m:
                    ps = max(0, a - 10) + m.start(1)
                    out["period"] = (s.period, "월" if s.period == "연" else "연", ps, ps + 1)
        return [(k, *v) for k, v in out.items()]

    # ---------- 상품 ----------
    def product(self, text: str, doc_type: str, rng: random.Random) -> tuple | None:
        """초안에 나온 '개별 상품명'을 같은 범주의 다른 개별 상품명으로 바꾼다.

        상품군 이름(product_family: 가계대출, ISA 모델포트폴리오 등)은 쓰지 않는다.
        'ISA 모델포트폴리오' ↔ '일임형 개인종합자산관리계좌' 처럼 같은 것을 가리키는
        이름이 섞여 있어, 치환해도 오류가 아닐 수 있기 때문이다 (첫 실행에서 실제 발생).
        변별 어절이 겹치는 형제가 있으면 hard_negative(예: 주택청약종합저축 → 주택청약예금),
        없으면 easy_negative 로 하위 유형을 나눠 따로 보고한다.
        """
        specific = [e for e in self.catalog.covered_products if e.canonical_name]
        named = sorted(specific, key=lambda e: -len(e.canonical_name))
        for e in named:
            name = e.canonical_name
            if name not in text:
                continue
            siblings = [o for o in specific
                        if o.category == e.category and o.canonical_name != name
                        and o.doc_id != e.doc_id
                        and not self.matcher.same_product(name, o.canonical_name)
                        and self.matcher.informative(o.canonical_name)]
            if not siblings:
                self.dropped["PRODUCT: 같은 범주에 다른 개별 상품 없음"] += 1
                return None
            qt = set(self.matcher.informative(name))

            def overlap(o):
                return sum(1 for a in qt for b in self.matcher.informative(o.canonical_name)
                           if a[:2] == b[:2])

            best = max(overlap(o) for o in siblings)
            pool = sorted([o for o in siblings if overlap(o) == best], key=lambda o: o.canonical_name)
            sib = rng.choice(pool).canonical_name
            i = text.index(name)
            sub = "hard_negative" if best > 0 else "easy_negative"
            return (sub, name, sib, i, i + len(name), doc_type == "설명서")
        if any(f.canonical_name and f.canonical_name in text for f in self.catalog.covered_families):
            self.dropped["PRODUCT: 상품군 이름만 있음(치환 제외)"] += 1
        return None

    # ---------- 기관 ----------
    def institution(self, text: str, rng: random.Random) -> tuple:
        bank = rng.choice(OTHER_BANKS)
        i = text.find("하나은행")
        if i >= 0:
            return ("replace", "하나은행", bank, i, i + 4)
        prefix = f"{bank} 기준으로 "
        return ("insert", "", prefix, 0, 0)

    # ---------- 단정 ----------
    _PARTICLE_END = re.compile(r"(을|를|이|가|은|는|도)$")
    # 관형형 어미 '-는'은 조사가 아니다: '제외된다는 점', '이용하는 경우'
    _ADNOMINAL = re.compile(r"(다는|라는|하는|되는|있는|없는|이는|않는|드리는)$")

    def intensifier(self, text: str, rng: random.Random) -> tuple | None:
        """서술어 직전의 목적어·주어 어절 뒤에 강조어를 넣는다.

            '수수료가 부과됩니다'       → '수수료가 무조건 부과됩니다'
            '거래를 시도하셔야 합니다'  → '거래를 무조건 시도하셔야 합니다'

        첫 구현은 마지막 어절 앞에 넣어 '시도하셔야 무조건 합니다'처럼 비문이 됐다.
        비문은 탐지기가 문체 이상으로 잡을 수도 있어 측정을 오염시킨다.
        """
        sents = [m for m in re.finditer(r"(?:[^.!?]|\.(?=\d))+[.!?]?", text) if len(m.group(0).strip()) > 8]
        cands = []
        for m in sents:
            if any(w in m.group(0) for w in INTENSIFIERS):
                continue
            toks = list(re.finditer(r"\S+", m.group(0)))
            marked = [t for t in toks[:-1]
                      if self._PARTICLE_END.search(t.group(0).rstrip(",."))
                      and not self._ADNOMINAL.search(t.group(0).rstrip(",."))]
            if marked:
                cands.append(m.start() + marked[-1].end() + 1)
        if not cands:
            return None
        pos = rng.choice(cands)
        word = rng.choice(INJECT_INTENSIFIERS)
        return ("intensifier", "", word + " ", pos, pos, word)


# ==================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--show", type=int, default=2, help="유형별 예시 출력 수")
    args = ap.parse_args()

    for p in (DRAFTS, ANSWERS, CATALOG):
        if not p.exists():
            sys.exit(f"{p.relative_to(PROJECT_ROOT)} 가 없습니다.")
    drafts = read_csv_any(DRAFTS)
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    inj = Injector(CATALOG, args.seed)

    samples: list[Sample] = []
    excluded: Counter = Counter()

    for d in drafts:
        if d.get("human_is_correct", "").strip().upper() == "N" or \
           d.get("human_style_ok", "").strip().upper() == "N":
            excluded["사람 검수 N"] += 1
            continue
        text = d.get("draft_text", "").strip()
        if not text:
            excluded["draft_text 비어 있음"] += 1
            continue
        confirmed = d.get("human_is_correct", "").strip().upper() == "Y" and \
            d.get("human_style_ok", "").strip().upper() == "Y"
        status = "confirmed" if confirmed else "provisional"
        evid = (answers.get(d["question"]) or {}).get("evidences", [])
        evidence = "\n".join(e.get("text", "") for e in evid)
        rng = random.Random(f"{args.seed}|{d['draft_id']}")

        def add(etype, sub, orig, new, s, e, judgeable=True, vocab=""):
            new_text, ns, ne = splice(text, s, e, new)
            if new_text == text:
                inj.dropped[f"{etype}/{sub}: 변화 없음"] += 1
                return
            codes, verdict = EXPECT[etype]
            samples.append(Sample(
                sample_id=f"{d['draft_id']}_{etype[:3].lower()}_{sub}_{sid(new_text)[:4]}",
                base_id=d["draft_id"], query_id=d.get("query_id", ""), question=d["question"],
                doc_type=d["doc_type"], error_type=etype, subtype=sub, text=new_text,
                original_fragment=orig, injected_fragment=new, span_start=ns, span_end=ne,
                expected_codes="|".join(codes), expected_min_verdict=verdict,
                rule_judgeable=judgeable, in_detector_vocab=vocab,
                evidence_chunk_ids=d.get("evidence_chunk_ids", ""), label_status=status))

        # 대조군
        codes, verdict = EXPECT["NONE"]
        samples.append(Sample(
            sample_id=f"{d['draft_id']}_none", base_id=d["draft_id"], query_id=d.get("query_id", ""),
            question=d["question"], doc_type=d["doc_type"], error_type="NONE", subtype="clean",
            text=text, original_fragment="", injected_fragment="", span_start=-1, span_end=-1,
            expected_codes="", expected_min_verdict=verdict, rule_judgeable=True,
            in_detector_vocab="", evidence_chunk_ids=d.get("evidence_chunk_ids", ""),
            label_status=status))

        for sub, orig, new, s, e in inj.numeric(text, evidence):
            add("NUMERIC", sub, orig, new, s, e)
        p = inj.product(text, d["doc_type"], rng)
        if p:
            sub, orig, new, s, e, judgeable = p
            add("PRODUCT", sub, orig, new, s, e, judgeable=judgeable)
        sub, orig, new, s, e = inj.institution(text, rng)
        add("INSTITUTION", sub, orig, new, s, e)
        it = inj.intensifier(text, rng)
        if it:
            sub, orig, new, s, e, word = it
            add("OVERSTATED", sub, orig, new, s, e,
                vocab="Y" if word.strip() in INTENSIFIERS else "N")
        h = drop_hedge(text)
        if h:
            orig, new, s, e = h
            add("OVERSTATED", "hedge_drop", orig, new, s, e)

    # ---------- 저장 ----------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = list(Sample.__dataclass_fields__)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(asdict(s) for s in samples)

    by_type = Counter(s.error_type for s in samples)
    by_sub = Counter(f"{s.error_type}/{s.subtype}" for s in samples)
    status = Counter(s.label_status for s in samples if s.error_type == "NONE")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generator": GENERATOR_VERSION, "seed": args.seed,
        "input_sha8": hashlib.sha256(DRAFTS.read_bytes()).hexdigest()[:8],
        "n_samples": len(samples), "by_type": dict(by_type), "by_subtype": dict(by_sub),
        "base_drafts": dict(status), "excluded_drafts": dict(excluded),
        "dropped_injections": dict(inj.dropped),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 콘솔 ----------
    print("=" * 64)
    print("오류 주입 평가셋 v0 (규칙 기반)")
    print("=" * 64)
    print(f"기반 초안 {sum(status.values())}건 (confirmed {status['confirmed']} / "
          f"provisional {status['provisional']})  제외 {dict(excluded) or 0}")
    print(f"샘플 {len(samples)}건\n")
    print(f"{'유형/하위유형':<30}{'건수':>5}   규칙 판정 가능")
    groups = defaultdict(list)
    for s in samples:
        groups[f"{s.error_type}/{s.subtype}"].append(s)
    for k in sorted(groups):
        g = groups[k]
        print(f"  {k:<28}{len(g):>5}   {sum(x.rule_judgeable for x in g):>3}")
    iv = Counter(s.in_detector_vocab for s in samples if s.subtype == "intensifier")
    print(f"\n강조어 삽입: 탐지기 목록 안 {iv['Y']}건 / 밖 {iv['N']}건 (밖은 일부러 섞은 것)")
    if inj.dropped:
        print("\n[폐기된 주입]")
        for k, v in inj.dropped.most_common():
            print(f"  {k:<40}{v:>3}")
    if args.show:
        print("\n[예시]")
        for k in sorted(groups):
            if k.startswith("NONE"):
                continue
            for s in groups[k][: args.show]:
                ctx_s = max(0, s.span_start - 15)
                ctx = s.text[ctx_s: s.span_end + 15].replace("\n", " ")
                print(f"  {k:<24} [{s.base_id}] '{s.original_fragment}' → '{s.injected_fragment}'  …{ctx}…")
    print(f"\n→ {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
