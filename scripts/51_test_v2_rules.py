"""51 — v2 claim 규칙 검사 테스트 + 정상 답변 오탐 사전 점검.

두 부분으로 나뉜다.

[1] 단위 테스트 (verification/rules.py)
    E. 단정·헤지   F. 기관   G. 상품(실제 카탈로그 사용)   H. 고객정보·시점·공란·구버전
    I. 묶음 실행(run_claim_rules) + PolicyGate 연결

[2] 정상 답변 오탐 사전 점검 (--no-scan 으로 끌 수 있음)
    v1 생성 답변 중 정상 초안 풀 30건(data/eval/v2/clean_drafts_v0.csv)의 각 문장을
    그 답변이 인용한 근거와 대조해, 규칙이 '정상 문장'에 경고를 얼마나 내는지 본다.
    아직 LLM 정렬(문장 ↔ 근거 span)이 없으므로 두 가지 근거로 각각 돌린다.
      chunk  근거 청크 전체 — 헤지 소실은 과대, 강조어 추가는 과소 추정
      proxy  청크 문장 중 claim 과 글자 2-gram 이 가장 많이 겹치는 1~2문장
             — LLM 정렬의 값싼 대용품. 실제 파이프라인에 더 가깝다
    v1 자동 평가로 '정상'이라 했을 뿐 사람 검수 전이므로, 걸린 문장 일부는 실제 문제일 수 있다.
    목적은 규칙 설계의 큰 결함(예: 정상 문장의 절반이 걸림)을 일찍 찾는 것이다.

LLM·API 키 없이 돈다.

실행
----
    python scripts/51_test_v2_rules.py
    python scripts/51_test_v2_rules.py --verbose     # 사전 점검에서 걸린 문장 전부 출력
    python scripts/51_test_v2_rules.py --no-scan
    python scripts/51_test_v2_rules.py --scan-set test   # 정상 초안 풀 30건(평가용)으로 점검

dev / test 분리
--------------
사전 점검 결과를 보고 규칙을 고치면, 그 데이터는 더 이상 공정한 평가셋이 아니다.
그래서 기본 점검 대상은 'dev' = 적격 후보 73건 중 정상 초안 풀 30건을 뺀 43건이다.
규칙 튜닝은 dev 로만 하고, 풀 30건(test)은 오류 주입 평가에서 Clean False Alarm 을
잴 때 쓴다. (규칙 v0.3 은 개발 중 test 30건을 보고 두 번 조정됐다 — README 한계에 기록)

산출물
-----
    reports/v2_rules_test_<날짜>.md
    reports/v2_rules_clean_scan_<날짜>.csv   (사전 점검 문장별 결과)
종료 코드: 단위 테스트 실패가 있으면 1 (사전 점검 수치는 종료 코드에 영향 없음)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.generation.product_match import ProductMatcher  # noqa: E402
from finguide_rag.verification.policy import build_verification  # noqa: E402
from finguide_rag.verification.rules import (  # noqa: E402
    assertiveness_of,
    check_assertiveness,
    check_blank_value,
    check_customer_data,
    check_institution,
    check_outdated,
    check_product,
    check_time_variant,
    detect_institutions,
    run_claim_rules,
)
from finguide_rag.verification.schema import (  # noqa: E402
    CatalogStatus,
    EvidenceRef,
    ReasonCode,
    Verdict,
)

CATALOG = PROJECT_ROOT / "data" / "interim" / "product_catalog.json"
ANSWERS = PROJECT_ROOT / "data" / "interim" / "g_full_answers.json"
CLEAN_SHEET = PROJECT_ROOT / "data" / "eval" / "v2" / "clean_drafts_v0.csv"
CANDIDATES = PROJECT_ROOT / "data" / "eval" / "v2" / "clean_pool_candidates.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

MATCHER = ProductMatcher.from_catalog(CATALOG)

CASES: list[tuple[str, str, callable]] = []


def case(group: str, name: str):
    def deco(fn):
        CASES.append((group, name, fn))
        return fn
    return deco


def EV(quote="", doc_name="", doc_type="설명서", institution="하나은행", is_latest=True):
    return EvidenceRef(doc_id="d", chunk_id="d_c001", quote=quote, is_latest=is_latest,
                       doc_name=doc_name, doc_type=doc_type, institution=institution)


def codes(hits):
    return [h.code for h in hits]


# ==================================================================
# E. 단정 · 헤지
# ==================================================================

ASSERT = [
    # (이름, claim, 근거, 기대 OVERSTATED 여부)
    ("강조어 추가 '그대로'", "중도해지해도 우대금리는 그대로 적용됩니다",
     "중도해지 시 우대금리는 적용하지 않습니다", True),
    ("강조어 추가 '무조건'", "무조건 비과세 혜택을 받습니다", "요건 충족 시 비과세 혜택", True),
    ("근거에도 있는 '반드시'는 정상", "신분증을 반드시 지참하셔야 합니다",
     "반드시 본인 신분증을 지참하여야 합니다", False),
    ("헤지 소실 '수 있'", "해지하면 원금 손실이 발생합니다",
     "중도해지 시 원금 손실이 발생할 수 있습니다", True),
    ("헤지 소실 '에 한해'", "우대금리가 적용됩니다",
     "급여이체 실적이 있는 경우에 한해 우대금리를 적용합니다", True),
    ("단서('다만')는 규칙이 아닌 LLM(MISSING_CONDITION) 담당 → 규칙 통과", "타행 이체 수수료는 면제됩니다",
     "타행 이체 수수료는 면제됩니다. 다만 창구 이용 시 제외합니다", False),
    ("동등 표현: 근거 '지참하여야' → claim '반드시 지참'은 정상", "신분증을 반드시 지참하셔야 합니다",
     "본인 신분증을 지참하여야 합니다", False),
    ("표현을 바꾼 한정은 통과", "조건을 충족하신 경우 우대금리가 적용됩니다",
     "급여이체 실적이 있는 경우에 한해 우대금리를 적용합니다", False),
    ("헤지 없는 근거, 헤지 없는 claim", "만기 시 이자가 지급됩니다", "만기일에 이자를 지급합니다", False),
    ("능력·방법 안내 '할 수 있습니다'는 헤지 아님", "인터넷뱅킹에서 등록하시면 됩니다",
     "인터넷뱅킹에서 등록할 수 있습니다", False),
    ("'변동될 수 있' 소실은 탐지", "평가손익이 변동됩니다", "환율에 따라 평가손익이 변동될 수 있습니다", True),
]
for n, c, e, want in ASSERT:
    def _mk(c=c, e=e, want=want):
        def run():
            hits, chk = check_assertiveness(c, e)
            got = ReasonCode.OVERSTATED in codes(hits)
            return got == want, f"{chk} {[h.detail for h in hits]}"
        return run
    CASES.append(("E.단정·헤지", n, _mk()))


@case("E.단정·헤지", "assertiveness_of: strong / hedged / neutral")
def _():
    got = (assertiveness_of("무조건 됩니다"), assertiveness_of("받을 수 있습니다"),
           assertiveness_of("지급됩니다"))
    return got == ("strong", "hedged", "neutral"), str(got)


# ==================================================================
# F. 기관
# ==================================================================

@case("F.기관", "별칭 정규화: KB국민은행 → 국민은행 (국민 이중 매칭 없음)")
def _():
    got = detect_institutions("KB국민은행과 하나은행")
    return got == {"국민은행", "하나은행"}, str(sorted(got))


@case("F.기관", "타 기관 언급 → INSTITUTION_MISMATCH")
def _():
    hits, chk = check_institution("신한은행 기준으로 연 3.0%입니다", [EV("연 3.0%")])
    return codes(hits) == [ReasonCode.INSTITUTION_MISMATCH], chk


@case("F.기관", "같은 기관 언급 → pass")
def _():
    hits, chk = check_institution("하나은행에서는 연 3.0%입니다", [EV("연 3.0%")])
    return not hits and chk == "pass", chk


@case("F.기관", "기관 언급 없음 → n/a")
def _():
    hits, chk = check_institution("연 3.0%입니다", [EV("연 3.0%")])
    return not hits and chk == "n/a", chk


@case("F.기관", "'우리' 같은 일반어는 기관으로 보지 않음")
def _():
    got = detect_institutions("우리 은행 창구에서 신청하시면 됩니다")
    return got == set(), str(got)


# ==================================================================
# G. 상품 (실제 카탈로그)
# ==================================================================

DESC_JUTAEK = "『주택청약종합저축』 상품설명서"
DESC_YEGEUM = "『주택청약예금』 요약 상품설명서"


@case("G.상품", "같은 상품 → pass")
def _():
    hits, chk = check_product("주택청약종합저축", [EV(doc_name=DESC_JUTAEK)], MATCHER)
    return not hits and chk == "pass", chk


@case("G.상품", "유사 상품명 hard negative: 주택청약예금 ↔ 주택청약종합저축 → PRODUCT_MISMATCH")
def _():
    hits, chk = check_product("주택청약예금", [EV(doc_name=DESC_JUTAEK)], MATCHER)
    return codes(hits) == [ReasonCode.PRODUCT_MISMATCH], f"{chk} {[h.detail for h in hits]}"


@case("G.상품", "v1 실패 사례: 정기예금 ↔ 주택청약예금 설명서 → PRODUCT_MISMATCH")
def _():
    hits, chk = check_product("정기예금", [EV(doc_name=DESC_YEGEUM)], MATCHER)
    return codes(hits) == [ReasonCode.PRODUCT_MISMATCH], chk


@case("G.상품", "근거가 약관·FAQ 뿐 → n/a (이름 대조 불가)")
def _():
    ev = [EV(doc_name="입출금이 자유로운 예금약관", doc_type="약관"),
          EV(doc_name="미성년자가 현금IC카드를 발급할 수 있나요?", doc_type="FAQ")]
    hits, chk = check_product("자유적금", ev, MATCHER)
    return not hits and chk == "n/a", chk


@case("G.상품", "설명서 여럿 중 하나라도 같으면 pass")
def _():
    hits, chk = check_product("주택청약예금", [EV(doc_name=DESC_JUTAEK), EV(doc_name=DESC_YEGEUM)], MATCHER)
    return not hits, chk


@case("G.상품", "변별력 없는 이름('하나은행 서비스') → n/a")
def _():
    hits, chk = check_product("하나은행 서비스", [EV(doc_name=DESC_JUTAEK)], MATCHER)
    return not hits and chk == "n/a", chk


KNOWN = ["주택청약종합저축", "주택청약예금", "하나 화상상담데스크", "장외파생"]


@case("G.상품", "카탈로그에 없는 상품명('외화보통예금통장') → n/a (불일치 단정 안 함)")
def _():
    hits, chk = check_product("외화보통예금통장", [EV(doc_name="『고객센터 환전지갑 서비스 설명서』")],
                              MATCHER, known_products=KNOWN)
    return not hits and chk == "n/a", chk


@case("G.상품", "근거 본문에 상품명이 직접 나옴(상품군 문서) → pass")
def _():
    ev = [EV("< 상품명 : Range Forward(고객이 조건부로 외화매수) >", doc_name="장외파생상품설명서")]
    hits, chk = check_product("Range Forward 상품", ev, MATCHER, known_products=KNOWN + ["Range Forward"])
    return not hits and chk == "pass", chk


@case("G.상품", "카탈로그 상품끼리 불일치는 여전히 PRODUCT_MISMATCH")
def _():
    hits, chk = check_product("주택청약예금", [EV("연 3.0%", doc_name=DESC_JUTAEK)], MATCHER, known_products=KNOWN)
    return codes(hits) == [ReasonCode.PRODUCT_MISMATCH], chk


@case("G.상품", "catalog ambiguous → AMBIGUOUS_PRODUCT")
def _():
    hits, _ = check_product("주택청약", [EV(doc_name=DESC_JUTAEK)], MATCHER, CatalogStatus.AMBIGUOUS)
    return codes(hits) == [ReasonCode.AMBIGUOUS_PRODUCT], str(codes(hits))


# ==================================================================
# H. 고객정보 · 시점 · 공란 · 구버전
# ==================================================================

CUST = [
    ("확정형 '받으실 수 있'", "고객님은 우대금리를 받으실 수 있습니다", True),
    ("고객 속성 '신용등급'", "고객님 신용등급이면 대출이 가능합니다", True),
    ("고객 속성 '거래실적'", "고객님의 거래실적으로 우대 대상입니다", True),
    ("일반 규정 존댓말은 통과", "중도해지 시에는 중도해지이율이 적용됩니다", False),
    ("일반 조건 설명은 통과", "급여이체 실적이 있으면 우대금리를 받을 수 있습니다", False),
]
for n, c, want in CUST:
    def _mk(c=c, want=want):
        def run():
            hits, flag = check_customer_data(c)
            return flag == want, str([h.detail for h in hits])
        return run
    CASES.append(("H.고객·시점·공란", n, _mk()))


@case("H.고객·시점·공란", "시점 의존 '현재' → TIME_VARIANT (v1 패턴)")
def _():
    return codes(check_time_variant("현재 금리는 연 3.0%입니다")) == [ReasonCode.TIME_VARIANT], ""


@case("H.고객·시점·공란", "'다음달까지' 같은 문서상 기간은 시점 의존 아님 (v1 결정 유지)")
def _():
    return check_time_variant("가입 후 다음달까지 수수료가 면제됩니다") == [], ""


@case("H.고객·시점·공란", "'최근 3개월 이내 발급'은 문서상 기간 → 통과")
def _():
    return check_time_variant("최근 3개월 이내 발급분만 가능합니다") == [], ""


@case("H.고객·시점·공란", "원문에 이미 있는 '현재'를 옮긴 것 → 통과")
def _():
    return check_time_variant("현재 6개국에서 이용 가능합니다", "현재 6개국에서 서비스 중") == [], ""


@case("H.고객·시점·공란", "근거 공란 + 금리 확정 → BLANK_VALUE")
def _():
    hits = check_blank_value("연체가산율은 연 3%입니다", [EV("연체가산율 : 연 (   )%")])
    return codes(hits) == [ReasonCode.BLANK_VALUE], str([h.detail for h in hits])


@case("H.고객·시점·공란", "근거 공란이어도 수치 없는 claim → 통과")
def _():
    return check_blank_value("연체가산율은 계약서에 따릅니다", [EV("연 (   )%")]) == [], ""


@case("H.고객·시점·공란", "구버전 근거 → OUTDATED_SOURCE")
def _():
    return codes(check_outdated([EV(is_latest=False)])) == [ReasonCode.OUTDATED_SOURCE], ""


# ==================================================================
# I. 묶음 실행 + PolicyGate
# ==================================================================

@case("I.묶음+정책", "기획 예시: '중도해지해도 우대금리는 그대로' + LLM SUPPORTED → CHECK 이상, 충돌 기록")
def _():
    ev = [EV("중도해지 시에는 약정이율이 아닌 중도해지이율을 적용하며, 우대금리는 적용하지 않을 수 있습니다",
             doc_name="『하나 자유적금』 상품설명서")]
    r = run_claim_rules("중도해지해도 우대금리는 그대로 적용됩니다", ev,
                        claim_product=None, matcher=MATCHER)
    v = build_verification("c1", ["SUPPORTED"], r.hits, rule_checks=r.rule_checks)
    ok = v.verdict.severity >= Verdict.CHECK.severity and v.llm_rule_conflict
    return ok, f"{v.verdict.value} conflict={v.llm_rule_conflict} {r.rule_checks}"


@case("I.묶음+정책", "상품 불일치 + 수치 일치 → PROHIBITED")
def _():
    r = run_claim_rules("주택청약예금은 연 3.0%입니다", [EV("연 3.0%", doc_name=DESC_JUTAEK)],
                        claim_product="주택청약예금", matcher=MATCHER)
    v = build_verification("c1", ["SUPPORTED"], r.hits)
    return v.verdict == Verdict.PROHIBITED, f"{v.verdict.value} {r.rule_checks}"


@case("I.묶음+정책", "깨끗한 claim → 규칙 hit 없음, OK")
def _():
    r = run_claim_rules("만기일에 원금과 이자를 지급합니다",
                        [EV("만기일에 원금과 이자를 지급합니다", doc_name=DESC_JUTAEK)],
                        claim_product="주택청약종합저축", matcher=MATCHER)
    v = build_verification("c1", ["SUPPORTED"], r.hits)
    return v.verdict == Verdict.OK and not r.hits, f"{v.verdict.value} {r.rule_checks}"


@case("I.묶음+정책", "rule_checks 에 8개 검사 키가 모두 있음")
def _():
    r = run_claim_rules("연 3.0%", [EV("연 3.0%")], claim_product=None, matcher=MATCHER)
    want = {"numeric", "assertiveness", "institution", "product", "customer_data",
            "time_variant", "blank_value", "outdated"}
    return set(r.rule_checks) == want, str(sorted(r.rule_checks))


# ==================================================================
# J. 초안 수준 범위 검사 (scope.py)
# ==================================================================

from finguide_rag.verification.schema import Claim, Verification  # noqa: E402
from finguide_rag.verification.scope import DocIndex, DocInfo, check_draft_scope  # noqa: E402

_IDX = DocIndex({
    "dA": DocInfo("dA", "설명서", "『주택청약종합저축』 상품설명서", "주택청약종합저축", False),
    "dB": DocInfo("dB", "설명서", "『해외 ATM 출금 서비스』 설명서", "해외 ATM 출금 서비스", False),
    "dG": DocInfo("dG", "약관", "예금거래 기본약관", "", True),
    "fq": DocInfo("fq", "FAQ", "국가간 ATM 서비스란?", "", False),
    "dE": DocInfo("dE", "설명서", "『easy-one 외화송금서비스』 상품설명서", "easy-one 외화송금서비스", False),
})


def _cv(*pairs):
    claims, vers = [], []
    for i, (text, docs) in enumerate(pairs, 1):
        cid = f"c{i}"
        claims.append(Claim(cid, text, (0, len(text))))
        vers.append(Verification(cid, Verdict.OK, [], evidence=[
            EvidenceRef(d, d + "_c1", q) for d, q in docs]))
    return claims, vers


@case("J.범위", "서로 다른 상품 설명서 근거 2개 → MIXED_SCOPE")
def _():
    c, v = _cv(("청약은 24회 납입", [("dA", "24회")]), ("해외 출금은 앱에서", [("dB", "앱")]))
    hits = check_draft_scope("청약이랑 해외출금 알려줘", c, v, _IDX, MATCHER)
    return [h.code for h in hits][:1] == [ReasonCode.MIXED_SCOPE], str([h.detail for h in hits])


@case("J.범위", "상품 1개 + 그 상품을 언급하지 않는 FAQ → MIXED_SCOPE")
def _():
    c, v = _cv(("해외 ATM 출금 서비스는 앱 가입", [("dB", "앱")]), ("6개국 가능", [("fq", "국가간 ATM 6개국")]))
    hits = check_draft_scope("해외 ATM 출금 서비스 어떻게 써요", c, v, _IDX, MATCHER)
    return ReasonCode.MIXED_SCOPE in [h.code for h in hits], str([h.detail for h in hits])


@case("J.범위", "일반 약관 + 상품 1개 (질문이 그 상품 언급) → 경고 없음")
def _():
    c, v = _cv(("영업점에서 거래", [("dG", "영업점")]), ("주택청약종합저축은 24회", [("dA", "24회")]))
    hits = check_draft_scope("주택청약종합저축 납입 횟수", c, v, _IDX, MATCHER)
    return hits == [], str([h.detail for h in hits])


@case("J.범위", "일반 질문에 특정 상품명으로 답함 → SCOPE_MISMATCH")
def _():
    c, v = _cv(("easy-one 외화송금서비스는 취소 불가", [("dE", "취소 불가")]),)
    hits = check_draft_scope("외화 송금한 후에 취소할 수 있나요?", c, v, _IDX, MATCHER)
    return [h.code for h in hits] == [ReasonCode.SCOPE_MISMATCH], str([h.detail for h in hits])


@case("J.범위", "질문이 '어떤 서비스'를 물으면 상품명 답변은 축소 아님")
def _():
    c, v = _cv(("easy-one 외화송금서비스가 있습니다", [("dE", "송금")]),)
    hits = check_draft_scope("외화 송금은 어떤 서비스가 있나요?", c, v, _IDX, MATCHER)
    return hits == [], str([h.detail for h in hits])


# ==================================================================
# [2] 정상 답변 오탐 사전 점검
# ==================================================================

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CHUNK_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+|\s(?=\d+\.\s)|\s(?=[·•\-]\s)")


def _bigrams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def proxy_span(claim: str, chunks: list[str], k: int = 2) -> str:
    """claim 과 2-gram 이 가장 많이 겹치는 청크 문장 k개 (LLM 정렬 대용)."""
    cb = _bigrams(claim)
    sents = [s for c in chunks for s in _CHUNK_SPLIT.split(c) if len(s.strip()) > 4]
    scored = sorted(sents, key=lambda s: len(cb & _bigrams(s)) / (len(cb) or 1), reverse=True)
    return "\n".join(scored[:k])


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean_scan(verbose: bool, scan_set: str) -> dict | None:
    if not (CLEAN_SHEET.exists() and ANSWERS.exists() and CANDIDATES.exists()):
        print("\n[사전 점검] 입력 파일이 없어 건너뜀 (49번을 먼저 실행)")
        return None
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    pool = _read(CLEAN_SHEET)
    if scan_set == "test":
        sheet = pool
    else:
        picked = {r["question"] for r in pool}
        sheet = [dict(r, draft_id=r.get("query_id") or f"dev{i:03d}")
                 for i, r in enumerate(_read(CANDIDATES))
                 if r["eligible"] == "True" and r["question"] not in picked]

    rows = []
    stats = {m: Counter() for m in ("chunk", "proxy")}
    for s in sheet:
        a = answers.get(s["question"]) or {}
        evs = a.get("evidences", [])
        chunks = [e.get("text", "") for e in evs]
        for sent in [x.strip() for x in _SENT_SPLIT.split(a.get("answer", "")) if x.strip()]:
            row = {"draft_id": s["draft_id"], "sentence": sent}
            for mode in ("chunk", "proxy"):
                quote = "\n".join(chunks) if mode == "chunk" else proxy_span(sent, chunks)
                ev = [EvidenceRef(doc_id="", chunk_id="", quote=quote)]   # 상품 대조 제외
                r = run_claim_rules(sent, ev, claim_product=None, matcher=MATCHER)
                fired = [h.code.value for h in r.hits]
                st = stats[mode]
                st["_n"] += 1
                st["_flagged"] += bool(fired)
                for c in set(fired):
                    st[c] += 1
                row[f"{mode}_codes"] = "|".join(fired)
                row[f"{mode}_details"] = " / ".join(h.detail for h in r.hits)
                if mode == "proxy":
                    row["proxy_span"] = quote
            rows.append(row)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"v2_rules_clean_scan_{scan_set}_{date.today().isoformat()}.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["draft_id", "sentence", "chunk_codes", "proxy_codes",
                                          "proxy_details", "proxy_span", "chunk_details"])
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 64)
    print(f"[사전 점검 · {scan_set} {len(sheet)}건] 정상 답변 문장에 규칙이 낸 경고 (상품 대조 제외)")
    print("=" * 64)
    for mode, label in (("chunk", "근거=청크 전체 "), ("proxy", "근거=유사문장 2개")):
        st = stats[mode]
        n = st["_n"]
        codes_ = ", ".join(f"{k} {v}" for k, v in st.most_common() if not k.startswith("_"))
        print(f"{label}  경고 {st['_flagged']:>3}/{n} ({st['_flagged'] / max(n, 1):.1%})  {codes_}")
    if verbose:
        print("\n[proxy 모드에서 걸린 문장]")
        for r in rows:
            if r["proxy_codes"]:
                print(f"  - [{r['draft_id']}] {r['sentence'][:70]}\n      {r['proxy_details'][:170]}")
    print(f"→ {out.relative_to(PROJECT_ROOT)}")
    p = stats["proxy"]
    return {"set": scan_set, "n": p["_n"], "flagged_chunk": stats["chunk"]["_flagged"], "flagged_proxy": p["_flagged"],
            "by_code_proxy": {k: v for k, v in p.items() if not k.startswith("_")}}


# ==================================================================
# 실행
# ==================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--scan-set", choices=["dev", "test"], default="dev")
    args = ap.parse_args()

    if not MATCHER.generic:
        print(f"경고: {CATALOG.relative_to(PROJECT_ROOT)} 을 읽지 못해 G 그룹이 무의미합니다.")

    results = []
    for g, n, fn in CASES:
        try:
            ok, d = fn()
        except Exception as e:
            ok, d = False, f"예외: {type(e).__name__}: {e}"
        results.append((g, n, ok, d))

    print("=" * 64)
    print("FinGuide-RAG v2 claim 규칙 테스트")
    print("=" * 64)
    groups: dict[str, list] = {}
    for g, n, ok, d in results:
        groups.setdefault(g, []).append(ok)
    for g, oks in groups.items():
        print(f"{g:<18} {sum(oks):>3}/{len(oks):<3} {'OK ' if all(oks) else 'FAIL'}")
    fails = [(g, n, d) for g, n, ok, d in results if not ok]
    if fails:
        print("\n[실패 상세]")
        for g, n, d in fails:
            print(f"  - {g} / {n}\n      실제: {d}")
    if args.verbose:
        print("\n[전체]")
        for g, n, ok, d in results:
            print(f"  {'✔' if ok else '✘'} {g} / {n}: {d}")
    print(f"\n총 {len(results)}건 중 실패 {len(fails)}건")

    scan = None if args.no_scan else clean_scan(args.verbose, args.scan_set)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"v2_rules_test_{date.today().isoformat()}.md"
    lines = [f"# v2 claim 규칙 테스트 ({date.today().isoformat()})", "",
             "| 그룹 | 케이스 | 결과 | 실제 |", "|---|---|---|---|"]
    lines += [f"| {g} | {n} | {'통과' if ok else '**실패**'} | `{d.replace('|', '/')[:200]}` |"
              for g, n, ok, d in results]
    if scan:
        lines += ["", f"## 정상 답변 사전 점검 ({scan['set']})", "",
                  f"- 문장 {scan['n']}개",
                  f"- 근거=청크 전체: 경고 {scan['flagged_chunk']}개",
                  f"- 근거=유사문장(LLM 정렬 대용): 경고 {scan['flagged_proxy']}개, {scan['by_code_proxy']}"]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {out.relative_to(PROJECT_ROOT)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
