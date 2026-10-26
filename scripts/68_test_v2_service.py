"""68 — v2 화면 서비스 층(V2Service) 테스트. API 키 없이, 가짜 OpenAI 클라이언트로 몇 초.

확인하는 것
----------
    A  정상 흐름: 검증 → 쉬운 설명 → 역검증 PASS, API 호출 수 집계
    B  호출 상한: 상한 0 / 일부만 허용 → fail-closed (확인 필요 · 안내 보류), 틀린 안내가 나가지 않음
    C  캐시: 같은 입력을 다시 돌리면 API 0콜로 같은 결과
    D  근거 없음 · 근거 변환

실행
----
    python scripts/68_test_v2_service.py
종료 코드: 실패가 있으면 1
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finguide_rag.verification.back_verify import COND_SYSTEM  # noqa: E402
from finguide_rag.verification.claim_verifier import ALIGN_SYSTEM, EXTRACT_SYSTEM  # noqa: E402
from finguide_rag.verification.schema import BVStatus, FinalLabel, Verdict  # noqa: E402
from finguide_rag.verification.service import V2Service, chunks_from_evidences  # noqa: E402
from finguide_rag.verification.simplify import FIX_SYSTEM, SIMPLIFY_SYSTEM  # noqa: E402

Q = "정기예금을 만기 전에 해지하면 어떻게 되나요?"
DRAFT = ("만기 전에 해지하면 중도해지이율이 적용됩니다. "
         "우대금리는 급여이체 실적이 있는 경우에 한해 연 0.2%p 적용됩니다.")
EVIDENCES = [{"chunk_id": "doc_a_c001", "citation": "예시 정기예금 상품설명서", "doc_type": "설명서",
              "text": "만기 전에 해지하면 중도해지이율이 적용됩니다. 우대금리는 급여이체 실적이 있는 경우에 한해 연 0.2%p 적용됩니다."}]

EXTRACT = {"claims": [
    {"text": "만기 전에 해지하면 중도해지이율이 적용됩니다.", "product": None, "event": "중도해지",
     "conditions": ["만기 전에 해지하면"], "needs_customer_data": False},
    {"text": "우대금리는 급여이체 실적이 있는 경우에 한해 연 0.2%p 적용됩니다.", "product": None, "event": None,
     "conditions": ["급여이체 실적이 있는 경우에 한해"], "needs_customer_data": False}]}
ALIGN = {"results": [
    {"claim_id": "c1", "evidence_ids": ["E1"], "code": "SUPPORTED", "missing_conditions": [], "fix": None},
    {"claim_id": "c2", "evidence_ids": ["E2"], "code": "SUPPORTED", "missing_conditions": [], "fix": None}]}
SIMPLE = {"core": [{"text": "만기 전에 해지하면 중도해지이율이 적용됩니다.", "claim_ids": ["c1"]},
                   {"text": "우대금리로 연 0.2%p가 더해집니다.", "claim_ids": ["c2"]}],
          "conditions": [{"text": "만기 전에 해지하면 중도해지이율이 적용됩니다.", "claim_ids": ["c1"],
                          "condition_ids": ["k1"]},
                         {"text": "우대금리는 급여이체 실적이 있는 경우에 한해 받습니다.", "claim_ids": ["c2"],
                          "condition_ids": ["k2"]}]}


class FakeClient:
    """openai.OpenAI 와 같은 모양. system 프롬프트로 역할을 구분한다."""

    def __init__(self):
        self.n = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, temperature, max_tokens, response_format, messages):
        self.n += 1
        system, user = messages[0]["content"], messages[1]["content"]
        if system == EXTRACT_SYSTEM:
            data = EXTRACT
        elif system == ALIGN_SYSTEM:
            data = ALIGN
        elif system in (SIMPLIFY_SYSTEM, FIX_SYSTEM):
            data = SIMPLE
        elif system == COND_SYSTEM:
            conds = re.findall(r"^(k\d+) \(c\d+\): (.+)$", user, re.M)
            data = {"results": [{"cond_id": k, "status": "PRESERVED", "where": ["n1" if k == "k1" else "n2"]}
                                for k, _ in conds], "extra": []}
        else:
            raise AssertionError("알 수 없는 프롬프트")
        msg = SimpleNamespace(content=json.dumps(data, ensure_ascii=False))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10))


RESULTS: list[tuple[str, bool, str]] = []


def case(name):
    def deco(fn):
        try:
            RESULTS.append((name, True, str(fn() or "")))
        except AssertionError as e:
            RESULTS.append((name, False, str(e)))
        except Exception as e:   # noqa: BLE001
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


def service(tmp: Path, client) -> V2Service:
    return V2Service.build(PROJECT_ROOT, client=client, cache_path=tmp / "cache.json")


TMP = Path(tempfile.mkdtemp())


@case("A 정상 흐름: OK 2건 → 쉬운 설명 PASS, API 4콜 (추출·정렬·쉬운 설명·조건 검사)")
def _():
    c = FakeClient()
    r = service(TMP / "a", c).run(Q, DRAFT, EVIDENCES, max_calls=20)
    assert [v.verdict for v in r.verify.verifications] == [Verdict.OK, Verdict.OK], \
        [(v.verdict, [h.code.value for h in v.reasons]) for v in r.verify.verifications]
    assert r.verify.final.final == FinalLabel.OK
    assert r.simplify.status == BVStatus.PASS, [p.code.value for _, b in r.simplify.history for p in b.problems]
    assert r.api_calls == 4 and c.n == 4 and r.blocked_calls == 0, (r.api_calls, c.n)
    assert r.simplify.customer_visible and "0.2%p" in r.simplify.to_dict()["customer_text"]


@case("B 호출 상한 0: 새 호출 없음 → claim 추출 실패 → 확인 필요, 쉬운 설명 대상 없음")
def _():
    c = FakeClient()
    r = service(TMP / "b0", c).run(Q, DRAFT, EVIDENCES, max_calls=0)
    assert c.n == 0 and r.api_calls == 0 and r.blocked_calls >= 1
    assert r.verify.final.final == FinalLabel.CHECK
    assert r.simplify.status is None and not r.simplify.customer_visible
    assert any("호출 상한" in n for n in r.notes)


@case("B 호출 상한 2: 검증은 되지만 쉬운 설명 호출이 막힘 → BLOCK, 고객 문구 없음")
def _():
    c = FakeClient()
    r = service(TMP / "b2", c).run(Q, DRAFT, EVIDENCES, max_calls=2)
    assert r.verify.final.final == FinalLabel.OK and c.n == 2
    assert r.simplify.status == BVStatus.BLOCK and r.simplify.to_dict()["customer_text"] is None
    assert r.simplify.fallback_lines, "BLOCK 이면 OK claim 원문 대체 안내가 있어야 함"


@case("C 캐시: 같은 입력 재실행은 상한 0 이어도 API 0콜로 같은 결과")
def _():
    s = service(TMP / "c", FakeClient())
    r1 = s.run(Q, DRAFT, EVIDENCES, max_calls=20)
    c2 = FakeClient()
    s2 = service(TMP / "c", c2)            # 같은 캐시 파일을 새로 읽음
    r2 = s2.run(Q, DRAFT, EVIDENCES, max_calls=0)
    assert c2.n == 0 and r2.api_calls == 0 and r2.blocked_calls == 0
    assert r2.simplify.status == r1.simplify.status == BVStatus.PASS
    assert r2.simplify.to_dict()["customer_text"] == r1.simplify.to_dict()["customer_text"]


@case("C 오프라인(클라이언트 없음) + 캐시 없음 → 확인 필요, 예외 없음")
def _():
    r = service(TMP / "off", None).run(Q, DRAFT, EVIDENCES, max_calls=20)
    assert r.verify.final.final == FinalLabel.CHECK and not r.simplify.customer_visible


@case("D 근거 없음 → 안내 문구 + 근거 없음 판정")
def _():
    r = service(TMP / "d", FakeClient()).run(Q, DRAFT, [], max_calls=20)
    assert any("근거 문서가 없습니다" in n for n in r.notes)
    assert all(v.verdict != Verdict.OK for v in r.verify.verifications)


@case("D 근거 변환: chunk_id 없는 근거는 버리고 citation 을 문서명으로")
def _():
    ch = chunks_from_evidences([{"chunk_id": "x_c1", "text": "t", "citation": "문서", "doc_type": "FAQ"},
                                {"text": "no id"}])
    assert len(ch) == 1 and ch[0].doc_name == "문서" and ch[0].doc_type == "FAQ"


def main() -> int:
    fails = [r for r in RESULTS if not r[1]]
    print("=" * 66)
    print(f"v2 서비스 층 테스트  {len(RESULTS) - len(fails)}/{len(RESULTS)} 통과")
    print("=" * 66)
    for name, ok, detail in RESULTS:
        print(f"  {'✔' if ok else '✘'} {name}" + (f"\n      {detail[:300]}" if not ok else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
