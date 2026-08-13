"""검색 성능 지표.

정보 검색(IR) 표준 지표를 구현한다. 라이브러리를 쓰지 않고 직접 구현한
이유는, 우리 데이터의 특성(질의당 정답이 여러 개)에 맞춰 정의를 명확히
하기 위해서다. 지표를 블랙박스로 두면 숫자가 이상할 때 원인을 못 찾는다.

지표 정의
--------
Recall@k
    상위 k개 안에 정답이 하나라도 있으면 1, 없으면 0. 전체 평균.
    정답이 여러 개인 경우 "모두 찾았는가"가 아니라 "하나라도 찾았는가"로
    센다. 직원은 근거 문서 하나만 확인하면 답할 수 있으므로, 이 정의가
    실사용에 부합한다.

MRR (Mean Reciprocal Rank)
    첫 번째 정답이 나온 순위의 역수. 1위면 1.0, 2위면 0.5, 3위면 0.33.
    "찾긴 하는데 순위가 낮다"를 잡아낸다. Recall@5가 같아도 MRR이 높으면
    사용자가 더 빨리 답을 본다.

NDCG@k
    순위와 정답 개수를 함께 반영한다. 정답이 여러 개일 때 상위에 많이
    모여 있을수록 높다. 이진 관련도(정답/비정답)를 가정한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ------------------------------------------------------------------
# 개별 지표
# ------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """상위 k개 안에 정답이 하나라도 있으면 1.0, 없으면 0.0.

    엄밀히는 'success@k' 또는 'hit rate@k' 로 부르는 지표다. 정답을 모두
    찾았는지 보는 전통적 recall과는 다르다. 다만 RAG 문헌에서 관행적으로
    Recall@k 라고 부르므로 이름을 따랐다.
    """
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def full_recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """상위 k개가 전체 정답 중 몇 %를 포함하는지.

    전통적 정의의 recall이다. 참고용으로 함께 계산한다.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """상위 k개 중 정답의 비율."""
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """첫 정답이 나온 순위의 역수. 못 찾으면 0."""
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def first_hit_rank(retrieved: list[str], relevant: set[str]) -> int:
    """첫 정답의 순위. 못 찾으면 0.

    지표는 아니지만 실패 분석에 쓴다. 6~20위인지 그 밖인지에 따라
    처방이 달라진다.
    """
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return i
    return 0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """정규화 할인 누적 이득. 이진 관련도 기준.

    DCG = sum(rel_i / log2(i + 1))
    IDCG = 정답이 모두 상위에 몰려 있는 이상적 경우의 DCG
    """
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc_id in enumerate(retrieved[:k], 1)
        if doc_id in relevant
    )

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ------------------------------------------------------------------
# 집계
# ------------------------------------------------------------------


@dataclass
class QueryResult:
    """질의 1건의 평가 결과."""

    query_id: str
    question: str
    difficulty: str
    doc_type: str
    category: str
    relevant: set[str]
    retrieved: list[str]
    scores: list[float] = field(default_factory=list)

    @property
    def hit_rank(self) -> int:
        return first_hit_rank(self.retrieved, self.relevant)

    @property
    def top1_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    @property
    def score_gap(self) -> float:
        """1위와 2위의 점수 차.

        절대 점수가 좁은 구간에 몰려 변별력이 없을 때, 상대적 신호로
        쓸 수 있다. 거절 규칙 설계에 활용할 값이다.
        """
        if len(self.scores) < 2:
            return 0.0
        return self.scores[0] - self.scores[1]

    def metrics(self, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in ks:
            out[f"recall@{k}"] = recall_at_k(self.retrieved, self.relevant, k)
        out["mrr"] = reciprocal_rank(self.retrieved, self.relevant)
        out["ndcg@10"] = ndcg_at_k(self.retrieved, self.relevant, 10)
        out["full_recall@10"] = full_recall_at_k(self.retrieved, self.relevant, 10)
        out["precision@5"] = precision_at_k(self.retrieved, self.relevant, 5)
        return out


def aggregate(results: list[QueryResult], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """질의별 지표를 평균낸다."""
    if not results:
        return {}

    keys = list(results[0].metrics(ks).keys())
    totals = {k: 0.0 for k in keys}
    for r in results:
        for k, v in r.metrics(ks).items():
            totals[k] += v

    out = {k: round(v / len(results), 4) for k, v in totals.items()}
    out["n"] = len(results)
    return out


def group_by(results: list[QueryResult], attr: str,
             ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, dict[str, float]]:
    """속성별로 나눠 집계한다.

    난이도별·문서유형별 성능을 따로 보는 것이 중요하다. 전체 평균 하나만
    보면 중요한 차이가 묻힌다. 예컨대 FAQ는 질문으로 질문을 찾는 구조라
    유리하고, 약관은 문어체라 구어체 질의와 거리가 멀다. 이 격차를 알아야
    개선 노력을 어디에 쏟을지 정할 수 있다.
    """
    groups: dict[str, list[QueryResult]] = {}
    for r in results:
        groups.setdefault(getattr(r, attr), []).append(r)
    return {key: aggregate(rs, ks) for key, rs in sorted(groups.items())}


def failure_analysis(results: list[QueryResult]) -> dict[str, int]:
    """검색 실패를 원인별로 분류한다.

    처방이 다르므로 구별해야 한다.
    - 1~5위      : 성공
    - 6~20위     : 찾긴 하지만 순위가 낮다. 재순위화로 개선 가능
    - 20위 밖    : 후보에도 못 들었다. 질의-문서 표현 격차 문제
    """
    buckets = {"top5": 0, "rank_6_20": 0, "beyond_20": 0}
    for r in results:
        rank = r.hit_rank
        if rank == 0 or rank > 20:
            buckets["beyond_20"] += 1
        elif rank <= 5:
            buckets["top5"] += 1
        else:
            buckets["rank_6_20"] += 1
    return buckets
