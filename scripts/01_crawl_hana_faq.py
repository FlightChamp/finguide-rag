"""
하나은행 FAQ 카테고리별 수집 크롤러
=====================================
- 실제 페이지에서 확인된 카테고리 URL/페이지네이션 구조를 그대로 반영.
- 카테고리별 '목표 수집 개수'만큼만 수집 (대표성 확보용 샘플링).
- 출력: data/raw/hana/faq/faq_hana.jsonl  (사용자 스키마와 동일 필드 + source_url)

주의:
1) Claude 샌드박스는 hanabank.com에 접근 불가 → 로컬 PC에서 실행.
2) `pip install requests beautifulsoup4 --break-system-packages`
3) REQUEST_DELAY(1.5초)는 서버 부담 방지를 위해 절대 줄이지 말 것.
4) CSS 셀렉터는 사이트 구조 변경에 대비해 '텍스트 마커(질문/답변) 기반 폴백'을 함께 둠.
   최초 1회는 반드시 브라우저 F12로 실제 구조를 확인해 SELECTORS를 검증할 것.
"""

import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.hanabank.com/cont/customer/customer01"
HEADERS = {"User-Agent": "Mozilla/5.0 (research/portfolio; contact: your_email@example.com)"}
REQUEST_DELAY = 1.5

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "hana" / "faq"
OUT_PATH = OUT_DIR / "faq_hana.jsonl"

# (category_code, 표시명, URL경로, 목표 수집 개수)  ── 실측 URL 반영
# 사용자가 배정한 개수를 그대로 사용. 실제 보유 개수가 목표보다 적으면 있는 만큼만 수집.
CATEGORIES = [
    ("deposit_trust",            "예금/신탁",            "customer0102/customer010201", 32),
    ("loan",                     "대출",                 "customer0102/customer010203", 32),
    ("certificate",              "인증서",               "customer0101/customer010102", 22),
    ("internet_banking",         "인터넷뱅킹",           "customer0103/customer010301", 20),
    ("fx",                       "외환",                 "customer0102/customer010204", 16),
    ("auth_otp_security_card",   "인증/OTP/보안카드",    "customer0101/customer010103", 16),
    ("login_signup",             "로그인/회원가입",      "customer0101/customer010101", 16),
    ("hana_oneq_mobile_banking", "하나원큐(스마트폰뱅킹)", "customer0103/customer010302", 16),
    ("fund",                     "펀드",                 "customer0102/customer010202", 12),
    ("cd_atm",                   "CD/ATM",               "customer0103/customer010304", 8),
    ("bill_payment",             "공과금납부",           "customer0104/customer010401", 4),
    ("overseas_branch_others",   "해외/영업점/기타",     "customer0104/customer010403", 4),
    ("phone_banking",            "폰뱅킹",               "customer0103/customer010303", 2),
    # 필요 시 아래 '하나 알리미' 주석 해제 (사이트에는 존재하는 14번째 카테고리)
    # ("hana_alrimi",            "하나 알리미",          "customer0104/customer010402", 0),
]

# F12로 확인 후 검증할 셀렉터 후보. 실제 클래스명이 다르면 여기만 고치면 됨.
QA_ITEM_SELECTORS = ["ul.faqList > li", "div.faq_list li", "dl.faq", ".board_list li"]


def fetch_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return BeautifulSoup(resp.text, "html.parser")


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    # 맨 앞의 'Q.', 'A.', '질문', '답변' 라벨 제거
    text = re.sub(r"^(Q\.|A\.|질문|답변)\s*", "", text)
    text = re.sub(r"^(Q\.|A\.|질문|답변)\s*", "", text)  # 'Q. 질문' 이중 라벨 대비
    return text.strip()


def parse_qa(soup: BeautifulSoup):
    """Q/A 쌍 리스트 반환. 셀렉터 우선 시도 후, 실패 시 텍스트 마커 기반 폴백."""
    # 1) 셀렉터 기반
    for sel in QA_ITEM_SELECTORS:
        items = soup.select(sel)
        if items:
            out = []
            for it in items:
                q_el = it.find(["dt", "a", "strong"])
                a_el = it.find(["dd", "div", "p"])
                q, a = clean(q_el.get_text() if q_el else ""), clean(a_el.get_text() if a_el else "")
                if q and a:
                    out.append((q, a))
            if out:
                return out
    # 2) 폴백: 'Q.' ~ 'A.' 텍스트 패턴 분해
    raw = soup.get_text("\n")
    blocks = re.split(r"Q\.\s*질문", raw)
    out = []
    for b in blocks[1:]:
        m = re.split(r"A\.\s*답변", b, maxsplit=1)
        if len(m) == 2:
            q, a = clean(m[0]), clean(m[1].split("Q. 질문")[0])
            if q and a:
                out.append((q, a))
    return out


def crawl_category(code, name, path, target, seen, seq_start):
    url_base = f"{BASE}/{path}"
    collected, page, seq = [], 1, seq_start
    while len(collected) < target and page <= 60:
        url = f"{url_base}/index.jsp" if page == 1 else f"{url_base}/index,1,list,{page}.jsp"
        try:
            soup = fetch_soup(url)
        except requests.HTTPError:
            break
        pairs = parse_qa(soup)
        if not pairs:
            break
        for q, a in pairs:
            key = (q, a)
            if key in seen:          # 전역 중복 제거
                continue
            seen.add(key)
            seq += 1
            collected.append({
                "faq_id": f"hana_faq_{seq:03d}",
                "bank_code": "hana",
                "bank_name": "하나은행",
                "category": code,
                "question": q,
                "answer": a,
                "source_url": url,
                "collected_at": time.strftime("%Y-%m-%d"),
                "status": "active",
            })
            if len(collected) >= target:
                break
        page += 1
    print(f"[{name}] 목표 {target} / 수집 {len(collected)}")
    return collected, seq


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen, seq, all_rows = set(), 0, []
    for code, name, path, target in CATEGORIES:
        rows, seq = crawl_category(code, name, path, target, seen, seq)
        all_rows.extend(rows)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n총 {len(all_rows)}건 저장 → {OUT_PATH.relative_to(ROOT)}")
    print("빈 질문/답변은 저장되지 않으며, 전역 중복도 제거되었습니다.")


if __name__ == "__main__":
    main()
