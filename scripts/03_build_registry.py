"""
문서 대장(document registry) 자동 생성 스크립트
================================================
data/raw/hana/{desc,terms} 의 PDF 파일명을 파싱하고, FAQ jsonl 을 합쳐
data/registry/document_registry.csv 를 만든다.

파일명 규칙: hana_{doc_type}_{category}_{product...}_{YYYYMMDD}.pdf
 - 뒤에서부터 파싱: 끝 8자리=시행일, 앞 3토큰=은행/유형/카테고리(두 단어 카테고리 보정),
   가운데 나머지 전부=상품명 (상품명에 '_'가 많아도 안전)

실행:
    python 03_build_registry.py
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # 스크립트를 프로젝트 루트에 두는 경우
# scripts/ 안에 둔다면 위 줄을 parents[1] 로 바꾸세요.
RAW = ROOT / "data" / "raw" / "hana"
OUT = ROOT / "data" / "registry" / "document_registry.csv"

# 두 단어로 된 카테고리 (파일명에서 이 접두사가 나오면 두 토큰을 합쳐 카테고리로 인식)
TWO_WORD_CATS = {
    ("digital", "banking"): "digital_banking",
    ("retirement", "pension"): "retirement_pension",
    ("trust", "isa"): "trust_isa",
}

DOC_TYPE_KR = {"desc": "설명서", "terms": "약관", "faq": "FAQ"}
DATE_RE = re.compile(r"(\d{8})")  # YYYYMMDD


def parse_date(token: str) -> str:
    """8자리 숫자를 YYYY-MM-DD 로. 실패 시 빈 문자열."""
    m = DATE_RE.fullmatch(token.strip())
    if not m:
        return ""
    s = m.group(1)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def parse_pdf_name(fname: str):
    """파일명 -> (doc_type, category, product, version_date)"""
    stem = fname[:-4] if fname.lower().endswith(".pdf") else fname
    stem = stem.strip()
    tokens = stem.split("_")

    # 1) 끝 토큰 = 시행일 (공백/문자 섞였을 수 있어 정규식으로 추출)
    version_date = ""
    if tokens:
        m = DATE_RE.search(tokens[-1])
        if m:
            version_date = parse_date(m.group(1))
            tokens = tokens[:-1]  # 날짜 토큰 제거

    # 2) 앞 토큰: hana / doc_type / category
    #    tokens = [hana, doc_type, cat1, (cat2?), product...]
    doc_type = tokens[1] if len(tokens) > 1 else ""
    rest = tokens[2:]  # 카테고리 + 상품명

    category = rest[0] if rest else ""
    prod_tokens = rest[1:]
    # 두 단어 카테고리 보정
    if len(rest) >= 2 and (rest[0], rest[1]) in TWO_WORD_CATS:
        category = TWO_WORD_CATS[(rest[0], rest[1])]
        prod_tokens = rest[2:]

    product = " ".join(prod_tokens).strip()
    return doc_type, category, product, version_date


def collect_pdfs(doc_type_dir: str, rows: list, seq_counter: dict):
    folder = RAW / doc_type_dir
    if not folder.exists():
        return
    for pdf in sorted(folder.glob("*.pdf")):
        doc_type, category, product, vdate = parse_pdf_name(pdf.name)
        seq_counter[doc_type_dir] = seq_counter.get(doc_type_dir, 0) + 1
        n = seq_counter[doc_type_dir]
        doc_id = f"hana_{doc_type_dir}_{category}_{n:03d}"
        rows.append({
            "doc_id": doc_id,
            "bank_code": "hana",
            "doc_type": DOC_TYPE_KR.get(doc_type_dir, doc_type_dir),
            "category": category,
            "product": product,
            "version_date": vdate,
            "n_faq": "",
            "source_path": str(pdf.relative_to(ROOT)).replace("\\", "/"),
            "orig_filename": pdf.name,
        })


def collect_faq(rows: list):
    faq_path = RAW / "faq" / "faq_hana.jsonl"
    if not faq_path.exists():
        return
    # FAQ는 카테고리별로 한 줄(그룹)로 대장에 요약 등록
    cat_counter = {}
    with open(faq_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = o.get("category", "unknown")
            cat_counter[c] = cat_counter.get(c, 0) + 1
    for i, (cat, cnt) in enumerate(sorted(cat_counter.items()), 1):
        rows.append({
            "doc_id": f"hana_faq_{cat}_{i:03d}",
            "bank_code": "hana",
            "doc_type": "FAQ",
            "category": cat,
            "product": f"FAQ 모음 ({cat})",
            "version_date": "",
            "n_faq": cnt,
            "source_path": "data/raw/hana/faq/faq_hana.jsonl",
            "orig_filename": "faq_hana.jsonl",
        })


def main():
    rows, seq_counter = [], {}
    collect_pdfs("desc", rows, seq_counter)
    collect_pdfs("terms", rows, seq_counter)
    collect_faq(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["doc_id", "bank_code", "doc_type", "category", "product",
              "version_date", "n_faq", "source_path", "orig_filename"]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 콘솔 요약
    from collections import Counter
    by_type = Counter(r["doc_type"] for r in rows)
    print("=" * 55)
    print(f"문서 대장 생성 완료 → {OUT.relative_to(ROOT)}")
    print(f"총 {len(rows)}행")
    for t, c in by_type.items():
        print(f"  - {t}: {c}행")
    # 파싱 경고: 시행일/상품명 빈 항목
    missing_date = [r["orig_filename"] for r in rows if r["doc_type"] != "FAQ" and not r["version_date"]]
    missing_prod = [r["orig_filename"] for r in rows if r["doc_type"] != "FAQ" and not r["product"]]
    if missing_date:
        print(f"\n[경고] 시행일 파싱 실패 {len(missing_date)}건:")
        for m in missing_date[:10]:
            print(f"    {m}")
    if missing_prod:
        print(f"\n[경고] 상품명 비어있음 {len(missing_prod)}건:")
        for m in missing_prod[:10]:
            print(f"    {m}")
    if not missing_date and not missing_prod:
        print("\n모든 PDF의 시행일·상품명 파싱 성공.")
    print("=" * 55)


if __name__ == "__main__":
    main()
