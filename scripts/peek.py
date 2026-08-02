import json
from pathlib import Path

PATH = Path(r"C:\Users\sky-56\data\raw\hana\faq\faq_hana.jsonl")
IDS = {"hana_faq_006", "hana_faq_018", "hana_faq_110",
       "hana_faq_026", "hana_faq_028", "hana_faq_045", "hana_faq_058",
       "hana_faq_123", "hana_faq_142",
       "hana_faq_151", "hana_faq_152", "hana_faq_153"}

with open(PATH, encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        if o["faq_id"] in IDS:
            print(f'{o["faq_id"]} [{o["category"]}]')
            print(f'  Q: {o["question"]}')
            print(f'  A: {o["answer"][:80]}')
            print()