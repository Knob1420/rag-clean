import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.query_engineer.keyword_extractor import ChineseKeywordExtractor

KE = ChineseKeywordExtractor()

# with open("queries.txt", "r", encoding="utf-8") as f:
#     queries = f.readlines()
queries = ["之江实验室发射了多少颗卫星"]
for query in queries:
    key = KE.extract(query)
    print(key)
