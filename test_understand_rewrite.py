#!/usr/bin/env python3
"""
测试 Query Understanding + Query Rewrite（不执行检索）

用法：
  python test_understand_rewrite.py
  python test_understand_rewrite.py "G1和G2的对比"
  python test_understand_rewrite.py "G1/G2/G3适合什么场景" "三体计算星座的定义"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.query_engineer.query_understanding import QueryUnderstandingService
from core.query_engineer.query_rewrite import QueryRewriteServiceV2


def test(query: str):
    print(f"\n{'='*60}")
    print(f"原始查询: {query}")
    print(f"{'='*60}")

    # 1. Query Understanding
    understanding_svc = QueryUnderstandingService()
    understanding = understanding_svc.parse(query)
    print(f"\n[Query Understanding]")
    print(f"  sub_questions: {len(understanding.sub_queries)}")
    for i, sq in enumerate(understanding.sub_queries, 1):
        print(f"  [{i}] intent={sq.intent}")
        print(f"      query: {sq.query}")

    # 2. Query Rewrite
    rewrite_svc = QueryRewriteServiceV2()
    print(f"\n[Query Rewrite]")
    for sq in understanding.sub_queries:
        rr = rewrite_svc.rewrite(sq.query, sq.intent)
        print(f"\n  sub_question: {sq.query}")
        print(f"  intent: {sq.intent}")
        print(f"  rewritten_queries: {rr.rewritten_queries}")
        print(f"  entities: {rr.entities}")
        print(f"  required_fields: {rr.required_fields}")
        print(f"  numerical_constraints: {rr.numerical_constraints}")


if __name__ == "__main__":
    queries = (
        sys.argv[1:]
        if len(sys.argv) > 1
        else [
            "三体计算星座的定义",
            "介绍一下三体计算星座，帮我翻译成英文",
            "三体计算星座建设规划",
            "地卫二项目代号",
            "具身智能卫星模型介绍，一段话介绍一下它的能力，不超过50字",
            "宇宙X射线偏振探测器原理",
            "简单介绍橄榄叶计划",
            "3kg以内的星载智算机，可以帮我推荐一个吗？",
            "我们首发之前的智算机的在轨验证，有哪几次啊？",
            "3618号新型胞元的来历写一下",
            "介绍一下3D打印卫星",
            "nx1 gpu板的数据盘落盘速度是多少",
            "上海的合作单位有哪些？",
            "长三角地区合作单位及合作形式、预期成果",
            "我们和蓝箭鸿擎的合作有什么，写一段话即可",
            "我们与国星宇航的合作有什么",
            "之江实验室发射了多少颗卫星",
            "智加G3支持什么接口",
            "天基分布式操作系统的特点",
            "NX系列和G系列有什么区别 ",
            "G1、G2、G3分别适合什么场景？",
            "智加全系列产品的尺寸和重量对比",
            "推荐一款轻量级星载智算机",
            "推荐一款2kg以内，算力大于250TFlops的智算机",
        ]
    )

    for q in queries:
        test(q)
