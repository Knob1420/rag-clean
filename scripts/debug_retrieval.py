#!/usr/bin/env python3
"""
RAG 检索流程调试脚本

分步测试：
1. Query Rewrite — 查询理解与重写
2. BM25 检索 — 关键词匹配
3. Vector 检索 — 语义向量搜索
4. RRF 融合 — BM25 + Vector 合并
5. Rerank — 重排序（可选）
6. LLM 生成 — 生成回答
"""

import sys
import time
import pprint
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent.parent))


def step1_query_rewrite(query: str):
    """Step 1: 查询理解与重写"""
    print("\n" + "=" * 60)
    print("Step 1: Query Rewrite（查询理解与重写）")
    print("=" * 60)

    from core.query_engineer.query_rewrite import get_query_rewrite_service

    svc = get_query_rewrite_service()
    result = svc.rewrite(query)

    print(f"原始查询: {query}")
    print(f"重写后:   {result.rewritten_query}")
    print(f"策略:     {result.strategy}")
    print(f"意图:     {result.intent_type}")
    print(f"实体:     {result.target_entities}")
    print(f"关键词:    {result.keywords}")
    if result.sub_queries:
        print(f"子查询:   {pprint.pformat(result.sub_queries)}")

    return result


def step2_bm25(query: str, rewritten=None, top_k: int = 20):
    """Step 2: BM25 检索"""
    print("\n" + "=" * 60)
    print("Step 2: BM25 检索（关键词匹配）")
    print("=" * 60)

    from core.retrieve.retrieval import get_retrieval_service
    from core.retrieve.retrieval_models import RetrievalOptions, HighlightOptions

    svc = get_retrieval_service()

    # 解析 intent_type -> chunk_types 列表
    chunk_types = None
    if rewritten and rewritten.intent_type and rewritten.intent_type != "other":
        chunk_types = [t.strip() for t in rewritten.intent_type.split(",") if t.strip()]

    options = RetrievalOptions(
        top_k=top_k,
        use_rerank=False,
        target_models=(
            rewritten.target_entities
            if rewritten and rewritten.target_entities
            else None
        ),
        keywords=rewritten.keywords if rewritten else None,
        chunk_types=chunk_types,
    )

    # 使用 _bm25_search 而不是 search（避免混合检索干扰）
    highlight = HighlightOptions()
    chunks = svc._bm25_search(query, options, highlight)

    print(f"查询: {query}")
    print(f"chunk_types: {chunk_types}")
    print(f"keywords: {rewritten.keywords if rewritten else None}")
    print(f"结果数: {len(chunks)}")
    for i, c in enumerate(chunks[:5]):
        print(
            f"\n  [{i+1}] chunk_id={c.chunk_id}, doc_id={c.doc_id}, score={c.score:.4f}"
        )
        print(f"       section={c.section_title}, type={c.chunk_type}")
        print(f"       content={c.content[:100]}...")

    return chunks


def step3_vector(query: str, top_k: int = 20):
    """Step 3: Vector 检索"""
    print("\n" + "=" * 60)
    print("Step 3: Vector 检索（语义向量搜索）")
    print("=" * 60)

    from core.retrieve.retrieval import get_retrieval_service
    from core.retrieve.embedder import encode
    from core.retrieve.retrieval_models import RetrievalOptions

    svc = get_retrieval_service()

    # 向量化查询
    t0 = time.time()
    query_vec = encode(query)
    print(f"向量化耗时: {time.time() - t0:.3f}s")
    print(f"向量维度: {query_vec.shape if query_vec is not None else 'None'}")

    if query_vec is None:
        print("⚠️ 向量化为空，可能 embedding 服务未启动")
        return []

    options = RetrievalOptions(top_k=top_k, use_rerank=False)
    chunks = svc._execute_vector_search(query_vec, options, top_k)

    print(f"结果数: {len(chunks)}")
    for i, c in enumerate(chunks[:5]):
        print(
            f"\n  [{i+1}] chunk_id={c.chunk_id}, doc_id={c.doc_id}, score={c.score:.4f}"
        )
        print(f"       section={c.section_title}, type={c.chunk_type}")
        print(f"       content={c.content[:100]}...")

    return chunks


def step4_rrf(bm25_chunks, vector_chunks, top_k: int = 20):
    """Step 4: RRF 融合"""
    print("\n" + "=" * 60)
    print("Step 4: RRF 融合（BM25 + Vector 合并）")
    print("=" * 60)

    from core.retrieve.retrieval import get_retrieval_service

    svc = get_retrieval_service()

    bm25_with_rank = [(c, i) for i, c in enumerate(bm25_chunks)]
    vector_with_rank = [(c, i) for i, c in enumerate(vector_chunks)]

    fused = svc.rrf.fuse(
        bm25_results=bm25_with_rank,
        vector_results=vector_with_rank,
        bm25_weight=0.5,
        vector_weight=0.5,
    )

    # 截断到 top_k
    fused = fused[:top_k]

    print(f"融合结果数: {len(fused)}")
    for i, (c, score) in enumerate(fused):
        c.score = score  # 更新 score
        print(f"\n  [{i+1}] chunk_id={c.chunk_id}, score={score:.4f}")
        print(f"       doc_type={c.doc_type}, chunk_type={c.chunk_type}")
        print(f"       content={c.content[:80]}...")

    return [c for c, _ in fused]


def step5_rerank(query: str, chunks, top_k: int = 10):
    """Step 5: Rerank 重排序"""
    print("\n" + "=" * 60)
    print("Step 5: Rerank 重排序")
    print("=" * 60)

    if not chunks:
        print("无 chunks 可 rerank")
        return []

    from core.retrieve.retrieval import get_retrieval_service
    from core.retrieve.retrieval_models import RetrievalOptions

    svc = get_retrieval_service()
    options = RetrievalOptions(top_k=top_k, rerank_top_k=top_k, use_rerank=True)

    reranked = svc._rerank(query, chunks, options)

    print(f"Rerank 后结果数: {len(reranked)}")
    for i, c in enumerate(reranked[:5]):
        print(f"\n  [{i+1}] chunk_id={c.chunk_id}, score={c.score:.4f}")
        print(f"       doc_type={c.doc_type}, section={c.section_title}")
        print(f"       content={c.content[:80]}...")

    return reranked


def step6_generate(query: str, chunks):
    """Step 6: LLM 生成回答"""
    print("\n" + "=" * 60)
    print("Step 6: LLM 生成回答")
    print("=" * 60)

    if not chunks:
        print("⚠️ 无 chunks，跳过生成")
        return None

    from core.generation import get_generation_service

    svc = get_generation_service()

    t0 = time.time()
    answer, usage = svc.generate(query=query, chunks=chunks)
    elapsed = time.time() - t0

    print(f"生成耗时: {elapsed:.2f}s")
    print(
        f"Token: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}"
    )
    print(f"\n回答:\n{answer}")

    return answer


def step_full_pipeline(
    query: str, use_rewrite: bool = True, use_rerank: bool = True, top_k: int = 20
):
    """完整流程"""
    print("\n" + "=" * 60)
    print(f"完整 RAG 流程测试: query='{query}'")
    print("=" * 60)

    # 1. Query Rewrite
    if use_rewrite:
        rewritten = step1_query_rewrite(query)
        search_query = rewritten.rewritten_query
    else:
        rewritten = None
        search_query = query

    # 2. BM25
    bm25_chunks = step2_bm25(search_query, rewritten, top_k)

    # 3. Vector
    vector_chunks = step3_vector(search_query, top_k)

    # 4. RRF
    fused_chunks = step4_rrf(bm25_chunks, vector_chunks, top_k)

    if not fused_chunks:
        print("\n⚠️ 融合后无结果，终止")
        return

    # 5. Rerank
    if use_rerank:
        final_chunks = step5_rerank(search_query, fused_chunks, top_k)
    else:
        final_chunks = fused_chunks[:top_k]
        print(f"\n跳过 Rerank，返回 top_k={len(final_chunks)}")

    if not final_chunks:
        print("\n⚠️ Rerank 后无结果，终止")
        return

    # 6. Generate
    step6_generate(query, final_chunks)


def step_es_health():
    """检查 ES 健康"""
    print("\n" + "=" * 60)
    print("ES 健康检查")
    print("=" * 60)

    from store import get_store

    store = get_store()
    info = store.es.info()
    print(f"ES 版本: {info['version']['number']}")
    print(f"集群名: {info['cluster_name']}")

    # 检查索引
    chunks_index = "rag_chunks"
    if store.es.indices.exists(index=chunks_index):
        count = store.es.count(index=chunks_index)
        print(f"✅ 索引 '{chunks_index}' 存在，文档数: {count['count']}")
    else:
        print(f"⚠️ 索引 '{chunks_index}' 不存在")


def step_embedding_health():
    """检查 Embedding 服务"""
    print("\n" + "=" * 60)
    print("Embedding 服务检查")
    print("=" * 60)

    from core.retrieve.embedder import encode
    from config import settings

    print(f"服务地址: http://localhost:{settings.embedding_port}")
    print(f"模型: {settings.embedding_model}")
    print(f"维度: {settings.embedding_dim}")

    t0 = time.time()
    vec = encode("测试文本")
    elapsed = time.time() - t0

    if vec is not None:
        print(f"✅ 向量化成功，耗时: {elapsed:.3f}s")
        print(f"   向量维度: {vec.shape}, 前3维: {vec[:3]}")
    else:
        print(f"⚠️ 向量化失败（服务可能未启动）")


def step_rerank_health():
    """检查 Rerank 服务"""
    print("\n" + "=" * 60)
    print("Rerank 服务检查")
    print("=" * 60)

    from config import settings
    import httpx

    print(f"服务地址: http://localhost:{settings.rerank_port}")

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"http://localhost:{settings.rerank_port}/health")
            if resp.status_code == 200:
                print(f"✅ Rerank 服务正常: {resp.json()}")
            else:
                print(f"⚠️ Rerank 服务异常: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Rerank 服务未启动: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 检索流程调试")
    parser.add_argument("query", nargs="?", default="G1参数", help="查询内容")
    parser.add_argument("--no-rewrite", action="store_true", help="跳过 Query Rewrite")
    parser.add_argument("--no-rerank", action="store_true", help="跳过 Rerank")
    parser.add_argument("--top-k", type=int, default=20, help="返回结果数")
    parser.add_argument("--health", action="store_true", help="仅检查服务健康")
    parser.add_argument(
        "--steps",
        type=str,
        default="1,2,3,4,5,6",
        help="执行的步骤，逗号分隔 (1=rewrite, 2=bm25, 3=vector, 4=rrf, 5=rerank, 6=generate)",
    )

    args = parser.parse_args()

    # 健康检查
    if args.health:
        step_es_health()
        step_embedding_health()
        step_rerank_health()
        sys.exit(0)

    steps = [int(s.strip()) for s in args.steps.split(",")]
    query = args.query

    print(f"\n{'='*60}")
    print(f"RAG 检索调试 — 查询: '{query}'")
    print(
        f"步骤: {args.steps}, rewrite={not args.no_rewrite}, rerank={not args.no_rerank}, top_k={args.top_k}"
    )
    print(f"{'='*60}")

    # 执行指定步骤
    rewritten = None
    bm25_chunks = []
    vector_chunks = []
    fused_chunks = []
    final_chunks = []

    if 1 in steps:
        step_es_health()
        step_embedding_health()
        step_rerank_health()

    if 1 in steps and not args.no_rewrite:
        rewritten = step1_query_rewrite(query)
        search_query = rewritten.rewritten_query
    else:
        search_query = query

    if 2 in steps:
        bm25_chunks = step2_bm25(search_query, rewritten, args.top_k)

    if 3 in steps:
        vector_chunks = step3_vector(search_query, args.top_k)

    if 4 in steps and bm25_chunks:
        if vector_chunks:
            fused_chunks = step4_rrf(bm25_chunks, vector_chunks, args.top_k)
        else:
            # 向量为空时，纯 BM25 结果
            fused_chunks = bm25_chunks
            print(f"\n⚠️ 向量结果为空，纯 BM25 模式，top_k={len(fused_chunks)}")
    elif 4 in steps:
        print("\n⚠️ 跳过 RRF: 需要先执行步骤 2")

    if 5 in steps and fused_chunks and not args.no_rerank:
        final_chunks = step5_rerank(search_query, fused_chunks, args.top_k)
    elif 5 in steps and fused_chunks:
        final_chunks = fused_chunks[: args.top_k]
        print(f"\n跳过 Rerank，返回 top_k={len(final_chunks)}")
    elif 5 in steps:
        print("\n⚠️ 跳过 Rerank: 需要先执行步骤 4")

    if 6 in steps and final_chunks:
        step6_generate(query, final_chunks)
    elif 6 in steps:
        print("\n⚠️ 跳过生成: 需要先执行步骤 5")

    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)
