"""
RAG API 服务 — FastAPI 接口

提供检索和 RAG 问答端点：
- POST /api/v1/chat/completions — 完整 RAG 流程（rewrite → retrieve → rerank → generate）
- POST /api/v1/search — 纯检索
- GET /health — 健康检查

基于 rag-knowledge-base api/main.py，移除：
- user_roles 参数
- chat_logger 依赖（用 loguru 代替）
- 文档上传/管理端点（rag-clean 有自己的 pipeline）
"""

import json
import time
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from config import settings
from core.generation import get_generation_service
from core.query_engineer.query_rewrite import get_query_rewrite_service, RewrittenQuery
from core.query_engineer.react_reasoning import get_react_reasoning_service
from core.retrieve.retrieval import get_retrieval_service
from core.retrieve.retrieval_models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HighlightOptions,
    RetrievalOptions,
    SearchRequest,
    SearchResponse,
    SourceInfo,
    TokenUsage,
)
from store import get_store


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="RAG Clean API",
    description="RAG 检索与问答 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 辅助函数
# ============================================================


def chunks_to_sources(chunks: list) -> list:
    """将检索分块转换为来源信息（含文档标题）"""
    # 批量获取 doc_id → title 映射
    doc_ids = {c.doc_id for c in chunks}
    store = get_store()
    doc_titles = store.get_doc_titles(list(doc_ids))

    sources = []
    for chunk in chunks:
        snippet = (
            chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
        )
        sources.append(
            SourceInfo(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_name=doc_titles.get(chunk.doc_id, ""),
                section_title=chunk.section_title,
                score=chunk.score,
                snippet=snippet,
            )
        )
    return sources


# ============================================================
# 健康检查
# ============================================================


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "RAG Clean API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/v1/chat/completions": "RAG 问答",
            "POST /api/v1/search": "检索",
            "POST /api/v1/parse": "PDF 解析（代理到 MinerU）",
            "GET /api/v1/parse/status/{parse_id}": "查询解析结果",
            "GET /api/v1/documents": "文档列表",
            "GET /health": "健康检查",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    services_status = {}

    # 检查 Elasticsearch
    try:
        store = get_store()
        store.es.info()
        services_status["elasticsearch"] = "ok"
    except Exception as e:
        services_status["elasticsearch"] = f"error: {str(e)}"

    # 检查 Embedding 服务
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"http://localhost:{settings.embedding_port}/health")
            services_status["embedding_service"] = (
                "ok" if response.status_code == 200 else "error"
            )
    except Exception as e:
        services_status["embedding_service"] = f"error: {str(e)}"

    # 检查 Rerank 服务
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"http://localhost:{settings.rerank_port}/health")
            services_status["rerank_service"] = (
                "ok" if response.status_code == 200 else "error"
            )
    except Exception as e:
        services_status["rerank_service"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in services_status.values())

    return HealthResponse(
        status="healthy" if all_ok else "degraded", services=services_status
    )


# ============================================================
# RAG 问答接口
# ============================================================


@app.post("/api/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    RAG 知识库问答接口

    流程：Query Rewrite → 混合检索 → Rerank → LLM 生成回答
    """
    timing = {}
    frontend_time = time.time()

    try:
        retrieval_svc = get_retrieval_service()
        rewrite_svc = get_query_rewrite_service()
        generation_svc = get_generation_service()

        # ---- 1. Query Rewrite ----
        rewritten: Optional[RewrittenQuery] = None
        timing["query_rewrite"] = 0

        if request.use_rewrite:
            t0 = time.time()
            rewritten = rewrite_svc.rewrite(request.query)
            timing["query_rewrite"] = time.time() - t0

            # 解析 intent_type: 支持逗号分隔多值
            intent_types = None
            if rewritten.intent_type and rewritten.intent_type != "other":
                intent_types = [
                    t.strip() for t in rewritten.intent_type.split(",") if t.strip()
                ]

            options = RetrievalOptions(
                top_k=request.top_k,
                target_models=(
                    rewritten.target_entities if rewritten.target_entities else None
                ),
                keywords=rewritten.keywords if rewritten.keywords else None,
                chunk_types=intent_types,
                use_rerank=request.use_rerank,
                rerank_top_k=request.rerank_top_k,
                min_score=request.min_score,
            )
            search_query = rewritten.rewritten_query

            # 三路分流: direct / parallel / sequential
            strategy = getattr(rewritten, "strategy", "direct")

            if strategy == "direct":
                # 直接检索（simple 或 strategy=direct）
                logger.info(f"[路由] strategy=direct → 普通混合检索")
                retrieval_result = retrieval_svc.search(
                    query=search_query,
                    options=options,
                    use_hybrid=True,
                )

            elif strategy == "parallel" and rewritten.sub_queries:
                # 独立子查询 → search_routed 并行检索
                logger.info(
                    f"[路由] strategy=parallel → search_routed "
                    f"({len(rewritten.sub_queries)} sub_queries)"
                )
                retrieval_result = retrieval_svc.search_routed(
                    rewritten_query=rewritten.rewritten_query,
                    sub_queries=rewritten.sub_queries,
                    options=options,
                )

            else:
                # 多步推理 → ReAct（限制轮次）
                logger.info(f"[路由] strategy=sequential → ReAct")
                react_svc = get_react_reasoning_service()
                retrieval_result = react_svc.reason(
                    original_query=request.query,
                    rewritten=rewritten,
                    options=options,
                )
        else:
            options = RetrievalOptions(
                top_k=request.top_k,
                use_rerank=request.use_rerank,
                rerank_top_k=request.rerank_top_k,
                min_score=request.min_score,
            )
            search_query = request.query

            retrieval_result = retrieval_svc.search(
                query=search_query,
                options=options,
                use_hybrid=True,
            )

        # 合并检索时间
        timing.update(retrieval_result.timing)
        chunks = retrieval_result.chunks

        if not chunks:
            raise HTTPException(
                status_code=404, detail="未找到相关内容，请尝试其他问题或上传相关文档"
            )

        # ---- 2. 生成回答 ----
        gen_kwargs = {
            "query": request.query,
            "chunks": chunks,
            "chat_history": None,
        }
        if rewritten:
            gen_kwargs["query_intent"] = rewritten.intent_type
            gen_kwargs["query_entities"] = rewritten.entities

        gen_start = time.time()
        answer, usage = generation_svc.generate(**gen_kwargs)
        timing["generation"] = time.time() - gen_start

        # 总时间
        total_time = time.time() - frontend_time
        timing["total"] = total_time

        # 记录请求参数到 timing（用于日志）
        timing["top_k"] = request.top_k
        timing["use_rerank"] = request.use_rerank
        timing["rerank_top_k"] = request.rerank_top_k or len(chunks)

        # 构建响应
        sources = chunks_to_sources(chunks)

        logger.info(
            f"问答成功: query='{request.query}', "
            f"chunks={len(chunks)}, "
            f"tokens={usage.total_tokens}, "
            f"time={total_time:.1f}s"
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            time=timing,
            usage=usage,
            chunks_count=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


# ============================================================
# 流式 RAG 问答接口 (SSE)
# ============================================================


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式 RAG 问答 — SSE 端点

    事件格式:
    - event: sources  data: {sources: [...]}
    - event: token    data: {content: "..."}
    - event: done     data: {usage: {...}, time: {...}}
    - event: error    data: {error: "..."}
    """
    import openai
    from config import settings

    async def event_stream():
        timing = {}
        frontend_time = time.time()

        try:
            retrieval_svc = get_retrieval_service()
            rewrite_svc = get_query_rewrite_service()
            generation_svc = get_generation_service()

            # ---- 1. Query Rewrite ----
            rewritten: Optional[RewrittenQuery] = None
            timing["query_rewrite"] = 0

            if request.use_rewrite:
                t0 = time.time()
                rewritten = rewrite_svc.rewrite(request.query)
                timing["query_rewrite"] = time.time() - t0

                intent_types = None
                if rewritten.intent_type and rewritten.intent_type != "other":
                    intent_types = [
                        t.strip() for t in rewritten.intent_type.split(",") if t.strip()
                    ]

                options = RetrievalOptions(
                    top_k=request.top_k,
                    target_models=(
                        rewritten.target_entities if rewritten.target_entities else None
                    ),
                    keywords=rewritten.keywords if rewritten.keywords else None,
                    chunk_types=intent_types,
                    use_rerank=request.use_rerank,
                    rerank_top_k=request.rerank_top_k,
                    min_score=request.min_score,
                )
                search_query = rewritten.rewritten_query

                strategy = getattr(rewritten, "strategy", "direct")
                if strategy == "direct":
                    retrieval_result = retrieval_svc.search(
                        query=search_query, options=options, use_hybrid=True
                    )
                elif strategy == "parallel" and rewritten.sub_queries:
                    retrieval_result = retrieval_svc.search_routed(
                        rewritten_query=rewritten.rewritten_query,
                        sub_queries=rewritten.sub_queries,
                        options=options,
                    )
                else:
                    react_svc = get_react_reasoning_service()
                    retrieval_result = react_svc.reason(
                        original_query=request.query,
                        rewritten=rewritten,
                        options=options,
                    )
            else:
                options = RetrievalOptions(
                    top_k=request.top_k,
                    use_rerank=request.use_rerank,
                    rerank_top_k=request.rerank_top_k,
                    min_score=request.min_score,
                )
                search_query = request.query
                retrieval_result = retrieval_svc.search(
                    query=search_query, options=options, use_hybrid=True
                )

            timing.update(retrieval_result.timing)
            chunks = retrieval_result.chunks

            if not chunks:
                yield f"event: error\ndata: {json.dumps({'error': '未找到相关内容'}, ensure_ascii=False)}\n\n"
                return

            # ---- 2. 发送 sources 事件 ----
            sources = chunks_to_sources(chunks)
            sources_data = [s.model_dump() for s in sources]
            yield f"event: sources\ndata: {json.dumps({'sources': sources_data}, ensure_ascii=False)}\n\n"

            # ---- 3. 流式生成 ----
            gen_kwargs = {
                "query": request.query,
                "chunks": chunks,
                "chat_history": None,
            }
            if rewritten:
                gen_kwargs["query_intent"] = rewritten.intent_type
                gen_kwargs["query_entities"] = rewritten.entities

            gen_start = time.time()
            for token in generation_svc.generate_stream(**gen_kwargs):
                yield f"event: token\ndata: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"

            timing["generation"] = time.time() - gen_start
            timing["total"] = time.time() - frontend_time

            # ---- 4. 发送 done 事件 ----
            yield f"event: done\ndata: {json.dumps({'time': timing, 'chunks_count': len(chunks)}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"流式问答失败: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 文档管理接口
# ============================================================


@app.get("/api/v1/documents")
async def list_documents(page: int = 1, page_size: int = 20):
    """文档列表（分页）"""
    try:
        store = get_store()
        result = store.list_documents(page=page, page_size=page_size)
        return result
    except Exception as e:
        logger.error(f"文档列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


# ============================================================
# PDF 解析代理接口
# ============================================================


@app.post("/api/v1/parse")
async def parse_pdf(file: UploadFile = File(..., description="PDF 文件")):
    """
    PDF 解析代理 — 转发到 MinerU 服务 (port 8003)

    外部只需调用此接口，无需直连 MinerU 服务。
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    file_content = await file.read()
    if len(file_content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小不能超过 100MB")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"http://localhost:{settings.mineru_port}/parse",
                files={"file": (file.filename, file_content, "application/pdf")},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="MinerU 服务未启动")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@app.get("/api/v1/parse/status/{parse_id}")
async def get_parse_status(parse_id: str):
    """查询 PDF 解析结果"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"http://localhost:{settings.mineru_port}/parse/{parse_id}"
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="MinerU 服务未启动")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


# ============================================================
# 检索接口
# ============================================================


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """检索接口（可选 Query Rewrite）"""
    import time as _time

    try:
        retrieval_svc = get_retrieval_service()
        timing = {}

        # ---- Query Rewrite（可选） ----
        rewritten = None
        rewrite_data = None
        search_query = request.query

        if request.use_rewrite:
            t0 = _time.time()
            rewrite_svc = get_query_rewrite_service()
            rewritten = rewrite_svc.rewrite(request.query)
            timing["query_rewrite"] = _time.time() - t0

            rewrite_data = {
                "original_query": rewritten.original_query,
                "rewritten_query": rewritten.rewritten_query,
                "strategy": rewritten.strategy,
                "intent_type": rewritten.intent_type,
                "target_entities": rewritten.target_entities,
                "keywords": rewritten.keywords,
                "sub_queries": rewritten.sub_queries,
            }

            search_query = rewritten.rewritten_query

            # 构建 options（含 rewrite 提取的实体、关键词、意图）
            intent_types = None
            if rewritten.intent_type and rewritten.intent_type != "other":
                intent_types = [
                    t.strip() for t in rewritten.intent_type.split(",") if t.strip()
                ]

            options = RetrievalOptions(
                top_k=request.top_k,
                target_models=(
                    rewritten.target_entities if rewritten.target_entities else None
                ),
                keywords=rewritten.keywords if rewritten.keywords else None,
                chunk_types=intent_types,
                use_rerank=request.use_rerank,
                min_score=request.min_score,
            )

            # 三路路由
            strategy = rewritten.strategy
            if strategy == "sequential":
                react_svc = get_react_reasoning_service()
                result = react_svc.reason(
                    original_query=request.query,
                    rewritten=rewritten,
                    options=options,
                )
            elif strategy == "parallel" and rewritten.sub_queries:
                result = retrieval_svc.search_routed(
                    rewritten_query=rewritten.rewritten_query,
                    sub_queries=rewritten.sub_queries,
                    options=options,
                )
            else:
                result = retrieval_svc.search(
                    query=search_query,
                    options=options,
                    use_hybrid=True,
                )
        else:
            options = RetrievalOptions(
                top_k=request.top_k,
                use_rerank=request.use_rerank,
                min_score=request.min_score,
            )
            result = retrieval_svc.search(
                query=search_query,
                options=options,
                use_hybrid=True,
            )

        timing.update(result.timing)

        # 批量获取 doc_id → title
        doc_ids = list({c.doc_id for c in result.chunks})
        store = get_store()
        doc_titles = store.get_doc_titles(doc_ids)

        # 转换 chunk 为 dict
        chunks_dict = []
        for chunk in result.chunks:
            chunks_dict.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": doc_titles.get(chunk.doc_id, ""),
                    "content": (
                        chunk.content[:300] + "..."
                        if len(chunk.content) > 300
                        else chunk.content
                    ),
                    "section_title": chunk.section_title,
                    "chunk_type": chunk.chunk_type,
                    "doc_type": chunk.doc_type,
                    "filter_terms": chunk.filter_terms,
                    "score": chunk.score,
                }
            )

        return SearchResponse(
            query=request.query,
            total=len(result.chunks),
            chunks=chunks_dict,
            timing=timing,
            rewrite=rewrite_data,
        )

    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
