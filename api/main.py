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
from typing import Optional, List, Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from config import settings
from core.generation import get_generation_service
from core.pipeline.query_rewrite_retrieval import QueryRewriteRetrievalPipeline
from core.query_engineer.query_rewrite import QueryRewriteServiceV2
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
from core.products.specs_service import query_specs, get_specs
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
    # chunk.doc_title 已在 RetrievalService 中从 ES 填充，直接使用
    sources = []
    for chunk in chunks:
        snippet = (
            chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
        )
        sources.append(
            SourceInfo(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_name=chunk.doc_title or "",
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

    流程：Query Understanding → Query Rewrite → 混合检索 → Rerank → LLM 生成回答
    """
    timing = {}
    frontend_time = time.time()

    try:
        generation_svc = get_generation_service()
        pipeline = QueryRewriteRetrievalPipeline()
        query_rewrite_svc = QueryRewriteServiceV2()

        # ---- Query Rewrite + Retrieval Pipeline ----
        t0 = time.time()
        pipeline_result = pipeline.run(
            query=request.query,
            top_k=request.top_k,
            use_rewrite=request.use_rewrite,
            use_rerank=request.use_rerank,
            rerank_top_k=request.rerank_top_k,
        )
        timing = pipeline_result.timing
        timing["total"] = time.time() - frontend_time

        # ---- 生成回答（多子问句合并逻辑）----
        answer = None
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        if pipeline_result.chunks:
            understanding = pipeline_result.understanding_result
            sub_queries = understanding.sub_queries if understanding else []

            if len(sub_queries) == 1:
                # 单子问句：直接生成
                sq = sub_queries[0]
                sq_chunks = pipeline_result.per_sub_question_chunks.get(sq.query, pipeline_result.chunks)
                sq_spec = pipeline_result.per_sub_question_spec_context.get(sq.query, "")
                sq_constraints = pipeline_result.per_sub_question_generation_constraints.get(sq.query, [])
                answer, usage = generation_svc.generate(
                    query=request.query,
                    chunks=sq_chunks,
                    query_intent=sq.intent,
                    spec_context=sq_spec,
                    generation_constraints=sq_constraints,
                )
            else:
                # 多子问句：逐条生成 → 合并 → 整合生成
                merged_answers = []
                all_chunks = []
                for sq in sub_queries:
                    sq_chunks = pipeline_result.per_sub_question_chunks.get(sq.query, [])
                    if not sq_chunks:
                        continue
                    all_chunks.extend(sq_chunks)
                    sq_spec = pipeline_result.per_sub_question_spec_context.get(sq.query, "")
                    sq_constraints = pipeline_result.per_sub_question_generation_constraints.get(sq.query, [])
                    sub_answer, sub_usage = generation_svc.generate(
                        query=sq.query,
                        chunks=sq_chunks,
                        query_intent=sq.intent,
                        spec_context=sq_spec,
                        generation_constraints=sq_constraints,
                    )
                    merged_answers.append(f"【{sq.query}】\n{sub_answer}")

                if merged_answers:
                    # 整合生成
                    integrated = "\n\n---\n\n".join(merged_answers)
                    answer, usage = generation_svc.generate(
                        query=request.query,
                        chunks=all_chunks,
                        spec_context="",
                    )
                    # 使用整合 prompt
                    system_prompt, user_prompt = generation_svc._build_integration_prompt(
                        original_query=request.query,
                        merged_answers=integrated,
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    # 直接调用 LLM 做整合
                    import openai
                    from config import settings
                    if settings.deepseek_api_key:
                        client = openai.OpenAI(
                            api_key=settings.deepseek_api_key,
                            base_url=settings.deepseek_base_url,
                        )
                        response = client.chat.completions.create(
                            model=settings.deepseek_model,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=2000,
                        )
                        answer = response.choices[0].message.content
                        usage = TokenUsage(
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                        )

        if not answer:
            raise HTTPException(
                status_code=404,
                detail="未找到相关内容，请尝试其他问题或上传相关文档",
            )

        # 构建响应
        sources = chunks_to_sources(pipeline_result.chunks)

        logger.info(
            f"问答成功: query='{request.query}', "
            f"sub_queries={len(sub_queries)}, "
            f"chunks={len(sources)}, "
            f"time={timing.get('total', 0):.1f}s"
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            time=timing,
            usage=usage,
            chunks_count=len(sources),
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
    流式 RAG 问答 — SSE 端点（使用 V2 Pipeline）

    事件格式:
    - event: sources  data: {sources: [...]}
    - event: token    data: {content: "..."}
    - event: done     data: {usage: {...}, time: {...}}
    - event: error    data: {error: "..."}
    """
    from core.pipeline import QueryRewriteRetrievalPipeline

    async def event_stream():
        timing = {}
        frontend_time = time.time()

        try:
            pipeline = QueryRewriteRetrievalPipeline()
            generation_svc = get_generation_service()

            # ---- 1. Query Rewrite + Retrieval (V2 Pipeline) ----
            t0 = time.time()
            pipeline_result = pipeline.run(
                query=request.query,
                top_k=request.top_k,
                use_rewrite=request.use_rewrite,
                use_rerank=request.use_rerank,
                rerank_top_k=request.rerank_top_k,
            )
            timing["query_rewrite"] = pipeline_result.timing.get("rewrite", 0)
            timing["retrieve"] = pipeline_result.timing.get("retrieve", 0)

            chunks = pipeline_result.chunks
            spec_context = pipeline_result.spec_context

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
                "query_intent": pipeline_result.intent,
            }
            # 如果有结构化查询结果，追加到 context
            if spec_context:
                gen_kwargs["spec_context"] = spec_context

            gen_start = time.time()
            async for token in generation_svc.async_generate_stream(**gen_kwargs):
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
# 产品参数查询
# ============================================================


@app.get("/api/v1/products/specs")
async def get_product_specs(
    product: Optional[str] = None,
    model: Optional[str] = None,
    filter: Optional[str] = None,
):
    """
    产品参数查询接口

    Args:
        product: 产品类型（星载智算机 / 星载路由器 / 星载激光通信机）
        model: 型号（如 NX1, G1, 智加G3）
        filter: 过滤条件，支持 <, <=, >, >=, ==（如 "重量<3kg", "算力>=10"）

    Examples:
        GET /api/v1/products/specs                           # 返回所有产品
        GET /api/v1/products/specs?product=星载智算机         # 返回智算机全系
        GET /api/v1/products/specs?product=星载智算机&model=NX1  # 返回 NX1
        GET /api/v1/products/specs?filter=重量<3kg           # 返回重量<3kg的型号
    """
    try:
        results = query_specs(product=product, model=model, filter_field=filter)
        return {
            "total": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"产品参数查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.get("/api/v1/products")
async def list_products():
    """返回所有产品类型和型号列表"""
    specs = get_specs()
    return {
        "products": [
            {
                "type": prod_type,
                "models": list(models.keys()) if isinstance(models, dict) else [],
            }
            for prod_type, models in specs.items()
        ]
    }


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
