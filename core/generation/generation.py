"""
RAG 回答生成服务

基于 rag-knowledge-base generation.py，适配 rag-clean：
- 复用 llm.py 的 LLMClient + parse_json_response，不再内部实现
- Prompt 模板: [章节] 引用代替 P[页码]
- _build_context 使用 chunk.doc_title，parent context 在 pipeline 中注入
"""

from typing import Dict, Generator, List, Optional, Tuple

from loguru import logger

from core.generation.llm import get_llm_client
from core.router.models import (
    INTENT_SIMPLE_LOOKUP,
    INTENT_COMPARE,
    INTENT_RECOMMEND,
    INTENT_AGGREGATE,
)
from core.products.specs_service import build_specs_context
from prompt import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
    SIMPLE_LOOKUP_SYSTEM_PROMPT,
    SIMPLE_LOOKUP_USER_PROMPT_TEMPLATE,
    COMPARE_SYSTEM_PROMPT,
    COMPARE_USER_PROMPT,
    RECOMMEND_SYSTEM_PROMPT,
    RECOMMEND_USER_PROMPT,
    AGGREGATE_SYSTEM_PROMPT,
    AGGREGATE_USER_PROMPT,
)
from core.retrieve.retrieval_models import RetrievedChunk, TokenUsage


class GenerationService:
    """RAG 回答生成服务"""

    def __init__(self):
        self.llm = get_llm_client()

    # ============================================================
    # 上下文构建
    # ============================================================

    def _build_context(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        max_context_length: int = 10000,
        spec_context: str = "",
    ) -> str:
        """
        构建上下文（parent context 已在 pipeline 中注入）

        Args:
            query: 用户查询
            chunks: 检索到的文档块
            max_context_length: 最大上下文长度
            spec_context: 结构化产品参数字符串（来自 pipeline 的 dual-path 检索）
                         如果提供，优先使用；否则使用关键词检测作为后备
        """
        if not chunks:
            return "知识库中暂无相关内容。"

        # 优先使用 pipeline 传来的 spec_context（LLM-based intent routing 结果）
        # 如果没有，则使用后备的关键词检测
        if not spec_context:
            spec_context = build_specs_context(query)

        context_parts = []
        current_length = 0

        # 先加入产品参数上下文
        if spec_context:
            context_parts.append(spec_context)
            current_length += len(spec_context)

        for i, chunk in enumerate(chunks):
            # 来源标注 — 直接使用 chunk.doc_title
            doc_name = chunk.doc_title or chunk.doc_id
            source_info = f"[来源: {doc_name}]"

            # 内容已在 pipeline 中注入 parent context，直接使用
            chunk_text = f"{source_info}\n{chunk.content}"

            if current_length + len(chunk_text) > max_context_length:
                logger.info(
                    f"上下文长度达到上限 {max_context_length}，截断: "
                    f"第 {i+1}/{len(chunks)} 个 chunk"
                )
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n\n---\n\n".join(context_parts)

    # ============================================================
    # Prompt 构建
    # ============================================================

    def _build_rag_prompt(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        query_intent: Optional[str] = None,
        spec_context: str = "",
        generation_constraints: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """构建 RAG Prompt（根据 intent 选择不同模板）"""
        if generation_constraints is None:
            generation_constraints = []

        context = self._build_context(query, chunks, spec_context=spec_context)
        intent_context = self._build_intent_context(query_intent)

        # 根据 intent 选择不同模板
        if query_intent == INTENT_COMPARE:
            system_prompt = COMPARE_SYSTEM_PROMPT
            user_prompt = COMPARE_USER_PROMPT.format(
                query=query,
                context=context,
            )
        elif query_intent == INTENT_RECOMMEND:
            system_prompt = RECOMMEND_SYSTEM_PROMPT
            user_prompt = RECOMMEND_USER_PROMPT.format(
                comparison_matrix=context,
                query=query,
            )
        elif query_intent == INTENT_AGGREGATE:
            system_prompt = AGGREGATE_SYSTEM_PROMPT
            user_prompt = AGGREGATE_USER_PROMPT.format(
                structured_data=context,
                query=query,
            )
        elif query_intent == INTENT_SIMPLE_LOOKUP:
            # simple_lookup 使用专用模板
            system_prompt = SIMPLE_LOOKUP_SYSTEM_PROMPT
            user_prompt = SIMPLE_LOOKUP_USER_PROMPT_TEMPLATE.format(
                context=context,
                query=query,
            )
        else:
            # 默认使用通用 RAG 模板
            system_prompt = RAG_SYSTEM_PROMPT
            user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
                context=context,
                intent_context=intent_context,
                query=query,
            )

        # 追加生成约束（翻译、字数限制等）
        if generation_constraints:
            constraint_str = "\n".join([f"- {c}" for c in generation_constraints])
            user_prompt += f"\n\n【回答格式要求】\n{constraint_str}"

        return system_prompt, user_prompt

    def _build_intent_context(
        self,
        query_intent: Optional[str],
    ) -> str:
        """构建意图上下文"""
        if not query_intent:
            return ""

        parts = []

        if query_intent:
            parts.append(f"## 查询意图\n意图类型: {query_intent}")

        return "\n".join(parts) + "\n" if parts else ""

    # ============================================================
    # 生成回答
    # ============================================================

    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        query_intent: Optional[str] = None,
        spec_context: str = "",
        generation_constraints: Optional[List[str]] = None,
    ) -> Tuple[str, TokenUsage]:
        """
        生成 RAG 回答

        Args:
            query: 用户查询
            chunks: 检索到的 chunks
            query_intent: 查询意图类型
            spec_context: 结构化产品参数字符串（来自 pipeline dual-path 检索）
            generation_constraints: 生成约束列表（如 ["翻译成英文", "不超过50字"]）

        Returns:
            (回复内容, TokenUsage)
        """
        if generation_constraints is None:
            generation_constraints = []

        # 构建 Prompt
        system_prompt, user_prompt = self._build_rag_prompt(
            query=query,
            chunks=chunks,
            query_intent=query_intent,
            spec_context=spec_context,
            generation_constraints=generation_constraints,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 调用 LLM
        try:
            import openai
            from config import settings

            if not settings.deepseek_api_key:
                return (
                    "抱歉，当前使用 Mock 模式。请配置 API Key 以获得真实回复。",
                    TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                )

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

            content = response.choices[0].message.content
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

            logger.info(
                f"LLM 生成成功: "
                f"prompt_tokens={usage.prompt_tokens}, "
                f"completion_tokens={usage.completion_tokens}"
            )

            return content, usage

        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return (
                f"抱歉，生成回答时出现错误: {str(e)}",
                TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

    # ============================================================
    # 流式生成回答
    # ============================================================

    def generate_stream(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        query_intent: Optional[str] = None,
        query_entities: Optional[Dict[str, str]] = None,
        intent: Optional[str] = None,
        spec_context: str = "",
    ) -> Generator[str, None, None]:
        """
        流式生成 RAG 回答

        Yields:
            逐个 token 字符串
        """
        import openai
        from config import settings

        if not settings.deepseek_api_key:
            yield "抱歉，当前使用 Mock 模式。请配置 API Key 以获得真实回复。"
            return

        # 构建 Prompt
        system_prompt, user_prompt = self._build_rag_prompt(
            query=query,
            chunks=chunks,
            query_intent=intent or query_intent,
            spec_context=spec_context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            client = openai.OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
            stream = client.chat.completions.create(
                model=settings.deepseek_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

        except Exception as e:
            logger.error(f"LLM 流式生成失败: {e}")
            yield f"\n\n抱歉，生成回答时出现错误: {str(e)}"

    # ============================================================
    # 异步流式生成回答
    # ============================================================

    async def async_generate_stream(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        query_intent: Optional[str] = None,
        query_entities: Optional[Dict[str, str]] = None,
        intent: Optional[str] = None,
        spec_context: str = "",
    ):
        """
        异步流式生成 RAG 回答（使用 AsyncOpenAI，不阻塞事件循环）

        Yields:
            逐个 token 字符串
        """
        from openai import AsyncOpenAI
        from config import settings

        if not settings.deepseek_api_key:
            yield "抱歉，当前使用 Mock 模式。请配置 API Key 以获得真实回复。"
            return

        # 构建 Prompt
        system_prompt, user_prompt = self._build_rag_prompt(
            query=query,
            chunks=chunks,
            query_intent=intent or query_intent,
            spec_context=spec_context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            client = AsyncOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
            )
            stream = await client.chat.completions.create(
                model=settings.deepseek_model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

        except Exception as e:
            logger.error(f"LLM 异步流式生成失败: {e}")
            yield f"\n\n抱歉，生成回答时出现错误: {str(e)}"


# ── 全局实例 ──────────────────────────────────────────

_generation_service: Optional[GenerationService] = None


def get_generation_service() -> GenerationService:
    """获取生成服务单例"""
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service
