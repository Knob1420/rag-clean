"""
RAG 回答生成服务

基于 rag-knowledge-base generation.py，适配 rag-clean：
- 复用 llm.py 的 LLMClient + parse_json_response，不再内部实现
- Prompt 模板: [章节] 引用代替 P[页码]
- _build_context 适配扁平字段结构，parent 内容通过 get_store().get_parent() 加载
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from core.generation.llm import LLMClient, get_llm_client
from prompt import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE, MULTI_TURN_SYSTEM_PROMPT
from core.retrieve.retrieval_models import RetrievedChunk, TokenUsage
from store import get_store


class GenerationService:
    """RAG 回答生成服务"""

    def __init__(self):
        self.llm = get_llm_client()

    # ============================================================
    # 上下文构建
    # ============================================================

    def _build_context(
        self,
        chunks: List[RetrievedChunk],
        max_context_length: int = 4000,
        use_parent_context: bool = True,
    ) -> str:
        """
        构建上下文

        适配 rag-clean 扁平字段：
        - 批量查询 doc_id → title，显示文档标题
        - parent 内容通过 get_store().get_parent() 加载
        """
        if not chunks:
            return "知识库中暂无相关内容。"

        store = get_store()

        # 批量获取 doc_id → title 映射
        doc_ids = list({c.doc_id for c in chunks})
        doc_titles = store.get_doc_titles(doc_ids)

        context_parts = []
        current_length = 0

        for chunk in chunks:
            # 来源标注 — 使用文档标题
            doc_name = doc_titles.get(chunk.doc_id, chunk.doc_id)
            source_info = f"[来源: {doc_name}"
            if chunk.section_title:
                source_info += f" | 章节: {chunk.section_title}"
            if chunk.chunk_type:
                source_info += f" | 类型: {chunk.chunk_type}"
            source_info += "]"

            # 构建内容
            content_parts = []

            # 加载父块内容扩展上下文
            if use_parent_context and chunk.parent_id:
                parent_data = store.get_parent(chunk.parent_id)
                if parent_data and parent_data.get("content"):
                    content_parts.append(
                        f"[父块上下文]\n{parent_data['content']}"
                    )

            content_parts.append(f"[检索内容]\n{chunk.content}")
            full_content = "\n\n".join(content_parts)

            chunk_text = f"{source_info}\n{full_content}"

            if current_length + len(chunk_text) > max_context_length:
                logger.info(f"上下文长度达到上限 {max_context_length}，截断")
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
        query_entities: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """构建 RAG Prompt"""
        context = self._build_context(chunks)
        intent_context = self._build_intent_context(query_intent, query_entities)

        system_prompt = RAG_SYSTEM_PROMPT
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            intent_context=intent_context,
            query=query,
        )

        return system_prompt, user_prompt

    def _build_multi_turn_prompt(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        chat_history: List[Dict[str, str]],
        query_intent: Optional[str] = None,
        query_entities: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, str]:
        """构建多轮对话 Prompt"""
        context = self._build_context(chunks)
        intent_context = self._build_intent_context(query_intent, query_entities)

        history_lines = []
        for msg in chat_history[-6:]:
            role = "用户" if msg["role"] == "user" else "助手"
            history_lines.append(f"{role}: {msg['content']}")

        chat_history_str = "\n".join(history_lines) if history_lines else "无历史对话"

        system_prompt = MULTI_TURN_SYSTEM_PROMPT.format(
            chat_history=chat_history_str,
        )

        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            intent_context=intent_context,
            query=query,
        )

        return system_prompt, user_prompt

    def _build_intent_context(
        self,
        query_intent: Optional[str],
        query_entities: Optional[Dict[str, str]],
    ) -> str:
        """构建意图上下文"""
        if not query_intent and not query_entities:
            return ""

        parts = []

        if query_intent:
            parts.append(f"## 查询意图\n意图类型: {query_intent}")

        if query_entities:
            entity_str = "、".join([f"{k}: {v}" for k, v in query_entities.items()])
            parts.append(f"## 提取实体\n{entity_str}")

        return "\n".join(parts) + "\n" if parts else ""

    # ============================================================
    # 生成回答
    # ============================================================

    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        chat_history: Optional[List[Dict[str, str]]] = None,
        query_intent: Optional[str] = None,
        query_entities: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, TokenUsage]:
        """
        生成 RAG 回答

        Returns:
            (回复内容, TokenUsage)
        """
        # 构建 Prompt
        if chat_history and len(chat_history) > 0:
            system_prompt, user_prompt = self._build_multi_turn_prompt(
                query=query,
                chunks=chunks,
                chat_history=chat_history,
                query_intent=query_intent,
                query_entities=query_entities,
            )
        else:
            system_prompt, user_prompt = self._build_rag_prompt(
                query=query,
                chunks=chunks,
                query_intent=query_intent,
                query_entities=query_entities,
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
                    TokenUsage(
                        prompt_tokens=0, completion_tokens=0, total_tokens=0
                    ),
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
                TokenUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            )


# ── 全局实例 ──────────────────────────────────────────

_generation_service: Optional[GenerationService] = None


def get_generation_service() -> GenerationService:
    """获取生成服务单例"""
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service
