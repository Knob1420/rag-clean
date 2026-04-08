"""
LLM 调用 — 从 generation.py 精简

只保留数据层需要的：
- _call_deepseek() — 基础 LLM 调用
- _parse_json_response() — JSON 解析
- extract_doc_info() — 供 analyzer 调用
- extract_chunk_info_batch() — 供 pipeline 调用
"""

import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from config import settings
from prompt import (
    DOC_INFO_SYSTEM_PROMPT,
    CHUNK_INFO_SYSTEM_PROMPT,
    build_doc_info_prompt,
    build_chunk_info_prompt,
)


class LLMClient:
    """DeepSeek LLM 客户端"""

    def __init__(self):
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url
        self.model = settings.deepseek_model

    def call(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """
        调用 DeepSeek API。

        Args:
            messages: 消息列表
            temperature: 温度参数

        Returns:
            LLM 回复文本
        """
        if not self.api_key:
            logger.warning("使用 Mock 模式（未配置 API Key）")
            return '{"doc_type": "其他", "domain": "Product_Tech", "entities": {}, "filter_terms": [], "topics": [], "doc_intent": null, "summary": "暂无摘要", "confidence": 0}'

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=2000,
        )
        content = response.choices[0].message.content
        logger.info(
            f"LLM 调用成功: "
            f"prompt_tokens={response.usage.prompt_tokens}, "
            f"completion_tokens={response.usage.completion_tokens}"
        )
        return content

    def extract_doc_info(self, doc_id: str, title: str, content: str) -> Dict[str, Any]:
        """
        提取文档级信息：doc_type + domain + entities + filter_terms + summary

        Args:
            doc_id: 文档 ID（用于日志）
            title: 文档标题
            content: 文档内容预览

        Returns:
            {"doc_type": str, "domain": str, "entities": dict,
             "filter_terms": list, "topics": list, "doc_intent": str,
             "summary": str, "confidence": int}
        """
        content_preview = content[:4000]
        if len(content) > 4000:
            content_preview += "\n...(内容已截断)"

        prompt = build_doc_info_prompt(title, content_preview)

        try:
            response = self.call(
                [
                    {
                        "role": "system",
                        "content": DOC_INFO_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return parse_json_response(response)
        except Exception as e:
            logger.error(f"文档信息提取失败 {doc_id}: {e}")
            return {
                "doc_type": "其他",
                "domain": "Product_Tech",
                "entities": {},
                "filter_terms": [],
                "topics": [],
                "doc_intent": None,
                "summary": "暂无摘要",
                "confidence": 0,
            }

    def extract_chunk_info_batch(
        self,
        chunks: List[Dict[str, Any]],
        doc_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        批量提取 chunk 信息（section_title, chunk_type, keywords, context_summary）。

        Args:
            chunks: chunk 列表，每个包含 chunk_index, content, section_title
            doc_context: 文档级上下文

        Returns:
            chunk 信息列表
        """
        chunks_content = []
        for chunk in chunks:
            index = chunk.get("chunk_index", 0)
            content = chunk.get("content", "")
            orig_title = chunk.get("section_title") or "（无）"
            chunks_content.append(f"[Chunk {index}]（原标题：{orig_title}）\n{content}")

        prompt = build_chunk_info_prompt(chunks_content, doc_context, len(chunks))

        try:
            response = self.call(
                [
                    {
                        "role": "system",
                        "content": CHUNK_INFO_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            result = parse_json_response(response)
            return result.get("chunks", [])
        except Exception as e:
            logger.warning(f"Chunk 信息提取失败: {e}，使用默认值")
            return [
                {
                    "index": i,
                    "section_title": None,
                    "chunk_type": "other",
                    "keywords": [],
                    "context_summary": chunk.get("content", "")[:100] + "...",
                }
                for i, chunk in enumerate(chunks)
            ]


# ── JSON 解析 ──────────────────────────────────────────


def parse_json_response(response: str) -> Dict[str, Any]:
    """解析 LLM 返回的 JSON"""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # 尝试提取 JSON 代码块
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        # 尝试提取花括号内容
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError(f"无法解析 JSON: {response[:200]}")


# ── 全局实例 ──────────────────────────────────────────

_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
