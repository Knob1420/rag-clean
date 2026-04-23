"""
LLM 调用 — 从 generation.py 精简

只保留数据层需要的：
- _call_deepseek() — 基础 LLM 调用
- _parse_json_response() — JSON 解析
- extract_doc_info() — 供 analyzer 调用
- extract_chunk_info_batch() — 供 pipeline 调用
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from config import settings
from prompt import (
    SUMMARY_SYSTEM_PROMPT,
    build_summary_prompt,
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

    def generate_summary(self, content: str) -> Dict[str, Any]:
        """
        为文档片段生成 summary 和 primary_entity。

        Args:
            content: 文档内容

        Returns:
            {"summary": str, "primary_entity": str}
        """
        prompt = build_summary_prompt(content)

        try:
            response = self.call(
                [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
            result = parse_json_response(response)
            return {
                "summary": result.get("summary", ""),
                "primary_entity": result.get("primary_entity", ""),
            }
        except Exception as e:
            logger.warning(f"Summary 生成失败: {e}，使用默认值")
            return {
                "summary": content[:100] + "...",
                "primary_entity": "",
            }

    async def _call_async(self, messages: List[Dict[str, str]]) -> str:
        """异步调用 DeepSeek API"""
        import httpx

        if not self.api_key:
            return '{"summary": "暂无摘要", "primary_entity": ""}'

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": settings.deepseek_model,
                    "messages": messages,
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def generate_summary_batch(
        self, contents: List[str], max_concurrency: int = 10
    ) -> List[Dict[str, Any]]:
        """
        批量为多个文档片段生成 summary（asyncio 并发）。

        Args:
            contents: 文档内容列表
            max_concurrency: 最大并发数，默认 10

        Returns:
            [{"summary": str, "primary_entity": str}, ...]
        """
        if not contents:
            return []

        async def _run() -> List[Dict[str, Any]]:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def _call_one(idx: int, content: str) -> tuple[int, Dict[str, Any]]:
                async with semaphore:
                    prompt = build_summary_prompt(content)
                    messages = [
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                    try:
                        response = await self._call_async(messages)
                        result = parse_json_response(response)
                        return idx, {
                            "summary": result.get("summary", ""),
                            "primary_entity": result.get("primary_entity", ""),
                        }
                    except Exception as e:
                        logger.warning(f"Summary 批量生成失败 [{idx}]: {e}，使用默认值")
                        return idx, {
                            "summary": content[:100] + "...",
                            "primary_entity": "",
                        }

            tasks = [_call_one(i, c) for i, c in enumerate(contents)]
            results = await asyncio.gather(*tasks)
            # 按原始顺序返回
            return [r for _, r in sorted(results, key=lambda x: x[0])]

        return asyncio.run(_run())


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
