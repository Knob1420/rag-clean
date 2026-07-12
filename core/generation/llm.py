"""
LLM 调用 — 从 generation.py 精简

只保留数据层需要的：
- _call_deepseek() — 基础 LLM 调用
- _parse_json_response() — JSON 解析
- extract_doc_info() — 供 analyzer 调用
- extract_chunk_info_batch() — 供 pipeline 调用
- call_with_tools() — ReAct Agent 专用（带 tool calling + 重试）
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, Generator, List, Optional
from loguru import logger
from openai import OpenAI

from config import settings
from core.retrieve.retrieval_models import TokenUsage
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
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """懒初始化 OpenAI client（复用）"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=300.0,  # 5 min — agent 末轮上下文很长，TTFT 可能超过 120s
            )
        return self._client

    def call(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 2000, model: Optional[str] = None) -> str:
        """
        调用 DeepSeek API，返回文本内容。

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            model: 可选模型名称，默认使用 self.model

        Returns:
            LLM 回复文本
        """
        content, _ = self.call_with_usage(messages, temperature, max_tokens, model)
        return content

    def call_with_usage(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 2000, model: Optional[str] = None) -> tuple:
        """
        调用 DeepSeek API，返回 (content, TokenUsage) 元组。

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            model: 可选模型名称，默认使用 self.model

        Returns:
            (content: str, usage: TokenUsage) 元组
        """
        if not self.api_key:
            logger.warning("使用 Mock 模式（未配置 API Key）")
            return (
                "抱歉，当前使用 Mock 模式。请配置 API Key 以获得真实回复。",
                TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        logger.info(
            f"LLM 调用成功: "
            f"prompt_tokens={usage.prompt_tokens}, "
            f"completion_tokens={usage.completion_tokens}"
        )
        return content, usage

    def call_stream(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 2000, model: Optional[str] = None):
        """
        简单流式调用（不带 tools），逐 token 返回。
        注意：DeepSeek 推理 token（reasoning_content）会被跳过，
        只返回正式的 content 内容。
        """
        stream = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            # DeepSeek 推理 token 会在 content 之前到达，这里跳过
            # （如需转发推理内容，应由调用方自行处理 streaming chunk）
            if delta.content:
                yield delta.content

    async def async_call_stream(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 2000, model: Optional[str] = None):
        """
        异步流式调用（不带 tools），逐 token 返回。
        使用 AsyncOpenAI，不阻塞事件循环。
        """
        from openai import AsyncOpenAI

        if not self.api_key:
            yield "抱歉，当前使用 Mock 模式。请配置 API Key 以获得真实回复。"
            return

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        stream = await client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def call_with_tools_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_retries: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """
        流式调用 LLM（带 tool calling），支持瞬态错误重试。

        yield 事件片段，return 最终结果字典（包含 message + usage）。

        注意：流式调用时 response.usage 通常为 None，需通过完整 content 估算。
        """
        if not self.api_key:
            result = {
                "message": {
                    "role": "assistant",
                    "content": "请配置 API Key 以使用 ReAct Agent。",
                    "tool_calls": None,
                },
                "usage": None,
            }
            yield {"type": "result", "data": result}
            return result

        for attempt in range(max_retries + 1):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )

                # 累积变量
                content_parts: List[str] = []
                tool_calls_acc: Dict[int, Dict[str, Any]] = {}  # index -> partial tc

                for chunk in stream:
                    delta = chunk.choices[0].delta

                    # DeepSeek 推理内容（reasoning_content）→ 作为 thought_token 转发
                    reasoning = getattr(delta, "reasoning_content", None) or ""
                    if reasoning:
                        yield {"type": "reasoning", "delta": reasoning}

                    # 内容 delta
                    if delta.content:
                        content_parts.append(delta.content)
                        # yield 内容片段（可用于前端实时显示）
                        yield {"type": "content", "delta": delta.content}

                    # tool_call delta
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_acc[idx]["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments
                                    yield {
                                        "type": "tool_arg",
                                        "index": idx,
                                        "arguments_delta": tc_delta.function.arguments,
                                        "tool_name": tool_calls_acc[idx]["function"]["name"],
                                    }

                # 构造完整 message
                msg_dict: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(content_parts),
                }
                if tool_calls_acc:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for tc in tool_calls_acc.values()
                    ]

                # 等待 usage（部分 API 可能在 stream 结束后才返回 usage）
                # 注意：流式调用 response.usage 通常为 None，需通过 complete 事件获取
                # 这里返回 None，后续由调用方从完整 response 中统计
                return {"message": msg_dict, "usage": None}

            except Exception as e:
                error_str = str(e)
                is_transient = any(
                    marker in error_str.lower()
                    for marker in ["429", "500", "502", "503", "timeout", "rate_limit", "overloaded"]
                )

                if is_transient and attempt < max_retries:
                    wait_time = attempt + 1
                    logger.warning(
                        f"[LLM] 流式调用瞬态错误 (attempt {attempt+1}/{max_retries+1}): "
                        f"{error_str[:100]}, {wait_time}s 后重试"
                    )
                    time.sleep(wait_time)
                    continue

                logger.error(f"[LLM] 流式 tool calling 调用失败: {e}")
                return None

        return None

    def generate_summary(self, content: str) -> Dict[str, Any]:
        """
        为文档片段生成 summary。

        Args:
            content: 文档内容

        Returns:
            {"summary": str}
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
            }
        except Exception as e:
            logger.warning(f"Summary 生成失败: {e}，使用默认值")
            return {
                "summary": content[:100] + "...",
            }

    async def _call_async(self, messages: List[Dict[str, str]], max_tokens: int = 2000) -> str:
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
                    "max_tokens": max_tokens,
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
                        }
                    except Exception as e:
                        logger.warning(f"Summary 批量生成失败 [{idx}]: {e}，使用默认值")
                        return idx, {
                            "summary": content[:100] + "...",
                        }

            tasks = [_call_one(i, c) for i, c in enumerate(contents)]
            results = await asyncio.gather(*tasks)
            # 按原始顺序返回
            return [r for _, r in sorted(results, key=lambda x: x[0])]

        return asyncio.run(_run())


# ── JSON 解析 ──────────────────────────────────────────


def parse_json_response(response: str) -> Dict[str, Any]:
    """解析 LLM 返回的 JSON"""
    # 如果返回了 HTML（API 错误/认证失败），直接抛出
    if isinstance(response, str) and response.startswith("<"):
        raise ValueError(
            f"API 返回了 HTML 而非 JSON（可能是认证失败）: {response[:200]}"
        )
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
