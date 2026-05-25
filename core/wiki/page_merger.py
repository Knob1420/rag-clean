"""
Page Merger - 合并同一页面的两个版本

当多个源文档产生相同的实体/概念页面时，需要合并为一个统一的页面

Ported from LLM Wiki (src/lib/ingest.ts) - buildPageMerger
"""

from typing import Tuple, List, Dict

from core.wiki.prompts import PAGE_MERGER_PROMPT


def merge_wiki_pages(
    existing_content: str,
    incoming_content: str,
    source_file_name: str,
    llm_call_func=None,
) -> str:
    """
    合并两个版本的 wiki 页面

    Args:
        existing_content: 已存在于磁盘的版本
        incoming_content: 新从另一个源文档生成的版本
        source_file_name: 来源文件名（用于日志）
        llm_call_func: LLM 调用函数，如果为 None 则使用默认实现

    Returns:
        合并后的页面内容
    """
    if not existing_content.strip():
        return incoming_content

    if not incoming_content.strip():
        return existing_content

    if llm_call_func is None:
        llm_call_func = _default_llm_call

    # 分离 frontmatter 和内容
    existing_fm, existing_body = _split_frontmatter(existing_content)
    incoming_fm, incoming_body = _split_frontmatter(incoming_content)

    # 构建合并 prompt
    system_prompt = PAGE_MERGER_PROMPT

    user_message = f"""Merge the following two versions of the same wiki page into one coherent document.

=== EXISTING VERSION (from wiki) ===
{existing_content}

=== INCOMING VERSION (from {source_file_name}) ===
{incoming_content}

Merge now. Output only the merged FILE block."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    result = llm_call_func(messages)

    # 验证合并结果是否有效
    if not result or len(result.strip()) < len(existing_content) * 0.5:
        # 如果合并结果太短（可能是解析失败），使用简单合并策略
        return _simple_merge(existing_content, incoming_content)

    return result


def _split_frontmatter(content: str) -> Tuple[str, str]:
    """
    分离 YAML frontmatter 和主体内容

    Args:
        content: 完整 markdown 内容

    Returns:
        (frontmatter, body) tuple
    """
    parts = content.split("---")
    if len(parts) >= 3:
        # 格式: --- frontmatter --- body
        return parts[1], "---".join(parts[2:])
    else:
        return "", content


def _simple_merge(existing: str, incoming: str) -> str:
    """
    简单合并策略（当 LLM 合并失败时使用）

    策略：保留两个版本的所有唯一内容，去重但不智能合并

    Args:
        existing: 已存在版本
        incoming: 新版本

    Returns:
        合并后内容
    """
    # 简单策略：追加 incoming 的内容到 existing
    existing_fm, existing_body = _split_frontmatter(existing)
    incoming_fm, incoming_body = _split_frontmatter(incoming)

    # 保留 existing 的 frontmatter，追加 incoming 的 sources
    existing_lines = [l for l in existing_fm.split("\n") if l.strip()]
    incoming_lines = [l for l in incoming_fm.split("\n") if l.strip()]

    # 合并 sources 字段
    existing_sources = []
    incoming_sources = []

    for line in existing_lines:
        if line.startswith("sources:"):
            existing_sources = _parse_yaml_array(line)

    for line in incoming_lines:
        if line.startswith("sources:"):
            incoming_sources = _parse_yaml_array(line)

    # 更新 existing sources
    new_sources = list(set(existing_sources + incoming_sources))
    if new_sources:
        # 找到 sources 行并替换
        new_lines = []
        for line in existing_lines:
            if line.startswith("sources:"):
                new_lines.append(f"sources: [{', '.join(new_sources)}]")
            else:
                new_lines.append(line)
        existing_fm = "\n".join(new_lines)

    # 合并 body（简单策略：保留两个）
    merged_body = existing_body.strip() + "\n\n---\n\n" + incoming_body.strip()

    return existing_fm + "\n---\n" + merged_body


def _parse_yaml_array(line: str) -> List[str]:
    """解析 YAML 数组行"""
    import re
    content = line.split(":", 1)[1].strip()
    if content.startswith("[") and content.endswith("]"):
        content = content[1:-1]
    return [s.strip() for s in content.split(",") if s.strip()]


def _default_llm_call(messages: List[Dict[str, str]]) -> str:
    """
    默认 LLM 调用实现

    当没有提供 llm_call_func 时使用
    """
    try:
        from config import settings

        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.deepseek_api_key}",
        }

        payload = {
            "model": settings.deepseek_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 8192,
        }

        response = requests.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        from loguru import logger
        logger.error(f"LLM call failed: {e}")
        raise


def merge_pages_batch(
    pages_to_merge: List[Tuple[str, str]],
    llm_call_func=None,
) -> Dict[str, str]:
    """
    批量合并多个页面

    Args:
        pages_to_merge: List of (existing_content, incoming_content) tuples
        llm_call_func: LLM 调用函数

    Returns:
        Dict mapping source_file to merged_content
    """
    results = {}

    for idx, (existing, incoming) in enumerate(pages_to_merge):
        try:
            merged = merge_wiki_pages(existing, incoming, f"source_{idx}", llm_call_func)
            results[f"page_{idx}"] = merged
        except Exception as e:
            from loguru import logger
            logger.warning(f"Failed to merge page, using simple merge: {e}")
            results[f"page_{idx}"] = _simple_merge(existing, incoming)

    return results