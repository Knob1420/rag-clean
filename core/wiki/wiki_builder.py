"""
Wiki 构建器

两阶段 Chain-of-Thought wiki 生成：
1. Analysis - 分析源文档
2. Generation - 基于分析生成 wiki 页面

Ported from LLM Wiki (src/lib/ingest.ts)
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

from loguru import logger

from core.wiki.prompts import (
    build_analysis_prompt,
    build_analysis_user_message,
    build_generation_prompt,
    build_generation_user_message,
    PAGE_MERGER_PROMPT,
    slugify,
)
from core.wiki.models import (
    WikiPage,
    EntityPage,
    ConceptPage,
    SourcePage,
    ReviewItem,
    ReviewType,
    ReviewPriority,
    LogEntry,
    WikiConfig,
    AnalysisResult,
)


# ══════════════════════════════════════════════════════════════════════════════
# LLM 调用（需要与 rag-clean 的 LLM client 集成）
# ══════════════════════════════════════════════════════════════════════════════


def call_llm(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
    stream: bool = False,
) -> str:
    """
    调用 LLM（需要与 rag-clean 的 LLM client 集成）

    这里需要根据 rag-clean 的实际 LLM 调用方式实现
    目前假设使用 OpenAI 兼容格式

    Args:
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        model: 模型名称
        temperature: 采样温度
        max_tokens: 最大 token 数
        stream: 是否流式返回

    Returns:
        LLM 响应文本
    """
    # TODO: 与 rag-clean 的 LLM client 集成
    # 目前使用 config.py 中的设置
    try:
        from config import settings

        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.deepseek_api_key}",
        }

        payload = {
            "model": model or settings.deepseek_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        response = requests.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,  # 5分钟 timeout
        )

        if stream:
            # 流式处理（简化版）
            result = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            result.append(content)
                    except:
                        pass
            return "".join(result)
        else:
            result = response.json()
            return result["choices"][0]["message"]["content"]

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


def stream_llm(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.1,
    max_tokens: int = 8192,
    callback=None,
) -> str:
    """
    流式调用 LLM，支持增量输出

    Args:
        messages: [{"role": ..., "content": ...}]
        model: 模型名称
        temperature: 采样温度
        max_tokens: 最大 token 数
        callback: 每收到一个 chunk 时的回调函数

    Returns:
        完整的 LLM 响应文本
    """
    try:
        from config import settings

        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.deepseek_api_key}",
        }

        payload = {
            "model": model or settings.deepseek_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        response = requests.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=300,  # 5分钟 timeout
        )

        result = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        result.append(content)
                        if callback:
                            callback(content)
                except:
                    pass

        return "".join(result)

    except Exception as e:
        logger.error(f"LLM stream failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# 解析 LLM 输出（FILE/REVIEW blocks）
# ══════════════════════════════════════════════════════════════════════════════


def parse_file_blocks(output: str) -> List[Tuple[str, str]]:
    """
    解析 LLM 输出中的 FILE blocks

    Args:
        output: LLM 原始输出

    Returns:
        List of (file_path, content) tuples
    """
    blocks = []
    current_path = None
    current_content = []
    frontmatter_ended = False  # 标记 frontmatter 是否已结束

    lines = output.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("FILE:"):
            # 保存上一个 block
            if current_path:
                blocks.append((current_path, "\n".join(current_content)))

            # 开始新 block
            current_path = stripped[5:].strip()
            current_content = []
            frontmatter_ended = False
            i += 1

        elif stripped == "---":
            if not frontmatter_ended:
                # 第一个 ---: 表示 frontmatter 开始，等待结束
                frontmatter_ended = True  # 标记 frontmatter 已结束，下一行开始就是内容
            else:
                # 第二个 ---: 理论上不应该在内容中出现，但以防万一忽略
                pass
            i += 1

        elif current_path is not None:
            # 收集内容（frontmatter 结束后）
            if frontmatter_ended:
                current_content.append(lines[i])
            i += 1

        else:
            i += 1

    # 保存最后一个 block
    if current_path:
        blocks.append((current_path, "\n".join(current_content)))

    return blocks


def parse_review_blocks(output: str) -> List[ReviewItem]:
    """
    解析 LLM 输出中的 REVIEW blocks

    Args:
        output: LLM 原始输出

    Returns:
        List of ReviewItem
    """
    reviews = []
    current_review = {}

    lines = output.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("REVIEW:"):
            # 保存上一个 review
            if current_review:
                try:
                    reviews.append(ReviewItem(
                        title=current_review.get("title", ""),
                        review_type=ReviewType(current_review.get("type", "Skip")),
                        priority=ReviewPriority(current_review.get("priority", "medium")),
                        reason=current_review.get("reason", ""),
                        suggestions=current_review.get("suggestions", []),
                    ))
                except:
                    pass

            # 开始新 review
            current_review = {"title": line[7:].strip(), "suggestions": []}
            i += 1

        elif line.startswith("Type:"):
            current_review["type"] = line[5:].strip()
            i += 1

        elif line.startswith("Priority:"):
            current_review["priority"] = line[9:].strip()
            i += 1

        elif line.startswith("Reason:"):
            current_review["reason"] = line[8:].strip()
            i += 1

        elif line.startswith("- "):
            if "suggestions" not in current_review:
                current_review["suggestions"] = []
            current_review["suggestions"].append(line[2:].strip())
            i += 1

        else:
            i += 1

    # 保存最后一个 review
    if current_review:
        try:
            reviews.append(ReviewItem(
                title=current_review.get("title", ""),
                review_type=ReviewType(current_review.get("type", "Skip")),
                priority=ReviewPriority(current_review.get("priority", "medium")),
                reason=current_review.get("reason", ""),
                suggestions=current_review.get("suggestions", []),
            ))
        except:
            pass

    return reviews


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════


def filter_out_reviews(content: str) -> str:
    """
    从 FILE block 内容中移除 REVIEW block

    Args:
        content: 包含 FILE block 的内容（可能有 REVIEW block 混入）

    Returns:
        只包含 FILE block 内容（移除 REVIEW block）
    """
    # REVIEW block 从 "REVIEW:" 开始，到下一个 "FILE:" 结束
    # 使用正则来移除 REVIEW block
    import re
    # 匹配 REVIEW: 开头的整个 block，直到下一个 FILE: 或字符串结束
    result = re.sub(r'\n?REVIEW:.*?(?=\nFILE:|\Z)', '', content, flags=re.DOTALL)
    return result


def _append_to_index_or_log(existing_content: str, incoming_content: str, file_path: str, source_file: str) -> str:
    """
    追加内容到 index.md 或 log.md（支持去重）

    Args:
        existing_content: 已存在的内容
        incoming_content: 新内容（来自 LLM 生成的 FILE block）
        file_path: 文件路径（"index.md" 或 "log.md"）
        source_file: 来源文件名（用于日志）

    Returns:
        合并后的内容
    """
    # 提取 incoming 的 body 部分（跳过 frontmatter）
    lines = incoming_content.split("\n")
    body_lines = []
    in_frontmatter = False

    for line in lines:
        if line.strip() == "---":
            if in_frontmatter:
                in_frontmatter = False
                continue
            else:
                in_frontmatter = True
                continue
        if not in_frontmatter:
            body_lines.append(line)

    incoming_body = "\n".join(body_lines).strip()
    if not incoming_body:
        return existing_content

    if file_path == "index.md":
        # index.md: 追加新页面标题（去重）
        # 提取所有 - [[...]] 行（去除标题行）
        new_lines = [l for l in incoming_body.split("\n") if l.strip().startswith("- [[")]
        if not new_lines:
            return existing_content

        # 去重：提取已有条目的 slug
        import re
        existing_slugs = set(re.findall(r'- \[\[([^\]]+)\]\]', existing_content))

        # 只添加不重复的条目
        unique_new_lines = []
        for line in new_lines:
            # 提取 slug（从 [[slug]] 或 [[slug|alias]] 中提取 slug）
            slug_match = re.search(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', line)
            if slug_match:
                slug = slug_match.group(1)
                if slug not in existing_slugs:
                    unique_new_lines.append(line)
                    existing_slugs.add(slug)  # 防止同一批次内重复

        if not unique_new_lines:
            return existing_content
        return existing_content.strip() + "\n" + "\n".join(unique_new_lines)

    elif file_path == "log.md":
        # log.md: 追加新条目（## YYYY-MM-DD\n- [ingest] ...）
        # 只提取 - [ingest] ... 行
        new_lines = [l for l in incoming_body.split("\n") if l.strip().startswith("- [ingest]")]
        if not new_lines:
            return existing_content
        return existing_content.strip() + "\n" + "\n".join(new_lines)

    else:
        return existing_content.strip() + "\n" + incoming_body


# ══════════════════════════════════════════════════════════════════════════════
# Wiki 构建器
# ══════════════════════════════════════════════════════════════════════════════


class WikiBuilder:
    """
    Wiki 构建器 - 管理整个 wiki 生成流程

    用法：
        builder = WikiBuilder(project_dir)
        result = builder.ingest_source(file_path, purpose_md, schema_md)
    """

    def __init__(
        self,
        project_dir: str,
        config: WikiConfig = None,
    ):
        """
        初始化 WikiBuilder

        Args:
            project_dir: 项目根目录（包含 wiki/ 和 raw/ 子目录）
            config: WikiConfig 配置（purpose, schema）
                    如果未提供，会自动从 project_dir/purpose.md 和 schema.md 读取
        """
        self.project_dir = Path(project_dir)
        self.config = config or WikiConfig()

        # 如果 config 中 purpose/schema 为空，尝试从文件读取
        if not self.config.purpose:
            purpose_file = self.project_dir / "purpose.md"
            if purpose_file.exists():
                self.config.purpose = purpose_file.read_text(encoding="utf-8")
                logger.info(f"Loaded purpose.md from {purpose_file}")

        if not self.config.schema:
            schema_file = self.project_dir / "schema.md"
            if schema_file.exists():
                self.config.schema = schema_file.read_text(encoding="utf-8")
                logger.info(f"Loaded schema.md from {schema_file}")

        self.wiki_dir = self.project_dir  # wiki 目录就是项目根目录
        self.raw_dir = self.project_dir / "raw"

        # 确保目录存在
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        (self.wiki_dir / "sources").mkdir(exist_ok=True)
        (self.wiki_dir / "entities").mkdir(exist_ok=True)
        (self.wiki_dir / "concepts").mkdir(exist_ok=True)
        (self.wiki_dir / "queries").mkdir(exist_ok=True)
        (self.wiki_dir / "synthesis").mkdir(exist_ok=True)

    def ingest_source(
        self,
        source_path: str,
        folder_context: str = "",
        index_md: str = "",
        overview_md: str = "",
    ) -> Dict[str, Any]:
        """
        摄入单个源文档，生成 wiki 页面

        Args:
            source_path: 源文件路径（可以是 markdown 或其他支持格式）
            folder_context: 文件夹上下文（用于分类提示）
            index_md: 当前 index.md 内容
            overview_md: 当前 overview.md 内容

        Returns:
            Dict containing:
            - pages: list of (path, content) tuples for generated pages
            - reviews: list of ReviewItem
            - log_entry: LogEntry
        """
        source_file = Path(source_path)
        file_name = source_file.name

        # 读取源文件内容
        content = source_file.read_text(encoding="utf-8")
        if len(content) > 150000:
            logger.warning(f"Source file {file_name} is too large, truncating to 150000 chars")
            content = content[:150000]

        # ═══════════════════════════════════════════════════════════════
        # Step 1: Analysis
        # ═══════════════════════════════════════════════════════════════
        logger.info(f"Step 1: Analyzing {file_name}")

        analysis_prompt = build_analysis_prompt(
            purpose=self.config.purpose,
            index=index_md,
            source_content=content,
            folder_context=folder_context,
            output_lang=self.config.output_lang,
        )

        analysis_user_msg = build_analysis_user_message(
            file_name=file_name,
            folder_context=folder_context,
            content=content,
        )

        messages = [
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": analysis_user_msg},
        ]

        analysis_result = call_llm(messages, temperature=0.1)

        # ═══════════════════════════════════════════════════════════════
        # Step 2: Generation
        # ═══════════════════════════════════════════════════════════════
        logger.info(f"Step 2: Generating wiki pages for {file_name}")

        generation_prompt = build_generation_prompt(
            schema=self.config.schema,
            purpose=self.config.purpose,
            index=index_md,
            source_file_name=file_name,
            overview=overview_md,
            source_content=content,
            output_lang=self.config.output_lang,
        )

        generation_user_msg = build_generation_user_message(
            analysis=analysis_result,
            source_file_name=file_name,
        )

        messages = [
            {"role": "system", "content": generation_prompt},
            {"role": "user", "content": generation_user_msg},
        ]

        generation_output = call_llm(messages, temperature=0.1)

        # ═══════════════════════════════════════════════════════════════
        # Parse output
        # ═══════════════════════════════════════════════════════════════
        file_blocks = parse_file_blocks(generation_output)
        review_items = parse_review_blocks(generation_output)

        # ═══════════════════════════════════════════════════════════════
        # Save reviews to cache
        # ═══════════════════════════════════════════════════════════════
        try:
            from core.wiki.review_cache import save_reviews
            if review_items:
                save_reviews(str(self.project_dir), file_name, review_items)
        except Exception as e:
            logger.warning(f"Failed to save review cache: {e}")

        # ═══════════════════════════════════════════════════════════════
        written_pages = []  # Initialize early for auto-generated pages from reviews

        # Auto-process reviews by type
        # ═══════════════════════════════════════════════════════════════
        if review_items:
            try:
                from core.wiki.review_cache import generate_from_review, remove_review, save_pending_research

                for review in review_items:
                    review_type = review.review_type.value.strip()

                    if review_type.lower() == "create page":
                        # 明确类型：直接生成页面
                        logger.info(f"Auto-generating page from CREATE_PAGE: {review.title}")
                        pages = generate_from_review(
                            str(self.project_dir),
                            review,
                            self.config.purpose,
                            self.config.schema,
                        )
                        for path, content in pages.items():
                            # 去掉 wiki/ 前缀（如果 LLM 输出带了这个前缀）
                            if path.startswith("wiki/"):
                                path = path[4:]
                            full_path = self.wiki_dir / path
                            full_path.parent.mkdir(parents=True, exist_ok=True)
                            full_path.write_text(content, encoding="utf-8")
                            written_pages.append((path, content))
                            logger.info(f"Auto-generated: {path}")
                        remove_review(str(self.project_dir), file_name, review.title)

                    elif review_type.lower() == "deep research":
                        # 模糊类型：保存到待研究列表，不自动生成
                        logger.info(f"Deferred DEEP_RESEARCH: {review.title}")
                        save_pending_research(str(self.project_dir), file_name, review)
                        # 不从缓存移除，保留供后续研究

                    else:
                        # 忽略（包括 "Skip" 和其他）
                        logger.info(f"Skipped review: {review.title}")
                        remove_review(str(self.project_dir), file_name, review.title)

            except Exception as e:
                logger.warning(f"Failed to auto-process reviews: {e}")

        # ═══════════════════════════════════════════════════════════════
        # Write pages (with special handling for index/log/overview)
        # ═══════════════════════════════════════════════════════════════
        written_pages = []

        for path, content in file_blocks:
            full_path = self.wiki_dir / path

            # 特殊文件处理
            if path in ("index.md", "log.md"):
                # index.md: 追加新页面标题（去重）
                # log.md: 追加新条目
                if full_path.exists():
                    existing_content = full_path.read_text(encoding="utf-8")
                    if existing_content.strip():
                        content = _append_to_index_or_log(existing_content, content, path, file_name)
                        logger.info(f"Appended to: {path}")
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")
                written_pages.append((path, content))
                logger.info(f"Written: {path}")

            elif path == "overview.md":
                # overview.md: 直接替换（LLM 生成的已经是覆盖全 wiki 的综合概述）
                # 注意：不需要合并，因为 generation prompt 中 LLM 已经看到了现有的 overview
                # 并被指示生成一个 2-5 段的综合概述来反映整个 wiki 的内容
                # 同时过滤掉 REVIEW block 内容（overview 不应包含 review）
                content = filter_out_reviews(content)
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")
                written_pages.append((path, content))
                logger.info(f"Updated overview.md")

            else:
                # entity/concept/sources 等普通页面：合并
                if full_path.exists():
                    existing_content = full_path.read_text(encoding="utf-8")
                    if existing_content.strip():
                        logger.info(f"Page exists, merging: {path}")
                        try:
                            from core.wiki.page_merger import merge_wiki_pages
                            merged_content = merge_wiki_pages(
                                existing_content=existing_content,
                                incoming_content=content,
                                source_file_name=file_name,
                            )
                            content = merged_content
                        except Exception as e:
                            logger.warning(f"Merge failed for {path}, using new content: {e}")

                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")
                written_pages.append((path, content))
                logger.info(f"Written: {path}")

        # ═══════════════════════════════════════════════════════════════
        # Update index.md
        # ═══════════════════════════════════════════════════════════════
        # 从 file_blocks 中提取新页面的标题，构建 index 更新
        # 注意：这里只是追加，实际的 index 更新应该在外部处理

        # ═══════════════════════════════════════════════════════════════
        # Log entry
        # ═══════════════════════════════════════════════════════════════
        log_entry = LogEntry(
            date=datetime.now().strftime("%Y-%m-%d"),
            operation="ingest",
            title=file_name,
            source_file=file_name,
        )

        return {
            "pages": written_pages,
            "reviews": review_items,
            "log_entry": log_entry,
            "analysis": analysis_result,
        }

    def ingest_batch(
        self,
        source_dir: str,
        recursive: bool = True,
        file_filter: callable = None,
        skip_existing: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量摄入源文档

        Args:
            source_dir: 源文件目录
            recursive: 是否递归子目录
            file_filter: 可选的文件过滤器函数
            skip_existing: 是否跳过已处理的文件（根据 log.md 判断）

        Returns:
            每个源文档的处理结果列表
        """
        source_path = Path(source_dir)
        results = []

        # 获取所有支持的文件
        if recursive:
            files = list(source_path.rglob("*"))
        else:
            files = list(source_path.glob("*"))

        # 过滤只保留文档文件
        supported_exts = {".md", ".txt", ".pdf", ".docx", ".doc", ".pptx", ".ppt"}
        files = [f for f in files if f.suffix.lower() in supported_exts]

        if file_filter:
            files = [f for f in files if file_filter(f)]

        # 获取当前的 index 和 overview
        index_md = ""
        overview_md = ""

        index_file = self.wiki_dir / "index.md"
        if index_file.exists():
            index_md = index_file.read_text(encoding="utf-8")

        overview_file = self.wiki_dir / "overview.md"
        if overview_file.exists():
            overview_md = overview_file.read_text(encoding="utf-8")

        # 获取已处理的文件列表（用于跳过）
        ingested_files: set = set()
        if skip_existing:
            log_file = self.wiki_dir / "log.md"
            if log_file.exists():
                log_content = log_file.read_text(encoding="utf-8")
                # 从 log.md 中提取已 ingest 的文件名
                import re
                ingested_files = set(re.findall(r'- \[ingest\] (.+)', log_content))
            logger.info(f"Found {len(ingested_files)} already-ingested files, will skip them")

        # 逐个处理
        for file in files:
            # 跳过已处理的文件
            if skip_existing and file.name in ingested_files:
                logger.info(f"Skipping already ingested: {file.name}")
                results.append({
                    "file": str(file),
                    "success": True,
                    "skipped": True,
                })
                continue

            try:
                # 计算 folder_context（相对于 source_dir 的路径）
                folder_context = str(file.parent.relative_to(source_path))

                result = self.ingest_source(
                    source_path=str(file),
                    folder_context=folder_context,
                    index_md=index_md,
                    overview_md=overview_md,
                )

                results.append({
                    "file": str(file),
                    "success": True,
                    **result,
                })

                # 更新 index_md 和 overview_md（为下一个文件提供上下文）
                index_file = self.wiki_dir / "index.md"
                if index_file.exists():
                    index_md = index_file.read_text(encoding="utf-8")

                overview_file = self.wiki_dir / "overview.md"
                if overview_file.exists():
                    overview_md = overview_file.read_text(encoding="utf-8")

            except Exception as e:
                logger.error(f"Failed to ingest {file}: {e}")
                results.append({
                    "file": str(file),
                    "success": False,
                    "error": str(e),
                })

        return results


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════════════════════════


def ingest_single_file(
    project_dir: str,
    source_file: str,
    purpose_md: str = "",
    schema_md: str = "",
    output_lang: str = "zh",
) -> Dict[str, Any]:
    """
    便捷函数：摄入单个文件到 wiki

    Args:
        project_dir: 项目根目录
        source_file: 源文件路径
        purpose_md: purpose.md 内容（可选，不提供则从 project_dir/purpose.md 读取）
        schema_md: schema.md 内容（可选，不提供则从 project_dir/schema.md 读取）
        output_lang: 输出语言

    Returns:
        处理结果
    """
    # 如果 purpose/schema 未提供，尝试从项目目录读取
    project_path = Path(project_dir)
    if not purpose_md:
        purpose_file = project_path / "purpose.md"
        if purpose_file.exists():
            purpose_md = purpose_file.read_text(encoding="utf-8")

    if not schema_md:
        schema_file = project_path / "schema.md"
        if schema_file.exists():
            schema_md = schema_file.read_text(encoding="utf-8")

    config = WikiConfig(
        purpose=purpose_md,
        schema=schema_md,
        output_lang=output_lang,
    )

    builder = WikiBuilder(project_dir, config)

    index_md = ""
    overview_md = ""

    index_file = builder.wiki_dir / "index.md"
    if index_file.exists():
        index_md = index_file.read_text(encoding="utf-8")

    overview_file = builder.wiki_dir / "overview.md"
    if overview_file.exists():
        overview_md = overview_file.read_text(encoding="utf-8")

    folder_context = ""
    result = builder.ingest_source(
        source_path=source_file,
        folder_context=folder_context,
        index_md=index_md,
        overview_md=overview_md,
    )

    return result