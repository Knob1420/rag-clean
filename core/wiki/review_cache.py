"""
Review Cache - 保存和管理 Review 结果

存储源文件对应的 review items，支持后续根据 review 生成建议的页面
缓存文件: .llm-wiki/review-cache.json

Ported from LLM Wiki review system
"""

import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from loguru import logger

from core.wiki.models import ReviewItem, ReviewType, ReviewPriority


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════


class ReviewEntry:
    """Review 条目"""
    def __init__(
        self,
        source_file: str,
        timestamp: int,
        reviews: List[ReviewItem],
    ):
        self.source_file = source_file
        self.timestamp = timestamp
        self.reviews = reviews

    def to_dict(self) -> dict:
        return {
            "sourceFile": self.source_file,
            "timestamp": self.timestamp,
            "reviews": [r.to_dict() for r in self.reviews],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewEntry":
        return cls(
            source_file=d["sourceFile"],
            timestamp=d["timestamp"],
            reviews=[ReviewItem.from_dict(r) for r in d.get("reviews", [])],
        )


# ══════════════════════════════════════════════════════════════════════════════
# 路径
# ══════════════════════════════════════════════════════════════════════════════


def review_cache_path(project_path: str) -> Path:
    """获取 review 缓存文件路径"""
    return Path(project_path) / ".llm-wiki" / "review-cache.json"


def ensure_cache_dir(project_path: str) -> None:
    """确保 .llm-wiki 目录存在"""
    cache_dir = Path(project_path) / ".llm-wiki"
    cache_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 读写
# ══════════════════════════════════════════════════════════════════════════════


def load_review_cache(project_path: str) -> Dict[str, ReviewEntry]:
    """加载 review 缓存"""
    cache_file = review_cache_path(project_path)
    if not cache_file.exists():
        return {}

    try:
        raw = cache_file.read_text(encoding="utf-8")
        data = json.loads(raw)
        entries = {}
        for filename, entry_data in data.get("entries", {}).items():
            entries[filename] = ReviewEntry.from_dict(entry_data)
        return entries
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load review cache: {e}")
        return {}


def save_review_cache_data(project_path: str, entries: Dict[str, ReviewEntry]) -> None:
    """保存 review 缓存到磁盘"""
    ensure_cache_dir(project_path)
    cache_file = review_cache_path(project_path)

    data = {
        "entries": {
            filename: entry.to_dict()
            for filename, entry in entries.items()
        }
    }

    try:
        cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to save review cache: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 核心 API
# ══════════════════════════════════════════════════════════════════════════════


def save_reviews(
    project_path: str,
    source_file_name: str,
    reviews: List[ReviewItem],
) -> None:
    """
    保存源文件的 review 结果

    Args:
        project_path: 项目根目录
        source_file_name: 源文件名
        reviews: ReviewItem 列表
    """
    entries = load_review_cache(project_path)

    entries[source_file_name] = ReviewEntry(
        source_file=source_file_name,
        timestamp=int(datetime.now().timestamp() * 1000),
        reviews=reviews,
    )

    save_review_cache_data(project_path, entries)
    logger.info(f"[review-cache] saved {len(reviews)} reviews for {source_file_name}")


def get_reviews(project_path: str, source_file_name: str) -> List[ReviewItem]:
    """
    获取源文件的 review 结果

    Args:
        project_path: 项目根目录
        source_file_name: 源文件名

    Returns:
        ReviewItem 列表，如果没有则返回空列表
    """
    entries = load_review_cache(project_path)
    entry = entries.get(source_file_name)
    return entry.reviews if entry else []


def get_pending_reviews(
    project_path: str,
    review_type: Optional[ReviewType] = None,
) -> List[Dict]:
    """
    获取所有待处理的 review

    Args:
        project_path: 项目根目录
        review_type: 可选，按类型过滤

    Returns:
        [{source_file, review_item}, ...] 列表
    """
    entries = load_review_cache(project_path)
    pending = []

    for filename, entry in entries.items():
        for review in entry.reviews:
            if review_type is None or review.review_type == review_type:
                pending.append({
                    "source_file": filename,
                    "review": review,
                })

    return pending


def get_create_page_reviews(project_path: str) -> List[Dict]:
    """获取所有建议创建页面的 review (DEEP_RESEARCH 类型)"""
    return get_pending_reviews(project_path, ReviewType.CREATE_PAGE)


def get_deep_research_reviews(project_path: str) -> List[Dict]:
    """获取所有建议深度研究的 review"""
    return get_pending_reviews(project_path, ReviewType.DEEP_RESEARCH)


def remove_review(project_path: str, source_file_name: str, review_title: str) -> None:
    """
    从缓存中移除一个 review（处理完成后）

    Args:
        project_path: 项目根目录
        source_file_name: 源文件名
        review_title: review 标题
    """
    entries = load_review_cache(project_path)
    entry = entries.get(source_file_name)

    if entry:
        original_count = len(entry.reviews)
        entry.reviews = [r for r in entry.reviews if r.title != review_title]

        if len(entry.reviews) < original_count:
            save_review_cache_data(project_path, entries)
            logger.info(f"[review-cache] removed review '{review_title}' from {source_file_name}")


def clear_review_cache(project_path: str) -> None:
    """清空所有 review 缓存"""
    cache_file = review_cache_path(project_path)
    if cache_file.exists():
        cache_file.unlink()
        logger.info(f"[review-cache] cleared all reviews for {project_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 基于 review 生成页面
# ══════════════════════════════════════════════════════════════════════════════


def generate_from_review(
    project_path: str,
    review_item: ReviewItem,
    purpose: str,
    schema: str,
) -> Dict[str, str]:
    """
    根据 review item 生成建议的页面

    这个函数会在用户确认 review 后调用，生成建议创建的新页面内容

    Args:
        project_path: 项目根目录
        review_item: ReviewItem，包含 title, reason, suggestions
        purpose: purpose.md 内容（用于 LLM）
        schema: schema.md 内容（用于 LLM）

    Returns:
        {page_path: content} 字典
    """
    from core.wiki.wiki_builder import call_llm
    from core.wiki.prompts import slugify

    # 构建 prompt 让 LLM 生成页面内容
    prompt = f"""根据以下 review 建议，生成一个新的 wiki 页面。

## Review 信息
- 标题: {review_item.title}
- 类型: {review_item.review_type.value}
- 优先级: {review_item.priority.value}
- 原因: {review_item.reason}

## 建议内容
"""

    for i, suggestion in enumerate(review_item.suggestions, 1):
        prompt += f"{i}. {suggestion}\n"

    prompt += f"""
## Purpose (项目目标)
{purpose}

## Schema (页面规范)
{schema}

## 任务
生成一个符合 schema 规范的新 wiki 页面。使用 FILE: 格式输出：

**重要：路径不要加 "wiki/" 前缀**

FILE: {('concepts' if review_item.review_type == ReviewType.CREATE_PAGE else 'entities')}/{slugify(review_item.title)}.md
---
type: concept  # 或根据实际情况
title: {review_item.title}
tags: []
related: []
created: {datetime.now().strftime('%Y-%m-%d')}
updated: {datetime.now().strftime('%Y-%m-%d')}

# {review_item.title}

<!-- 页面内容根据 reason 和 suggestions 生成 -->
"""

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"请为 '{review_item.title}' 创建一个 wiki 页面。"},
    ]

    try:
        output = call_llm(messages, temperature=0.1)

        # 解析 FILE blocks
        from core.wiki.wiki_builder import parse_file_blocks
        pages = parse_file_blocks(output)

        # 返回 {path: content} 字典
        return {path: content for path, content in pages}

    except Exception as e:
        logger.error(f"Failed to generate page from review: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# 待研究 Review（DEEP_RESEARCH 类型）
# ══════════════════════════════════════════════════════════════════════════════


def save_pending_research(project_path: str, source_file_name: str, review: ReviewItem) -> None:
    """
    保存待研究的 review 到单独的文件（DEEP_RESEARCH 类型）

    Args:
        project_path: 项目根目录
        source_file_name: 源文件名
        review: ReviewItem
    """
    from pathlib import Path

    research_dir = Path(project_path) / ".llm-wiki" / "pending-research"
    research_dir.mkdir(parents=True, exist_ok=True)

    # 用 review 标题作为文件名（slugify）
    from core.wiki.prompts import slugify
    safe_name = slugify(review.title)
    research_file = research_dir / f"{safe_name}.json"

    # 保存 review 信息
    research_data = {
        "source_file": source_file_name,
        "title": review.title,
        "reason": review.reason,
        "suggestions": review.suggestions,
        "priority": review.priority.value if review.priority else "medium",
        "created_at": datetime.now().isoformat(),
    }

    research_file.write_text(json.dumps(research_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[pending-research] saved: {research_file.name}")


def list_pending_research(project_path: str) -> List[Dict]:
    """
    列出所有待研究的 review

    Args:
        project_path: 项目根目录

    Returns:
        [{title, source_file, reason, suggestions, priority, file_path}, ...]
    """
    from pathlib import Path

    research_dir = Path(project_path) / ".llm-wiki" / "pending-research"
    if not research_dir.exists():
        return []

    pending = []
    for f in research_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["file_path"] = str(f)
            pending.append(data)
        except Exception:
            continue

    return pending


def remove_pending_research(project_path: str, title: str) -> None:
    """删除一个待研究的 review"""
    from pathlib import Path
    from core.wiki.prompts import slugify

    safe_name = slugify(title)
    research_dir = Path(project_path) / ".llm-wiki" / "pending-research"
    research_file = research_dir / f"{safe_name}.json"

    if research_file.exists():
        research_file.unlink()
        logger.info(f"[pending-research] removed: {research_file.name}")