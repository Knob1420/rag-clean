"""
Wiki 模块 - LLM Wiki 核心流程的 Python 实现

两阶段 Chain-of-Thought wiki 生成：
1. Analysis - 分析源文档，提取实体、概念
2. Generation - 基于分析生成 wiki 页面

Ported from LLM Wiki (TypeScript/React) to Python

主要组件：
- prompts.py: 两阶段 prompt 模板（Analysis + Generation）
- models.py: Wiki 页面、Review、Log 等数据模型
- wiki_builder.py: Wiki 构建器，核心生成逻辑
- page_merger.py: 合并同一页面的两个版本
- templates.py: 预定义模板（General/Business/Research/Aerospace）
- ingest_cache.py: SHA256 增量缓存，避免重复 ingest
- review_cache.py: Review 结果缓存，支持后续生成建议页面

用法：
    # 使用预定义模板快速创建项目
    from core.wiki import create_project_structure, get_template

    # 创建航天型号研制项目
    create_project_structure("/path/to/project", "aerospace")

    # 或手动配置
    from core.wiki import WikiBuilder, WikiConfig, get_template

    template = get_template("aerospace")
    config = WikiConfig(
        purpose=template.purpose,
        schema=template.schema,
        output_lang="zh"
    )

    builder = WikiBuilder("/path/to/project", config)
    result = builder.ingest_source("path/to/source_file.md")
"""

from core.wiki.prompts import (
    build_analysis_prompt,
    build_analysis_user_message,
    build_generation_prompt,
    build_generation_user_message,
    language_rule,
    slugify,
    PAGE_MERGER_PROMPT,
    LINT_PROMPT,
    DEEP_RESEARCH_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPT,
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
    PageType,
)

from core.wiki.wiki_builder import (
    WikiBuilder,
    ingest_single_file,
    call_llm,
    stream_llm,
    parse_file_blocks,
    parse_review_blocks,
)

from core.wiki.page_merger import (
    merge_wiki_pages,
    merge_pages_batch,
)

from core.wiki.ingest_cache import (
    check_ingest_cache,
    save_ingest_cache,
    remove_from_ingest_cache,
    clear_ingest_cache,
    compute_sha256,
    compute_file_sha256,
    check_and_ingest,
)

from core.wiki.templates import (
    WikiTemplate,
    templates,
    get_template,
    list_templates,
    create_project_structure,
    generalTemplate,
    businessTemplate,
    researchTemplate,
    aerospaceTemplate,
)

from core.wiki.review_cache import (
    save_reviews,
    get_reviews,
    get_pending_reviews,
    get_create_page_reviews,
    get_deep_research_reviews,
    remove_review,
    clear_review_cache,
    generate_from_review,
    save_pending_research,
    list_pending_research,
    remove_pending_research,
)

__all__ = [
    # Prompts
    "build_analysis_prompt",
    "build_analysis_user_message",
    "build_generation_prompt",
    "build_generation_user_message",
    "language_rule",
    "slugify",
    "PAGE_MERGER_PROMPT",
    "LINT_PROMPT",
    "DEEP_RESEARCH_SYSTEM_PROMPT",
    "CHAT_SYSTEM_PROMPT",
    # Models
    "WikiPage",
    "EntityPage",
    "ConceptPage",
    "SourcePage",
    "ReviewItem",
    "ReviewType",
    "ReviewPriority",
    "LogEntry",
    "WikiConfig",
    "AnalysisResult",
    "PageType",
    # Builder
    "WikiBuilder",
    "ingest_single_file",
    "call_llm",
    "stream_llm",
    "parse_file_blocks",
    "parse_review_blocks",
    # Merger
    "merge_wiki_pages",
    "merge_pages_batch",
    # Cache
    "check_ingest_cache",
    "save_ingest_cache",
    "remove_from_ingest_cache",
    "clear_ingest_cache",
    "compute_sha256",
    "compute_file_sha256",
    "check_and_ingest",
    # Templates
    "WikiTemplate",
    "templates",
    "get_template",
    "list_templates",
    "create_project_structure",
    "generalTemplate",
    "businessTemplate",
    "researchTemplate",
    "aerospaceTemplate",
    # Review Cache
    "save_reviews",
    "get_reviews",
    "get_pending_reviews",
    "get_create_page_reviews",
    "get_deep_research_reviews",
    "remove_review",
    "clear_review_cache",
    "generate_from_review",
    "save_pending_research",
    "list_pending_research",
    "remove_pending_research",
]