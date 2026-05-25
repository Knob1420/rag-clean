"""
Wiki 数据模型

定义 wiki 页面、Review 项等核心数据结构

Ported from LLM Wiki 的 TypeScript 类型
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class PageType(Enum):
    """Wiki 页面类型"""
    SOURCE = "source"
    ENTITY = "entity"
    CONCEPT = "concept"
    COMPARISON = "comparison"
    QUERY = "query"
    SYNTHESIS = "synthesis"


class ReviewType(Enum):
    """Review 项类型"""
    CREATE_PAGE = "Create Page"
    DEEP_RESEARCH = "Deep Research"
    SKIP = "Skip"


class ReviewPriority(Enum):
    """Review 优先级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ══════════════════════════════════════════════════════════════════════════════
# Wiki 页面模型
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class WikiPage:
    """
    Wiki 页面基础模型

    对应一个 .md 文件，包含 YAML frontmatter 和 Markdown 内容
    """
    type: PageType
    title: str
    content: str
    created: str = ""  # YYYY-MM-DD
    updated: str = ""  # YYYY-MM-DD
    tags: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)  # bare slugs, not [[...]]
    sources: List[str] = field(default_factory=list)
    description: str = ""
    wikilinks: List[str] = field(default_factory=list)  # [[Page Title]] in body

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().strftime("%Y-%m-%d")
        if not self.updated:
            self.updated = self.created

    def to_markdown(self) -> str:
        """
        生成完整的 Markdown 文件内容（包含 frontmatter）
        """
        lines = [
            "---",
            f"type: {self.type.value}",
            f'title: "{self.title}"' if ":" in self.title else f"title: {self.title}",
            f"created: {self.created}",
            f"updated: {self.updated}",
            f"tags: [{', '.join(self.tags)}]" if self.tags else "tags: []",
            f'related: [{", ".join(self.related)}]' if self.related else "related: []",
            f'sources: [{", ".join(self.sources)}]' if self.sources else "sources: []",
        ]
        if self.description:
            lines.append(f'description: "{self.description}"')
        lines.append("---")
        lines.append("")
        lines.append(self.content)
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, path: str, content: str) -> "WikiPage":
        """
        从 Markdown 文件内容解析 WikiPage

        Args:
            path: 文件路径（用于提取 type 和 title）
            content: 文件完整内容

        Returns:
            WikiPage 实例
        """
        import re
        from pathlib import Path

        frontmatter = {}
        body_start = content.find("---")
        body_end = content.find("---", body_start + 3)

        if body_start != -1 and body_end != -1:
            fm_text = content[body_start + 3:body_end].strip()
            for line in fm_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    value = value.strip().strip('"').strip("'")
                    if key == "tags" or key == "related" or key == "sources":
                        # Parse YAML array: [a, b, c]
                        value = re.findall(r'[\w-]+', value)
                    frontmatter[key] = value

        body = content[body_end + 3:].strip()
        filename = Path(path).stem
        page_type = PageType(frontmatter.get("type", "source"))

        return cls(
            type=page_type,
            title=frontmatter.get("title", filename),
            content=body,
            created=frontmatter.get("created", ""),
            updated=frontmatter.get("updated", ""),
            tags=frontmatter.get("tags", []),
            related=frontmatter.get("related", []),
            sources=frontmatter.get("sources", []),
            description=frontmatter.get("description", ""),
        )


@dataclass
class EntityPage(WikiPage):
    """实体页面 - 人物、组织、产品等"""

    def __init__(self, title: str, content: str = "", **kwargs):
        super().__init__(type=PageType.ENTITY, title=title, content=content, **kwargs)


@dataclass
class ConceptPage(WikiPage):
    """概念页面 - 理论、方法、技术等"""

    def __init__(self, title: str, content: str = "", **kwargs):
        super().__init__(type=PageType.CONCEPT, title=title, content=content, **kwargs)


@dataclass
class SourcePage(WikiPage):
    """源文档摘要页面"""

    def __init__(self, title: str, content: str = "", **kwargs):
        super().__init__(type=PageType.SOURCE, title=title, content=content, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Review 模型
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ReviewItem:
    """
    Review 项 - LLM 标记需要人工判断的内容

    对应 LLM Wiki 的 Review 功能
    """
    title: str
    review_type: ReviewType
    priority: ReviewPriority
    reason: str
    suggestions: List[str] = field(default_factory=list)
    source_file: str = ""
    created: str = ""
    dismissed: bool = False

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().strftime("%Y-%m-%d")

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        type_str = self.review_type.value
        priority_str = self.priority.value
        suggestions = "\n".join(f"- {s}" for s in self.suggestions) if self.suggestions else ""

        lines = [
            f"## {self.title}",
            "",
            f"**Type:** {type_str}",
            f"**Priority:** {priority_str}",
            f"**Source:** {self.source_file}" if self.source_file else "",
            f"**Created:** {self.created}",
            "",
            f"**Reason:** {self.reason}",
            "",
        ]
        if suggestions:
            lines.append("**Suggestions:**")
            lines.append(suggestions)
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "title": self.title,
            "type": self.review_type.value,
            "priority": self.priority.value,
            "reason": self.reason,
            "suggestions": self.suggestions,
            "source_file": self.source_file,
            "created": self.created,
            "dismissed": self.dismissed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewItem":
        """从字典创建 ReviewItem"""
        return cls(
            title=d["title"],
            review_type=ReviewType(d.get("type", "Skip")),
            priority=ReviewPriority(d.get("priority", "medium")),
            reason=d.get("reason", ""),
            suggestions=d.get("suggestions", []),
            source_file=d.get("source_file", ""),
            created=d.get("created", ""),
            dismissed=d.get("dismissed", False),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Log 条目
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class LogEntry:
    """
    Log 条目 - 记录 wiki 操作历史
    """
    date: str  # YYYY-MM-DD
    operation: str  # ingest / update / delete
    title: str
    source_file: str = ""
    details: str = ""

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        return f"## [{self.date}] {self.operation} | {self.title}"

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "date": self.date,
            "operation": self.operation,
            "title": self.title,
            "source_file": self.source_file,
            "details": self.details,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Wiki 配置
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class WikiConfig:
    """
    Wiki 配置 - 对应 purpose.md 和 schema.md
    """
    purpose: str = ""
    schema: str = ""
    output_lang: str = "zh"  # zh or en

    @classmethod
    def from_files(cls, purpose_path: str, schema_path: str) -> "WikiConfig":
        """从文件加载配置"""
        from pathlib import Path

        purpose = ""
        schema = ""

        if Path(purpose_path).exists():
            purpose = Path(purpose_path).read_text(encoding="utf-8")

        if Path(schema_path).exists():
            schema = Path(schema_path).read_text(encoding="utf-8")

        return cls(purpose=purpose, schema=schema)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "purpose": self.purpose,
            "schema": self.schema,
            "output_lang": self.output_lang,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 结果（Step 1 输出）
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class AnalysisResult:
    """
    Step 1 分析结果 - 包含从源文档提取的所有信息
    """
    source_file: str
    key_entities: List[Dict[str, str]] = field(default_factory=list)
    key_concepts: List[Dict[str, str]] = field(default_factory=list)
    main_arguments: str = ""
    connections: str = ""
    contradictions: str = ""
    recommendations: str = ""
    existing_pages: List[str] = field(default_factory=list)  # 已存在于 wiki 的页面
    new_pages_needed: List[str] = field(default_factory=list)  # 建议创建的新页面

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "source_file": self.source_file,
            "key_entities": self.key_entities,
            "key_concepts": self.key_concepts,
            "main_arguments": self.main_arguments,
            "connections": self.connections,
            "contradictions": self.contradictions,
            "recommendations": self.recommendations,
            "existing_pages": self.existing_pages,
            "new_pages_needed": self.new_pages_needed,
        }

    def to_text(self) -> str:
        """
        转换为文本格式（用于传递给 Step 2）
        """
        lines = [f"# Analysis of {self.source_file}", ""]

        if self.key_entities:
            lines.append("## Key Entities")
            for entity in self.key_entities:
                lines.append(f"- **{entity.get('name', '')}** ({entity.get('type', '')})")
                lines.append(f"  Role: {entity.get('role', '')}")
                if entity.get('existing', ''):
                    lines.append(f"  Wiki: {entity.get('existing', 'Not found')}")
            lines.append("")

        if self.key_concepts:
            lines.append("## Key Concepts")
            for concept in self.key_concepts:
                lines.append(f"- **{concept.get('name', '')}**: {concept.get('definition', '')}")
                if concept.get('existing', ''):
                    lines.append(f"  Wiki: {concept.get('existing', 'Not found')}")
            lines.append("")

        if self.main_arguments:
            lines.append("## Main Arguments & Findings")
            lines.append(self.main_arguments)
            lines.append("")

        if self.connections:
            lines.append("## Connections to Existing Wiki")
            lines.append(self.connections)
            lines.append("")

        if self.contradictions:
            lines.append("## Contradictions & Tensions")
            lines.append(self.contradictions)
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            lines.append(self.recommendations)
            lines.append("")

        return "\n".join(lines)