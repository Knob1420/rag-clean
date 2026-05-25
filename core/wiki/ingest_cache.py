"""
SHA256-based Ingest Cache

存储源文件内容的 hash → 如果内容未变则跳过重新 ingest
缓存文件: .llm-wiki/ingest-cache.json

Ported from LLM Wiki (src/lib/ingest-cache.ts)
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# Cache 数据结构
# ══════════════════════════════════════════════════════════════════════════════


class CacheEntry:
    """缓存条目"""
    def __init__(
        self,
        hash: str,
        timestamp: int,
        files_written: List[str],
    ):
        self.hash = hash
        self.timestamp = timestamp
        self.files_written = files_written

    def to_dict(self) -> dict:
        return {
            "hash": self.hash,
            "timestamp": self.timestamp,
            "filesWritten": self.files_written,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        return cls(
            hash=d["hash"],
            timestamp=d["timestamp"],
            files_written=d.get("filesWritten", []),
        )


CacheData = Dict[str, CacheEntry]


# ══════════════════════════════════════════════════════════════════════════════
# Hash 计算
# ══════════════════════════════════════════════════════════════════════════════


def compute_sha256(content: str) -> str:
    """
    计算字符串内容的 SHA256 hash

    Args:
        content: 要计算 hash 的字符串

    Returns:
        64字符的 hex hash
    """
    encoder = content.encode("utf-8")
    digest = hashlib.sha256(encoder).digest()
    return digest.hex()


def compute_file_sha256(file_path: str) -> str:
    """
    计算文件的 SHA256 hash（分块读取，支持大文件）

    Args:
        file_path: 文件路径

    Returns:
        64字符的 hex hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Cache 路径
# ══════════════════════════════════════════════════════════════════════════════


def cache_path(project_path: str) -> Path:
    """
    获取缓存文件路径

    Args:
        project_path: 项目根目录

    Returns:
        .llm-wiki/ingest-cache.json 路径
    """
    return Path(project_path) / ".llm-wiki" / "ingest-cache.json"


def ensure_cache_dir(project_path: str) -> None:
    """确保 .llm-wiki 目录存在"""
    cache_dir = Path(project_path) / ".llm-wiki"
    cache_dir.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cache 读写
# ══════════════════════════════════════════════════════════════════════════════


def load_cache(project_path: str) -> CacheData:
    """
    从磁盘加载缓存

    Args:
        project_path: 项目根目录

    Returns:
        CacheData dict，key 为源文件名
    """
    cache_file = cache_path(project_path)
    if not cache_file.exists():
        return {}

    try:
        raw = cache_file.read_text(encoding="utf-8")
        data = json.loads(raw)
        entries = {}
        for filename, entry_data in data.get("entries", {}).items():
            entries[filename] = CacheEntry.from_dict(entry_data)
        return entries
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load ingest cache: {e}")
        return {}


def save_cache(project_path: str, entries: CacheData) -> None:
    """
    保存缓存到磁盘

    Args:
        project_path: 项目根目录
        entries: CacheData dict
    """
    ensure_cache_dir(project_path)
    cache_file = cache_path(project_path)

    data = {
        "entries": {
            filename: entry.to_dict()
            for filename, entry in entries.items()
        }
    }

    try:
        cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to save ingest cache: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 核心 API
# ══════════════════════════════════════════════════════════════════════════════


def check_ingest_cache(
    project_path: str,
    source_file_name: str,
    source_content: str,
) -> Optional[List[str]]:
    """
    检查源文件是否已经以相同内容 ingest 过

    重要：只有当所有之前写入的文件都仍然存在于磁盘上时，
    才会返回缓存。否则视为缓存失效，进行完整的重新 ingest。

    Args:
        project_path: 项目根目录
        source_file_name: 源文件名（不含路径）
        source_content: 源文件内容字符串

    Returns:
        如果缓存命中，返回之前写入的文件路径列表
        如果缓存未命中，返回 None
    """
    entries = load_cache(project_path)
    entry = entries.get(source_file_name)

    if not entry:
        return None

    # 检查 hash 是否匹配
    current_hash = compute_sha256(source_content)
    if entry.hash != current_hash:
        return None

    # 检查所有之前写入的文件是否仍然存在
    project = Path(project_path)
    for file_path in entry.files_written:
        full_path = project / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not full_path.exists():
            logger.info(f"[ingest-cache] cache miss for {source_file_name}: {file_path} no longer on disk")
            return None

    return entry.files_written


def save_ingest_cache(
    project_path: str,
    source_file_name: str,
    source_content: str,
    files_written: List[str],
) -> None:
    """
    成功 ingest 后保存结果到缓存

    Args:
        project_path: 项目根目录
        source_file_name: 源文件名
        source_content: 源文件内容
        files_written: 写入的 wiki 页面路径列表
    """
    entries = load_cache(project_path)
    file_hash = compute_sha256(source_content)

    entries[source_file_name] = CacheEntry(
        hash=file_hash,
        timestamp=int(__import__("time").time() * 1000),
        files_written=files_written,
    )

    save_cache(project_path, entries)
    logger.info(f"[ingest-cache] saved cache for {source_file_name}: {len(files_written)} files")


def remove_from_ingest_cache(
    project_path: str,
    source_file_name: str,
) -> None:
    """
    从缓存中移除条目（例如当源文件被删除时）

    Args:
        project_path: 项目根目录
        source_file_name: 源文件名
    """
    entries = load_cache(project_path)

    if source_file_name in entries:
        del entries[source_file_name]
        save_cache(project_path, entries)
        logger.info(f"[ingest-cache] removed {source_file_name} from cache")


def clear_ingest_cache(project_path: str) -> None:
    """
    清空整个 ingest 缓存

    Args:
        project_path: 项目根目录
    """
    cache_file = cache_path(project_path)
    if cache_file.exists():
        cache_file.unlink()
        logger.info(f"[ingest-cache] cleared cache for {project_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 与 WikiBuilder 集成
# ══════════════════════════════════════════════════════════════════════════════


def check_and_ingest(
    project_path: str,
    source_file: str,
    source_content: str,
    ingest_func,
) -> dict:
    """
    快捷函数：先检查缓存，如果未命中则执行 ingest

    Args:
        project_path: 项目根目录
        source_file: 源文件路径
        source_content: 源文件内容字符串
        ingest_func: ingest 函数，签名: (source_file, content) -> dict

    Returns:
        dict with:
        - cached: bool，是否命中缓存
        - result: ingest 结果（如果未命中）
        - pages: list of (path, content) tuples
    """
    from pathlib import Path

    filename = Path(source_file).name

    # 尝试缓存命中
    cached_files = check_ingest_cache(project_path, filename, source_content)
    if cached_files is not None:
        logger.info(f"[ingest-cache] HIT for {filename}")
        return {
            "cached": True,
            "pages": cached_files,
            "result": None,
        }

    # 缓存未命中，执行 ingest
    logger.info(f"[ingest-cache] MISS for {filename}, running ingest")

    result = ingest_func(source_file, source_content)

    # 提取写入的文件路径
    pages = []
    if "pages" in result:
        pages = [p[0] for p in result["pages"]]

    # 保存到缓存
    save_ingest_cache(project_path, filename, source_content, pages)

    return {
        "cached": False,
        "pages": pages,
        "result": result,
    }