"""
批量导入 Markdown 文件到 FastGPT（通过 text API）

用法:
    python scripts/fastgpt_batch_import.py --dry-run
    python scripts/fastgpt_batch_import.py --limit 10
"""

import sys
import argparse
import time
import re
from pathlib import Path
from typing import Optional, List
from collections import defaultdict

import requests
from loguru import logger

# FastGPT 配置
FASTGPT_BASE_URL = "http://10.107.207.88:3000"
API_KEY = "fastgpt-lmYVgX2bTSkMeZjS4RVYfP9Sd3gnyZi2PSe5u7u2Kvyw9tYHLB4zAdls"
DATASET_ID = "69f588c798b97b5df2339af7"

# 文件目录
MARKDOWN_DIR = Path("/home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/data/raw")

# 分块参数
CHUNK_SIZE = 1000  # 原文长度大于 1000 时分块
OVERLAP = 100  # 重叠字符数


def get_headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}"
    }


def get_collection_list() -> list:
    """获取所有 collection"""
    url = f"{FASTGPT_BASE_URL}/api/core/dataset/collection/list"
    payload = {"datasetId": DATASET_ID, "pageNum": 1, "pageSize": 100}
    try:
        resp = requests.post(url, json=payload, headers=get_headers(), timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", {}).get("data", [])
    except Exception as e:
        logger.warning(f"获取集合列表失败: {e}")
    return []


def get_or_create_collection(name: str) -> Optional[str]:
    """获取或创建单个统一集合，用于存放所有文件"""
    collections = get_collection_list()
    for coll in collections:
        coll_name = coll.get("name", "")
        # 去掉 .txt 后缀后匹配
        if coll_name.removesuffix(".txt") == name:
            return coll.get("_id")

    # 不按文件夹区分，只创建一个统一集合
    url = f"{FASTGPT_BASE_URL}/api/core/dataset/collection/create/text"
    payload = {
        "datasetId": DATASET_ID,
        "name": name,
        "text": f"[{name}] 知识库集合"
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == 0 or data.get("success"):
                return data.get("data", {}).get("collectionId")
            else:
                logger.warning(f"创建集合响应异常: {data}")
    except Exception as e:
        logger.warning(f"创建集合失败: {e}")
    return None


def insert_text_chunks(collection_id: str, chunks: List[dict]) -> dict:
    """通过 insertData API 插入文本块"""
    url = f"{FASTGPT_BASE_URL}/api/core/dataset/data/insertData"
    results = {"ok": 0, "failed": 0}

    for chunk in chunks:
        payload = {
            "collectionId": collection_id,
            "q": chunk["q"],
            "a": chunk["a"]
        }
        try:
            resp = requests.post(url, json=payload, headers=get_headers(), timeout=30)
            if resp.status_code == 200:
                results["ok"] += 1
            else:
                results["failed"] += 1
        except Exception:
            results["failed"] += 1

    return results


def extract_title(content: str) -> str:
    """从 markdown 内容中提取标题"""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    for line in content.split("\n"):
        line = line.strip()
        if line:
            return line[:100]
    return "无标题"


def chunk_markdown(content: str, title: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[dict]:
    """将 markdown 内容分块"""
    if len(content) <= chunk_size:
        return [{"q": title, "a": content.strip()}] if content.strip() else []

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]

        if end < len(content):
            last_newline = chunk_text.rfind("\n")
            last_period = chunk_text.rfind("。")
            last_punctuation = max(last_newline, last_period)

            if last_punctuation > chunk_size * 0.7:
                chunk_text = chunk_text[:last_punctuation + 1]
                end = start + last_punctuation + 1

        chunk_text = chunk_text.strip()
        if chunk_text:
            chunk_title = f"{title} (片段{chunk_index + 1})" if chunk_index > 0 else title
            chunks.append({
                "q": chunk_title,
                "a": chunk_text
            })
            chunk_index += 1

        start += chunk_size - overlap

    return chunks


def process_markdown_file(file_path: Path) -> dict:
    """处理单个 markdown 文件"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"status": "failed", "error": f"读取失败: {e}"}

    if not content.strip():
        return {"status": "skipped", "reason": "空文件"}

    title = extract_title(content)
    chunks = chunk_markdown(content, title)

    return {
        "status": "ok",
        "title": title,
        "chunks": chunks
    }


def main():
    parser = argparse.ArgumentParser(description="批量导入 Markdown 到 FastGPT")
    parser.add_argument("--dry-run", action="store_true", help="只列出文件，不执行导入")
    parser.add_argument("--limit", type=int, default=0, help="限制文件数量（0=不限）")
    parser.add_argument("--delay", type=float, default=0.2, help="请求间隔（秒）")
    args = parser.parse_args()

    # 收集所有 markdown 文件
    md_files = sorted(MARKDOWN_DIR.rglob("*.md"))
    if args.limit > 0:
        md_files = md_files[:args.limit]

    if not md_files:
        print("未找到 markdown 文件")
        return

    print(f"\n{'='*70}")
    print(f"  FastGPT 批量导入")
    print(f"  知识库 ID: {DATASET_ID}")
    print(f"  源目录: {MARKDOWN_DIR}")
    print(f"  文件数量: {len(md_files)}")
    print(f"  分块大小: {CHUNK_SIZE} 字符，重叠: {OVERLAP} 字符")
    print(f"{'='*70}\n")

    if args.dry_run:
        for i, f in enumerate(md_files):
            rel = f.relative_to(MARKDOWN_DIR)
            print(f"  [{i+1:4d}] {rel}")
        print(f"\n  共 {len(md_files)} 个文件 (--dry-run)")
        return

    # 不按父目录分组，所有文件放入统一集合
    total_success = 0
    total_failed = 0
    total_chunks = 0

    collection_name = "知识库文档"
    collection_id = get_or_create_collection(collection_name)
    if not collection_id:
        print("创建/获取统一集合失败，退出")
        return
    print(f"  使用集合: {collection_name} (ID: {collection_id[:16]}...)\n")

    for i, file_path in enumerate(md_files):
        rel = file_path.relative_to(MARKDOWN_DIR)
        print(f"[{i+1}/{len(md_files)}] {rel.name}", end=" ... ")

        result = process_markdown_file(file_path)

        if result["status"] == "skipped":
            print(f"跳过: {result.get('reason', '')}")
            continue
        elif result["status"] == "failed":
            print(f"失败: {result.get('error', 'unknown')}")
            total_failed += 1
            continue

        chunks = result["chunks"]
        if not chunks:
            print("跳过: 分块结果为空")
            continue

        # 插入数据
        insert_result = insert_text_chunks(collection_id, chunks)
        inserted = insert_result["ok"]
        failed = insert_result["failed"]

        if inserted > 0:
            print(f"OK: {inserted} chunks, 失败 {failed}")
            total_success += 1
            total_chunks += inserted
        else:
            print(f"失败: 插入 0 chunks")
            total_failed += 1

        time.sleep(args.delay)

    print(f"\n{'='*70}")
    print(f"  完成")
    print(f"  成功文件: {total_success}, 失败文件: {total_failed}, 总分块: {total_chunks}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
