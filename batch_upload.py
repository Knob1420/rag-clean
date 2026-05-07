#!/usr/bin/env python3
"""批量上传 md 文件到 RAGFlow 知识库"""

import requests
import os
from pathlib import Path

# 配置
API_KEY = "ragflow-Gs-bps4m8SM4AppX6kvKSgbkvUceO-O-yXH7BOzU92M"
BASE_URL = "http://localhost:8080"
DATASET_ID = "62e152fc470011f19c9f9bf6992a017b"
DOCS_FOLDER = "/home/zjlab/Documents/build_LLMs/NLP_course_hf/RAG/data/raw"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

def upload_file(file_path):
    """上传单个文件"""
    filename = os.path.basename(file_path)
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{BASE_URL}/api/v1/datasets/{DATASET_ID}/documents",
                headers=headers,
                files={'file': (filename, f, 'text/markdown')}
            )
        if response.status_code == 200:
            print(f"✓ 上传成功: {filename}")
            return True
        else:
            print(f"✗ 上传失败: {filename} - {response.status_code} {response.text[:100]}")
            return False
    except Exception as e:
        print(f"✗ 错误: {filename} - {e}")
        return False

def main():
    # 找到所有 .md 文件
    md_files = list(Path(DOCS_FOLDER).rglob("*.md"))
    print(f"找到 {len(md_files)} 个 .md 文件")

    success = 0
    failed = 0

    for i, file_path in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}] ", end="", flush=True)
        if upload_file(file_path):
            success += 1
        else:
            failed += 1

    print(f"\n完成! 成功: {success}, 失败: {failed}")

if __name__ == "__main__":
    main()
