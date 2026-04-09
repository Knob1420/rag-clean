#!/usr/bin/env python
"""
RAG Clean 统一启动脚本

用法:
    python run.py              # 显示启动说明
    python run.py --main       # 仅启动主API
    python run.py --embedding  # 仅启动Embedding服务
    python run.py --rerank     # 仅启动Rerank服务
    python run.py --mineru     # 仅启动MinerU解析服务
    python run.py --frontend   # 启动前端界面
"""
import sys
import argparse
import subprocess
from pathlib import Path


def run_service(service_name: str, module: str, port: int):
    """启动单个服务"""
    print(f"\n{'='*50}")
    print(f"Starting {service_name} (port {port})")
    print(f"{'='*50}\n")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        f"{module}:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    subprocess.run(cmd)


def run_frontend():
    """启动前端界面 (Chainlit)"""
    print(f"\n{'='*50}")
    print(f"Starting Frontend - Chainlit (port 7860)")
    print(f"{'='*50}\n")

    cmd = [
        sys.executable, "-m", "chainlit",
        "run", "ui/chainlit_app.py",
        "--host", "0.0.0.0",
        "--port", "7860",
        "--root-path", "",
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="RAG Clean Service Launcher")
    parser.add_argument("--main", action="store_true", help="Start main API service")
    parser.add_argument("--embedding", action="store_true", help="Start Embedding service")
    parser.add_argument("--rerank", action="store_true", help="Start Rerank service")
    parser.add_argument("--mineru", action="store_true", help="Start MinerU parse service")
    parser.add_argument("--frontend", action="store_true", help="Start frontend UI")
    parser.add_argument("--all", action="store_true", help="Show launch instructions (default)")

    args = parser.parse_args()

    # 添加项目路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # 如果没有指定任何选项，显示启动说明
    if not any([args.main, args.embedding, args.rerank, args.mineru, args.frontend, args.all]):
        args.all = True

    if args.all:
        print("\n" + "=" * 50)
        print("RAG Clean - Service Launcher")
        print("=" * 50)
        print("\nStart each service in a separate terminal:")
        print(f"  1. Main API:     python run.py --main       (port 8000)")
        print(f"  2. Embedding:    python run.py --embedding  (port 8001)")
        print(f"  3. Rerank:       python run.py --rerank     (port 8002)")
        print(f"  4. MinerU:       python run.py --mineru     (port 8003)")
        print(f"  5. Frontend:     python run.py --frontend   (port 7860)")
        return

    if args.main:
        run_service("Main API", "api.main", 8000)
    elif args.embedding:
        run_service("Embedding Service", "api.embedding", 8001)
    elif args.rerank:
        run_service("Rerank Service", "api.rerank", 8002)
    elif args.mineru:
        run_service("MinerU Parse Service", "api.mineru", 8003)
    elif args.frontend:
        run_frontend()


if __name__ == "__main__":
    main()
