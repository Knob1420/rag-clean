#!/usr/bin/env python
"""
MinerU 3.x 服务管理脚本

用法:
    python scripts/start_mineru.py start     # 启动常驻服务（默认）
    python scripts/start_mineru.py stop      # 停止服务
    python scripts/start_mineru.py status    # 查看状态
    python scripts/start_mineru.py restart   # 重启

启动后，extractor._convert_with_mineru3 会自动连服务（~15s/文件），
服务挂了会自动 fallback 到冷启动 CLI（~58s/文件）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.client.mineru_client import (
    is_mineru_service_running,
    start_mineru_service,
    stop_mineru_service,
)


def cmd_start():
    from config import settings
    api_url = start_mineru_service(
        host=settings.mineru3_host,
        port=settings.mineru3_port,
    )
    print(f"\n✓ MinerU 服务运行中: {api_url}")
    print(f"  日志: {Path(settings.cache_dir) / 'mineru_service.log'}")
    print(f"  API 文档: {api_url}/docs")
    print(f"\n停止: python scripts/start_mineru.py stop")


def cmd_stop():
    if stop_mineru_service():
        print("✓ MinerU 服务已停止")
    else:
        print("✗ 无法停止（可能服务是手动起的，或在另一终端用 mineru-api 启动）")


def cmd_status():
    from config import settings
    running = is_mineru_service_running(settings.mineru3_api_url)
    print(f"MinerU 服务 ({settings.mineru3_api_url}): {'✓ 运行中' if running else '✗ 未启动'}")
    if running:
        try:
            r = __import__('httpx').get(f"{settings.mineru3_api_url.rstrip('/')}/", timeout=3)
            print(f"  详情: {r.json()}")
        except Exception as e:
            print(f"  详情查询失败: {e}")


def cmd_restart():
    stop_mineru_service()
    cmd_start()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    actions = {"start": cmd_start, "stop": cmd_stop, "status": cmd_status, "restart": cmd_restart}
    if cmd not in actions:
        print(f"未知命令: {cmd}；可选: {list(actions.keys())}")
        sys.exit(1)
    actions[cmd]()
