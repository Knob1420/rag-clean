"""
MinerU 3.x 服务客户端

提供两种使用方式：
1. 常驻服务（推荐）：先 start_mineru_service()，后续 convert_with_mineru() 走 HTTP 复用模型
2. 冷启动（默认 fallback）：每次调用都新起一个临时服务，模型重载（慢 4×）

设计要点：
- 服务通过 `conda run` 在独立 env（mineru3_env）跑，与主项目隔离 vllm/torch 依赖
- 自动起的服务用 start_new_session=True detach，父进程退出不影响
- PID 文件记录自动起的进程，stop_mineru_service() 只停自己起的（不动手动起的）
- 服务挂了，convert_with_mineru 自动 fallback 到冷启动 CLI
"""

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, List

import httpx
from loguru import logger

from config import settings

# PID 文件（追踪自动起的服务进程）
_PID_FILE = Path(settings.cache_dir) / "mineru_service.pid"
# 服务日志（自动起时 stdout/stderr 重定向到这）
_LOG_FILE = Path(settings.cache_dir) / "mineru_service.log"


# ============================================================
# 健康检查
# ============================================================


def is_mineru_service_running(api_url: Optional[str] = None) -> bool:
    """检查 mineru-api 服务是否在跑"""
    url = api_url or settings.mineru3_api_url
    if not url:
        return False
    try:
        r = httpx.get(f"{url.rstrip('/')}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ============================================================
# 服务生命周期
# ============================================================


def start_mineru_service(
    host: Optional[str] = None,
    port: Optional[int] = None,
    wait_timeout: int = 180,
    preload_vlm: bool = True,
) -> str:
    """
    后台启动 mineru-api 服务（detach 子进程）。

    Args:
        host: 监听地址，默认用 settings.mineru3_host
        port: 端口，默认用 settings.mineru3_port
        wait_timeout: 等待服务 ready 的秒数（VLM 预加载可能要 1-2 分钟）
        preload_vlm: 启动时是否预加载 VLM 模型

    Returns:
        api_url

    Raises:
        RuntimeError: 启动超时或失败
    """
    host = host or settings.mineru3_host
    port = port or settings.mineru3_port
    api_url = f"http://{host}:{port}"

    # 已在跑，直接返回
    if is_mineru_service_running(api_url):
        logger.info(f"[MinerU] 服务已在运行: {api_url}")
        return api_url

    cmd = [
        "conda", "run", "--no-capture-output", "-n", settings.mineru3_env,
        "env",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"CUDA_VISIBLE_DEVICES={settings.mineru3_gpu}",
        "mineru-api",
        "--host", host,
        "--port", str(port),
        "--enable-vlm-preload", "true" if preload_vlm else "false",
    ]

    # detach: start_new_session=True 让子进程脱离父进程的 session
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(_LOG_FILE, "a", encoding="utf-8")
    log_fp.write(f"\n{'=' * 60}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting mineru-api\n{'=' * 60}\n")
    log_fp.flush()

    logger.info(f"[MinerU] 启动服务 (detach): {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # 关键：detach
    )

    # 记录 PID（用 pgid，因为 conda run 会产生多层子进程）
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(proc.pid))

    # 等待 ready
    logger.info(f"[MinerU] 等待服务 ready（最多 {wait_timeout}s）...")
    start = time.time()
    while time.time() - start < wait_timeout:
        if is_mineru_service_running(api_url):
            elapsed = time.time() - start
            logger.success(f"[MinerU] 服务 ready: {api_url} ({elapsed:.1f}s, pid={proc.pid})")
            return api_url
        # 进程提前死了
        if proc.poll() is not None:
            tail = _tail_log(20)
            raise RuntimeError(f"mineru-api 进程退出 (rc={proc.returncode})\n日志末尾:\n{tail}")
        time.sleep(2)

    tail = _tail_log(30)
    raise RuntimeError(f"mineru-api 启动超时（{wait_timeout}s）\n日志末尾:\n{tail}")


def stop_mineru_service() -> bool:
    """
    停止服务。只停本模块 start_mineru_service() 启动的（基于 PID 文件）。

    Returns:
        True=已停止；False=无法停止（PID 文件不存在或进程已死）
    """
    if not _PID_FILE.exists():
        logger.warning("[MinerU] 无 PID 文件，跳过停止（服务可能是手动起的）")
        return False

    try:
        pid = int(_PID_FILE.read_text().strip())
    except (ValueError, OSError) as e:
        logger.error(f"[MinerU] PID 文件损坏: {e}")
        _PID_FILE.unlink(missing_ok=True)
        return False

    # kill 整个进程组（conda run 会产生 mineru-api / vllm 多层子进程）
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        logger.info(f"[MinerU] 已发送 SIGTERM 到进程组 (pgid={pid})")
    except ProcessLookupError:
        logger.info(f"[MinerU] 进程已不存在 (pid={pid})")
        _PID_FILE.unlink(missing_ok=True)
        return False
    except Exception as e:
        logger.error(f"[MinerU] SIGTERM 失败: {e}")
        return False

    # 等待退出
    for _ in range(15):
        try:
            os.kill(pid, 0)  # 检查进程是否还活着
        except ProcessLookupError:
            break
        time.sleep(1)

    # 如果还没死，SIGKILL
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
        logger.warning(f"[MinerU] SIGKILL 强制终止 (pgid={pid})")
    except Exception:
        pass

    _PID_FILE.unlink(missing_ok=True)
    return True


def _tail_log(n_lines: int = 20) -> str:
    """读日志末尾 n 行（错误诊断用）"""
    try:
        lines = _LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return "(无法读取日志)"


# ============================================================
# 转换（核心 API）
# ============================================================


def convert_with_mineru(
    path: Path,
    output_dir: Path,
    prefer_service: bool = True,
) -> Path:
    """
    用 mineru CLI 转换文件。

    优先连常驻服务（避免模型重载）；服务不可用时 fallback 到冷启动。

    Args:
        path: 输入文件（pdf/docx/pptx/xlsx）
        output_dir: 输出目录（mineru 会在里面建 {stem}/{hybrid_auto|office|pipeline}/ 结构）
        prefer_service: True=优先用常驻服务；False=强制冷启动

    Returns:
        生成的 .md 文件路径

    Raises:
        RuntimeError: mineru 失败或没生成 markdown
    """
    cmd: List[str] = [
        "conda", "run", "--no-capture-output", "-n", settings.mineru3_env,
        "env",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"CUDA_VISIBLE_DEVICES={settings.mineru3_gpu}",
        "mineru",
        "-p", str(path),
        "-o", str(output_dir),
        "-b", settings.mineru3_backend,
        "--effort", settings.mineru3_effort,
        "-l", settings.mineru3_lang,
    ]

    # 决定走服务还是冷启动
    use_service = (
        prefer_service
        and settings.mineru3_api_url
        and is_mineru_service_running(settings.mineru3_api_url)
    )
    if use_service:
        cmd += ["--api-url", settings.mineru3_api_url]
        logger.info(f"[MinerU] 走常驻服务: {settings.mineru3_api_url}")
    else:
        logger.info(f"[MinerU] 走冷启动（每次重载模型，~45s）")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=900,  # 15 分钟上限
    )

    if result.returncode != 0:
        tail = result.stderr[-800:] if result.stderr else "(无 stderr)"
        raise RuntimeError(
            f"mineru 失败 rc={result.returncode}: {path.name}\nstderr 末尾:\n{tail}"
        )

    # rglob 兜底找 .md（不同 backend 子目录名不同：hybrid_auto/office/pipeline）
    md_files = list(Path(output_dir).rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"mineru 未生成 markdown: {path.name}")

    return md_files[0]
