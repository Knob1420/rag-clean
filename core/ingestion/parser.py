"""
MinerU PDF 解析器 — 从 processors.py 精简

仅负责将 PDF 转换为 Markdown，不执行分块。
"""

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# 添加 MinerU 路径（可选）
MINERU_PATH = Path("/home/zjlab/Documents/build_LLMs/NLP_course_hf/MinerU")
if str(MINERU_PATH) not in sys.path:
    sys.path.insert(0, str(MINERU_PATH))

try:
    from mineru.cli.client import do_parse

    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False


class MinerUPDFProcessor:
    """MinerU PDF 处理器 — 仅 PDF → Markdown"""

    def __init__(
        self,
        backend: str = "hybrid-auto-engine",
        model_source: str = "local",
        gpu_device: Optional[int] = None,
        parse_method: str = "auto",
    ):
        if not MINERU_AVAILABLE:
            raise RuntimeError("MinerU 不可用，请先安装 MinerU")

        self.backend = backend
        self.model_source = model_source
        self.gpu_device = gpu_device
        self.parse_method = parse_method

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        将 PDF 转换为 Markdown 并返回内容。

        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录（默认为 PDF 同目录下的 output/）

        Returns:
            Markdown 文本内容
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        if output_dir is None:
            output_dir = str(pdf_path.parent / "output")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.gpu_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_device)

        os.environ["MINERU_MODEL_SOURCE"] = self.model_source
        logger.info(f"开始处理 PDF: {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        pdf_file_name = pdf_path.stem

        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[pdf_file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend=self.backend,
            parse_method=self.parse_method,
            formula_enable=True,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_make_md_mode="mm_markdown",
            start_page_id=0,
            end_page_id=None,
        )

        # 查找生成的 Markdown 文件
        backend_dir = self.backend.replace("-", "_").replace("_engine", "")
        md_file = output_dir / pdf_file_name / backend_dir / f"{pdf_file_name}.md"

        if not md_file.exists():
            md_files = list(output_dir.rglob("*.md"))
            if md_files:
                md_file = md_files[0]
            else:
                raise RuntimeError("未找到生成的 Markdown 文件")

        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()

        logger.info(f"PDF→MD 完成: {len(md_content)} 字符")
        return md_content
