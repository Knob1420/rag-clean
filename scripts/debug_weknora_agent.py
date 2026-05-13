"""
WeKnora-port AgentEngine 详细 Debug 脚本

消费 execute_stream 的 StreamEvent（含 debug 事件），逐步骤打印：
- 每轮迭代的 LLM 原始输出 vs StripThinkBlocks 清理后
- LLM thought 内容
- 工具调用名 + 参数预览
- 工具执行结果（截断显示）
- RRF 融合详情（BM25 命中数 / vector 命中数 / 融合后数）
- 上下文窗口管理（consolidation / compression 触发）
- 终止原因 + Token 用量
- 可选保存完整 trace 到 JSON

用法：
    python scripts/debug_weknora_agent.py "NX1智算机的重量是多少"
    python scripts/debug_weknora_agent.py "比较G1和G3" --max-iter 5
    python scripts/debug_weknora_agent.py "智加G1的功耗是多少" --save
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from core.agent.weknora_port.engine import AgentEngine, StreamEvent
from core.agent.weknora_port.prompts import KnowledgeBaseInfo
from core.agent.weknora_port.const import (
    DEFAULT_AGENT_MAX_ITERATIONS,
    DEFAULT_CONTEXT_TOKENS,
)
from core.retrieve.retrieval import RetrievalService
from core.retrieve.retrieval_models import RetrievedChunk, TokenUsage
from core.client.embedder import encode


# ════════════════════════════════════════════════════════════════
# MonkeyPatch RetrievalService._hybrid_search 来捕获 RRF 融合详情
# ════════════════════════════════════════════════════════════════════

_hybrid_debug_info: List[Dict[str, Any]] = []


def _patch_hybrid_search():
    """MonkeyPatch _hybrid_search 来记录 RRF 融合中间数据"""
    original_hybrid_search = RetrievalService._hybrid_search

    def _debug_hybrid_search(self, query, options):
        global _hybrid_debug_info

        query_vector = encode(query)

        candidate_k = options.top_k * 2

        # BM25: 先构建 query_string
        query_string = self._build_bm25_query(query, options)
        bm25_results = self._execute_bm25(query_string, options, candidate_k)
        bm25_count = len(bm25_results)

        # Vector
        vector_results = []
        if query_vector is not None:
            vector_options = options.model_copy(update={"top_k": candidate_k})
            vector_results = self._execute_vector_search(
                query_vector, vector_options, candidate_k
            )
        vector_count = len(vector_results)

        # RRF 融合
        vector_w = options.vector_weight if options.vector_weight is not None else 0.95
        bm25_w = 1.0 - vector_w

        rrf_results = self.rrf.fuse(
            bm25_results=[(c, i) for i, c in enumerate(bm25_results)],
            vector_results=[(c, i) for i, c in enumerate(vector_results)],
            bm25_weight=bm25_w,
            vector_weight=vector_w,
        )

        fused_count = len(rrf_results)

        # 记录 debug 信息
        debug_entry = {
            "query_preview": query[:80],
            "bm25_count": bm25_count,
            "vector_count": vector_count,
            "fused_count": fused_count,
            "bm25_weight": bm25_w,
            "vector_weight": vector_w,
            "use_hyde": options.use_hyde,
            "top_k_requested": options.top_k,
            "bm25_top_ids": [c.chunk_id for c in bm25_results[:3]],
            "vector_top_ids": [c.chunk_id for c in vector_results[:3]],
        }
        _hybrid_debug_info.append(debug_entry)

        # 打印融合详情
        print(f"\n  [RRF_FUSION]")
        print(f"    BM25 结果:   {bm25_count} 条 (候选 {candidate_k})")
        print(f"    Vector 结果: {vector_count} 条 (候选 {candidate_k})")
        print(f"    RRF 融合后:  {fused_count} 条 (权重 bm25={bm25_w:.2f} vector={vector_w:.2f})")
        if bm25_count > 0:
            print(f"    BM25 top3:  {[f'{c.chunk_id[:20]}...({c.score:.2f})' for c in bm25_results[:3]]}")
        if vector_count > 0:
            print(f"    Vector top3: {[f'{c.chunk_id[:20]}...({c.score:.2f})' for c in vector_results[:3]]}")

        final_chunks = []
        for chunk, fused_score in rrf_results:
            chunk.score = fused_score
            final_chunks.append(chunk)

        return final_chunks

    RetrievalService._hybrid_search = _debug_hybrid_search


def _chunk_to_dict(chunk: RetrievedChunk) -> Dict[str, Any]:
    """将 RetrievedChunk 转为可序列化 dict"""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "content": chunk.content,
        "score": chunk.score,
        "doc_title": chunk.doc_title,
        "dataset_id": chunk.dataset_id,
        "chunk_type": chunk.chunk_type,
        "doc_hash": chunk.doc_hash,
        "parent_id": chunk.parent_id,
    }


def _make_serializable(obj: Any) -> Any:
    """递归将对象转为 JSON 可序列化结构"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, RetrievedChunk):
        return _chunk_to_dict(obj)
    if isinstance(obj, TokenUsage):
        return obj.__dict__
    return str(obj)


def _truncate(text: str, max_len: int = 400) -> str:
    """截断长文本，保留首尾"""
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 3
    return text[:half] + " ... " + text[-half:]


def debug_run(
    query: str,
    max_iterations: int = DEFAULT_AGENT_MAX_ITERATIONS,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    save: bool = False,
    output_dir: str = "debug_output",
    debug_log: bool = False,
):
    """消费 execute_stream，逐步骤打印详细信息"""

    global _hybrid_debug_info
    _hybrid_debug_info = []

    if debug_log:
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    # ── 注入 RRF debug hook ──
    _patch_hybrid_search()

    # ── 构建 KB 信息 ──
    kb_info = KnowledgeBaseInfo(
        id="default",
        name="星载智能设备知识库",
        type="document",
        description="卫星/星载设备产品知识库，包含规格参数、技术文档",
        doc_count=100,
        capabilities=["vector", "keyword"],
    )

    # ── 创建引擎 ──
    engine = AgentEngine(
        max_iterations=max_iterations,
        max_context_tokens=max_context_tokens,
        knowledge_bases=[kb_info],
        language="zh-CN",
        session_id=f"debug_{int(time.time())}",
    )

    print("=" * 80)
    print("  WeKnora-port AgentEngine DEBUG")
    print(f"  QUERY: {query}")
    print(f"  MAX_ITERATIONS: {max_iterations}")
    print(f"  MAX_CONTEXT_TOKENS: {max_context_tokens}")
    print("=" * 80)

    # ── 收集 trace ──
    trace_steps: List[Dict[str, Any]] = []
    current_step: Optional[Dict[str, Any]] = None
    answer_parts: List[str] = []
    final_answer = ""
    terminated_reason = ""
    done_data: Dict[str, Any] = {}

    start_time = time.time()

    for event in engine.execute_stream(query):
        etype = event.event_type
        data = event.data

        # ── step_start: 新一轮迭代 ──
        if etype == "step_start":
            iteration = data.get("iteration", "?")
            max_iter = data.get("max_iterations", "?")
            print(f"\n{'─' * 80}")
            print(f"  ITERATION {iteration}/{max_iter}")
            print(f"{'─' * 80}")
            current_step = {
                "iteration": iteration,
                "thought": "",
                "tool_calls": [],
                "observation_previews": [],
                "debug_info": None,
                "rrf_info": None,
                "duration": 0.0,
            }

        # ── debug: LLM 原始响应详情 ──
        elif etype == "debug":
            if current_step is not None:
                current_step["debug_info"] = data

            raw_len = data.get("raw_content_len", 0)
            clean_len = data.get("cleaned_content_len", 0)
            think_stripped = data.get("think_stripped", False)
            tc_count = data.get("tool_calls_count", 0)
            usage = data.get("usage")

            print(f"\n  [LLM_RESPONSE]")
            if usage:
                print(f"    Token 用量: prompt={usage.get('prompt_tokens', '?')}, "
                      f"completion={usage.get('completion_tokens', '?')}, "
                      f"total={usage.get('total_tokens', '?')}")
            print(f"    原始内容长度: {raw_len} chars")
            print(f"    清理后长度:   {clean_len} chars")
            if think_stripped:
                stripped = raw_len - clean_len
                print(f"    [STRIP_THINK] 清除 {stripped} 字符推理痕迹")
                think_content = data.get("think_content", "")
                if think_content:
                    print(f"    推理内容预览: {_truncate(think_content, 300)}")
            cleaned_preview = data.get("cleaned_content_preview", "")
            if cleaned_preview:
                print(f"    清理后内容:   {_truncate(cleaned_preview, 300)}")
            print(f"    工具调用数:   {tc_count}")

        # ── step_end: 一轮迭代结束 ──
        elif etype == "step_end":
            iteration = data.get("iteration", "?")
            action = data.get("action", "?")
            duration = data.get("duration", 0)
            thought = data.get("thought", "")
            tool_calls = data.get("tool_calls", [])
            obs_preview = data.get("observation_preview", "")

            if current_step is not None:
                current_step["duration"] = duration
                current_step["thought"] = thought
                current_step["tool_calls"] = tool_calls
                current_step["observation_previews"] = [obs_preview] if obs_preview else []
                # 关联 RRF debug 信息
                if _hybrid_debug_info:
                    current_step["rrf_info"] = _hybrid_debug_info[-1]
                trace_steps.append(current_step)
                current_step = None

            print(f"\n  [STEP_END] iteration={iteration} action={action} "
                  f"duration={duration:.2f}s")

            if thought:
                print(f"    THOUGHT: {_truncate(thought, 500)}")

            for tc in tool_calls:
                tc_name = tc.get("name", "?")
                tc_args = tc.get("args_preview", "")
                print(f"    TOOL: {tc_name}")
                if tc_args:
                    print(f"      args: {_truncate(tc_args, 300)}")

            if obs_preview:
                print(f"    RESULT: {_truncate(obs_preview, 400)}")

        # ── answer_token: 最终回答的 token ──
        elif etype == "answer_token":
            token = data.get("content", "")
            answer_parts.append(token)

        # ── done: 整个 agent 完成 ──
        elif etype == "done":
            final_answer = "".join(answer_parts)
            terminated_reason = data.get("terminated_reason", "unknown")
            done_data = data
            elapsed = time.time() - start_time

            print(f"\n{'=' * 80}")
            print("  DONE")
            print(f"{'=' * 80}")
            print(f"  终止原因:     {terminated_reason}")
            print(f"  总迭代:       {data.get('iterations', '?')}")
            print(f"  累积 chunks: {data.get('chunks_count', '?')}")
            print(f"  耗时:         {elapsed:.2f}s")

            usage = data.get("usage", {})
            if usage:
                print(f"  Token 用量:   prompt={usage.get('prompt_tokens', '?')}, "
                      f"completion={usage.get('completion_tokens', '?')}, "
                      f"total={usage.get('total_tokens', '?')}")

            # 工具调用统计
            tool_counts: Dict[str, int] = {}
            for step in trace_steps:
                for tc in step.get("tool_calls", []):
                    name = tc.get("name", "unknown")
                    tool_counts[name] = tool_counts.get(name, 0) + 1

            if tool_counts:
                print(f"\n  工具调用统计:")
                for name, count in sorted(tool_counts.items()):
                    print(f"    {name}: {count}")

            # Deep Read 使用情况
            deep_read_count = tool_counts.get("list_knowledge_chunks", 0)
            search_count = tool_counts.get("knowledge_search", 0) + tool_counts.get("grep_chunks", 0)
            print(f"\n  Deep Read 使用: {deep_read_count} 次 (搜索 {search_count} 次)")
            if search_count > 0 and deep_read_count == 0:
                print("  ⚠️  搜索后未执行 Deep Read！")

            # RRF 融合统计
            if _hybrid_debug_info:
                total_bm25 = sum(d["bm25_count"] for d in _hybrid_debug_info)
                total_vector = sum(d["vector_count"] for d in _hybrid_debug_info)
                total_fused = sum(d["fused_count"] for d in _hybrid_debug_info)
                print(f"\n  RRF 融合统计 (共 {len(_hybrid_debug_info)} 次混合检索):")
                print(f"    BM25 结果总计:  {total_bm25} 条")
                print(f"    Vector 结果总计: {total_vector} 条")
                print(f"    RRF 融合后总计: {total_fused} 条")
                for i, info in enumerate(_hybrid_debug_info):
                    print(f"    [{i+1}] bm25={info['bm25_count']} vector={info['vector_count']} "
                          f"→ fused={info['fused_count']} (w_bm25={info['bm25_weight']:.2f} "
                          f"w_vector={info['vector_weight']:.2f})"
                          f"{' [HyDE]' if info['use_hyde'] else ''}")

            # 打印最终回答
            if final_answer:
                print(f"\n{'─' * 80}")
                print(f"  FINAL ANSWER ({len(final_answer)} chars)")
                print(f"{'─' * 80}")
                print(final_answer)

        # ── error: 错误事件 ──
        elif etype == "error":
            error_msg = data.get("error", "unknown error")
            print(f"\n  [ERROR] {error_msg}")

    # ── 打印逐步详情 ──
    print(f"\n{'━' * 80}")
    print("  逐步详情汇总")
    print(f"{'━' * 80}")

    for step in trace_steps:
        iter_num = step["iteration"]
        thought = step["thought"]
        tool_calls = step["tool_calls"]
        duration = step["duration"]
        debug_info = step.get("debug_info")
        rrf_info = step.get("rrf_info")

        print(f"\n  ┌─ ITERATION {iter_num} ({duration:.2f}s) ─")

        if debug_info:
            think_stripped = debug_info.get("think_stripped", False)
            if think_stripped:
                stripped = debug_info.get("raw_content_len", 0) - debug_info.get("cleaned_content_len", 0)
                print(f"  │ [STRIP_THINK] 清除 {stripped} chars")

        if rrf_info:
            print(f"  │ [RRF] bm25={rrf_info['bm25_count']} "
                  f"vector={rrf_info['vector_count']} "
                  f"→ fused={rrf_info['fused_count']} "
                  f"(w={rrf_info['bm25_weight']:.2f}/{rrf_info['vector_weight']:.2f})"
                  f"{' [HyDE]' if rrf_info['use_hyde'] else ''}")

        if thought:
            print(f"  │ THOUGHT: {_truncate(thought, 400)}")

        for tc in tool_calls:
            name = tc.get("name", "?")
            args_preview = tc.get("args_preview", "")
            print(f"  │ TOOL: {name}")
            if args_preview:
                print(f"  │   args: {_truncate(args_preview, 250)}")

        obs_list = step.get("observation_previews", [])
        for obs in obs_list:
            if obs:
                print(f"  │ RESULT: {_truncate(obs, 300)}")

        print(f"  └─")

    # ── 累积 chunks 详情 ──
    if engine._accumulated_chunks:
        print(f"\n  累积 chunks ({len(engine._accumulated_chunks)}):")
        for i, chunk in enumerate(engine._accumulated_chunks[:20]):
            doc_name = chunk.doc_title or chunk.doc_id
            print(f"    [{i+1}] {doc_name} | type={chunk.chunk_type} | "
                  f"score={chunk.score:.4f} | content_len={len(chunk.content)}")
        if len(engine._accumulated_chunks) > 20:
            print(f"    ... 共 {len(engine._accumulated_chunks)} 条，仅显示前 20")

    # ── 保存到 JSON ──
    if save:
        _save_trace(query, trace_steps, final_answer, terminated_reason,
                     done_data, engine, output_dir)

    return {
        "query": query,
        "final_answer": final_answer,
        "terminated_reason": terminated_reason,
        "steps": trace_steps,
        "chunks_count": len(engine._accumulated_chunks),
        "rrf_debug": _hybrid_debug_info,
    }


def _save_trace(
    query: str,
    steps: List[Dict[str, Any]],
    final_answer: str,
    terminated_reason: str,
    done_data: Dict[str, Any],
    engine: AgentEngine,
    output_dir: str,
):
    """保存完整 trace 到 JSON 文件"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query[:20])
    filename = f"weknora_trace_{timestamp}_{safe_query}.json"
    filepath = out_path / filename

    trace = {
        "query": query,
        "agent": "weknora_port",
        "terminated_reason": terminated_reason,
        "final_answer": final_answer,
        "steps": steps,
        "done_data": done_data,
        "rrf_debug": _hybrid_debug_info,
        "accumulated_chunks": [
            _chunk_to_dict(c) for c in engine._accumulated_chunks
        ],
    }

    serializable = _make_serializable(trace)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] 完整 trace 已保存: {filepath}")
    print(f"       文件大小: {filepath.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WeKnora-port AgentEngine Debug")
    parser.add_argument("query", help="查询问题")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_AGENT_MAX_ITERATIONS,
                        help=f"最大迭代数 (default: {DEFAULT_AGENT_MAX_ITERATIONS})")
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS,
                        help=f"上下文窗口大小 (default: {DEFAULT_CONTEXT_TOKENS})")
    parser.add_argument("--save", action="store_true", help="保存中间结果到 JSON")
    parser.add_argument("--output-dir", default="debug_output", help="输出目录")
    parser.add_argument("--debug", action="store_true", help="启用 loguru DEBUG 级别日志")
    args = parser.parse_args()

    debug_run(
        args.query,
        max_iterations=args.max_iter,
        max_context_tokens=args.max_context_tokens,
        save=args.save,
        output_dir=args.output_dir,
        debug_log=args.debug,
    )
