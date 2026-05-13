"""
Compare Agents — Side-by-side evaluation of rag-clean agent vs WeKnora port

Runs the same query through both agents and compares:
- Answer quality
- Steps / iterations
- Tool calls made
- Timing
- Terminated reason
- Chunk retrieval patterns
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def run_rag_clean_agent(query: str, max_iterations: int = 10) -> Dict[str, Any]:
    """Run the existing rag-clean ReActAgent."""
    from core.agent.react_agent import ReActAgent

    agent = ReActAgent(max_iterations=max_iterations)
    result = agent.run(query)

    return {
        "answer": result.answer,
        "steps": [
            {
                "iteration": s.iteration,
                "action": s.action,
                "observation_preview": s.observation[:200],
                "duration": round(s.duration, 2),
            }
            for s in result.steps
        ],
        "total_iterations": result.total_iterations,
        "chunks_count": len(result.chunks),
        "timing": result.timing,
        "usage": result.usage.__dict__ if result.usage else None,
        "terminated_reason": result.terminated_reason,
        "tool_calls": [s.action for s in result.steps if s.action],
    }


def run_weknora_port_agent(query: str, max_iterations: int = 20) -> Dict[str, Any]:
    """Run the WeKnora-port AgentEngine."""
    from core.agent.weknora_port.engine import AgentEngine
    from core.agent.weknora_port.prompts import KnowledgeBaseInfo

    # Build KB info (placeholder — uses default ES index)
    kb_info = KnowledgeBaseInfo(
        id="default",
        name="星载智能设备知识库",
        type="document",
        description="卫星/星载设备产品知识库",
        capabilities=["vector", "keyword"],
    )

    engine = AgentEngine(
        max_iterations=max_iterations,
        knowledge_bases=[kb_info],
        language="Chinese (Simplified)",
    )
    result = engine.execute(query)

    return {
        "answer": result.answer,
        "steps": [
            {
                "iteration": s.iteration,
                "action": s.action,
                "observation_preview": s.observation[:200],
                "duration": round(s.duration, 2),
            }
            for s in result.steps
        ],
        "total_iterations": result.total_iterations,
        "chunks_count": len(result.chunks),
        "timing": result.timing,
        "usage": result.usage.__dict__ if result.usage else None,
        "terminated_reason": result.terminated_reason,
        "tool_calls": [s.action for s in result.steps if s.action],
    }


def compare_agents(query: str, max_iterations: int = 10) -> Dict[str, Any]:
    """
    Run both agents on the same query and compare results.
    """
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")

    # Run rag-clean agent
    print("\n--- Running rag-clean agent ---")
    t0 = time.time()
    rag_clean_result = run_rag_clean_agent(query, max_iterations)
    rag_clean_time = time.time() - t0
    print(f"  Completed in {rag_clean_time:.2f}s")

    # Run WeKnora port agent
    print("\n--- Running WeKnora-port agent ---")
    t0 = time.time()
    weknora_result = run_weknora_port_agent(query, max_iterations=max_iterations + 10)
    weknora_time = time.time() - t0
    print(f"  Completed in {weknora_time:.2f}s")

    # Comparison
    comparison = {
        "query": query,
        "rag_clean": {
            **rag_clean_result,
            "wall_time": round(rag_clean_time, 2),
        },
        "weknora_port": {
            **weknora_result,
            "wall_time": round(weknora_time, 2),
        },
        "differences": {
            "iteration_count": (
                rag_clean_result["total_iterations"],
                weknora_result["total_iterations"],
            ),
            "chunks_retrieved": (
                rag_clean_result["chunks_count"],
                weknora_result["chunks_count"],
            ),
            "terminated_reason": (
                rag_clean_result["terminated_reason"],
                weknora_result["terminated_reason"],
            ),
            "tool_call_count": (
                len(rag_clean_result["tool_calls"]),
                len(weknora_result["tool_calls"]),
            ),
            "has_deep_read": (
                False,  # rag-clean has no separate Deep Read step
                any("list_knowledge_chunks" in tc for tc in weknora_result["tool_calls"]),
            ),
            "has_xml_output": (
                False,  # rag-clean uses plain text
                True,   # WeKnora port uses XML tool output
            ),
            "has_context_mgmt": (
                False,  # rag-clean has no context management
                True,   # WeKnora port has consolidation + compression
            ),
            "max_tool_output": (
                8000,   # rag-clean
                16000,  # WeKnora port
            ),
        },
    }

    # Print summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  Iterations:       rag-clean={rag_clean_result['total_iterations']}, "
          f"weknora={weknora_result['total_iterations']}")
    print(f"  Chunks retrieved: rag-clean={rag_clean_result['chunks_count']}, "
          f"weknora={weknora_result['chunks_count']}")
    print(f"  Terminated:       rag-clean={rag_clean_result['terminated_reason']}, "
          f"weknora={weknora_result['terminated_reason']}")
    print(f"  Tool calls:       rag-clean={len(rag_clean_result['tool_calls'])}, "
          f"weknora={len(weknora_result['tool_calls'])}")
    print(f"  Deep Read used:   rag-clean=No, weknora={comparison['differences']['has_deep_read'][1]}")
    print(f"  Wall time:        rag-clean={rag_clean_time:.2f}s, weknora={weknora_time:.2f}s")

    print(f"\n--- rag-clean answer ---")
    print(rag_clean_result["answer"][:500])
    print(f"\n--- WeKnora-port answer ---")
    print(weknora_result["answer"][:500])

    return comparison


# ── Test queries ────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "星载计算模块有哪些型号？",
    "G1星载计算机的功耗是多少？",
    "卫星在阴影期如何供电？",
    "比较SKC-1和SKC-2的性能差异",
    "蓄电池的充放电次数限制是多少？",
]


def main():
    """Run comparison on test queries."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare rag-clean agent vs WeKnora port")
    parser.add_argument(
        "--query", "-q", type=str, help="Single query to test"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all test queries"
    )
    parser.add_argument(
        "--max-iter", type=int, default=10, help="Max iterations for rag-clean (default: 10)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="results/agent_comparison.json",
        help="Output file for comparison results"
    )
    args = parser.parse_args()

    if args.query:
        results = [compare_agents(args.query, args.max_iter)]
    elif args.all:
        results = []
        for q in TEST_QUERIES:
            try:
                result = compare_agents(q, args.max_iter)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed on query '{q}': {e}")
                results.append({"query": q, "error": str(e)})
    else:
        # Default: run first test query
        results = [compare_agents(TEST_QUERIES[0], args.max_iter)]

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
