"""
WeKnora Faithful Port — Tool Registry

Ported from WeKnora internal/agent/tools/registry.go

Central registry for agent tools: registration, validation, execution,
and OpenAI function definition generation.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from core.agent.weknora_port.const import DEFAULT_MAX_TOOL_OUTPUT
from core.agent.weknora_port.tools.validation import validate_params, cast_params
from core.agent.weknora_port.tools.capabilities import ToolRequirement

# Error hint appended to tool error messages (WeKnora pattern)
TOOL_ERROR_HINT = "\n\n[Analyze the error above and try a different approach.]"


class ToolDefinition:
    """Defines a single tool with its schema, handler, and requirements."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], str],
        requirements: Optional[List[ToolRequirement]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.requirements = requirements or []

    def to_function_definition(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    Manages registration, retrieval, and execution of agent tools.

    Features ported from WeKnora:
    - First-wins registration policy (prevents tool hijacking)
    - JSON Schema validation + type casting before execution
    - Max tool output truncation
    - Error hint appending
    """

    def __init__(self, max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT):
        self._tools: Dict[str, ToolDefinition] = {}
        self._max_tool_output = max_tool_output if max_tool_output > 0 else DEFAULT_MAX_TOOL_OUTPUT

    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Register a tool. First-wins policy: rejects duplicate names
        to prevent tool execution hijacking.
        """
        if tool.name in self._tools:
            logger.warning(
                f"[ToolRegistry] Duplicate tool registration rejected: "
                f"{tool.name} (first-wins policy)"
            )
            return
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """
        Execute a tool with validation, casting, truncation, and error handling.

        Steps:
        1. Validate params against schema
        2. Cast param types
        3. Execute handler
        4. Truncate output if needed
        5. Append error hint on failure
        """
        tool = self._tools.get(name)
        if tool is None:
            error_msg = f"Unknown tool: {name}"
            return error_msg + TOOL_ERROR_HINT

        # Validate parameters
        validation_error = validate_params(args, tool.parameters)
        if validation_error:
            error_msg = f"Parameter validation failed for {name}: {validation_error}"
            return error_msg + TOOL_ERROR_HINT

        # Cast parameter types
        args = cast_params(args, tool.parameters)

        # Execute
        try:
            result = tool.handler(args)
        except Exception as e:
            error_msg = f"Tool execution error [{name}]: {str(e)}"
            logger.error(f"[ToolRegistry] {error_msg}")
            return error_msg + TOOL_ERROR_HINT

        # Truncate output
        if len(result) > self._max_tool_output:
            head_len = int(self._max_tool_output * 0.8)
            tail_len = self._max_tool_output - head_len - 60
            result = (
                result[:head_len]
                + f"\n\n... [TRUNCATED: {len(result)} chars, showing {self._max_tool_output}] ...\n\n"
                + result[-tail_len:]
            )

        return result

    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions in OpenAI function calling format."""
        return [tool.to_function_definition() for tool in self._tools.values()]

    def cleanup(self) -> None:
        """Clean up any tool resources."""
        self._tools.clear()

    @property
    def tool_names(self) -> List[str]:
        """List of registered tool names."""
        return list(self._tools.keys())
