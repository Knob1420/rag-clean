"""
WeKnora Faithful Port — Tool Capabilities & Requirements

Ported from WeKnora internal/agent/tools/capabilities.go

Defines KBCapability enum, ToolRequirement filters (AnyOf/AllOf/ConsumesFiles),
and helper functions for determining which tools are available based on KB capabilities.
"""

from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set


class KBCapability(str, Enum):
    """Knowledge base capability types."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    FAQ = "faq"


class RequirementType(str, Enum):
    """Tool requirement type."""
    ANY_OF = "any_of"
    ALL_OF = "all_of"
    CONSUMES_FILES = "consumes_files"


class ToolRequirement:
    """
    Declares what KB capabilities a tool needs.
    - AnyOf: requires at least one of the listed capabilities
    - AllOf: requires all listed capabilities
    - ConsumesFiles: tool needs files attached by user
    """

    def __init__(
        self,
        req_type: RequirementType,
        capabilities: Optional[Set[KBCapability]] = None,
    ):
        self.req_type = req_type
        self.capabilities = capabilities or set()

    def satisfied_by(self, available: Set[KBCapability]) -> bool:
        """Check if the available capabilities satisfy this requirement."""
        if self.req_type == RequirementType.ANY_OF:
            return bool(self.capabilities & available)
        elif self.req_type == RequirementType.ALL_OF:
            return self.capabilities.issubset(available)
        elif self.req_type == RequirementType.CONSUMES_FILES:
            return True  # file availability checked separately
        return False


def derive_kb_filter(
    kb_capabilities: Dict[str, Set[KBCapability]],
    requirement: ToolRequirement,
) -> Optional[List[str]]:
    """
    Derive which KB IDs satisfy the tool requirement.

    Returns list of KB IDs that have the required capabilities,
    or None if all KBs should be searched (no filtering needed).
    """
    if not kb_capabilities:
        return None

    satisfying = []
    for kb_id, caps in kb_capabilities.items():
        if requirement.satisfied_by(caps):
            satisfying.append(kb_id)

    if len(satisfying) == len(kb_capabilities):
        return None  # all KBs satisfy, no filter needed

    return satisfying if satisfying else None


def kb_satisfies_tool_requirements(
    kb_capabilities: Set[KBCapability],
    requirements: List[ToolRequirement],
) -> bool:
    """Check if a KB's capabilities satisfy all tool requirements."""
    return all(req.satisfied_by(kb_capabilities) for req in requirements)


def tools_consume_files(requirements: List[ToolRequirement]) -> bool:
    """Check if any tool requirement consumes files."""
    return any(req.req_type == RequirementType.CONSUMES_FILES for req in requirements)
