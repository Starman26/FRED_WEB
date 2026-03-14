"""
tool_registry.py

Centralized tool registry with metadata for execution, confirmation, timeouts,
and verification.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List


class ToolType(str, Enum):
    READ = "read"
    WRITE = "write"
    ACTUATE = "actuate"


class SafetyLevel(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"


@dataclass
class ToolSpec:
    name: str
    fn: Callable
    tool_type: ToolType
    safety_level: SafetyLevel
    description: str
    timeout_ms: int = 10_000
    retries: int = 0
    idempotent: bool = False
    requires_station: bool = False
    verify_fn: Optional[Callable] = None
    requires_confirmation: bool = False
    requires_safety_check: bool = False


class ToolRegistry:
    _tools: Dict[str, ToolSpec] = {}

    @classmethod
    def register(cls, spec: ToolSpec) -> None:
        cls._tools[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> Optional[ToolSpec]:
        return cls._tools.get(name)

    @classmethod
    def get_by_type(cls, tool_type: ToolType) -> List[ToolSpec]:
        return [t for t in cls._tools.values() if t.tool_type == tool_type]

    @classmethod
    def get_by_safety(cls, safety_level: SafetyLevel) -> List[ToolSpec]:
        return [t for t in cls._tools.values() if t.safety_level == safety_level]

    @classmethod
    def all_specs(cls) -> Dict[str, ToolSpec]:
        return dict(cls._tools)

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._tools.keys())
