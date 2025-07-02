"""Base classes for agent tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    message: Optional[str] = None


class Tool(ABC):
    """Abstract base class for agent tools."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass