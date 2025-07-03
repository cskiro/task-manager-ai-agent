"""Agent checkpoint domain entity."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class AgentCheckpoint:
    """Represents a saved state of an agent at a point in time."""

    version: str = "1.0"
    agent_type: str = ""
    agent_name: str = ""
    state: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    memory: dict[str, Any] = field(default_factory=dict)
    task_context: Optional[dict[str, Any]] = None
    reasoning_traces: list[dict[str, Any]] = field(default_factory=list)
    milestone: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            "version": self.version,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "state": self.state,
            "timestamp": self.timestamp,
            "memory": self.memory,
            "task_context": self.task_context,
            "reasoning_traces": self.reasoning_traces,
            "milestone": self.milestone,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCheckpoint":
        """Create checkpoint from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            agent_type=data.get("agent_type", ""),
            agent_name=data.get("agent_name", ""),
            state=data.get("state", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            memory=data.get("memory", {}),
            task_context=data.get("task_context"),
            reasoning_traces=data.get("reasoning_traces", []),
            milestone=data.get("milestone"),
        )