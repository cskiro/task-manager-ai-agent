"""Application layer models and data structures."""

from dataclasses import dataclass
from enum import Enum


class UpdateType(Enum):
    """Types of planning updates."""
    
    STARTED = "started"
    ANALYZING = "analyzing"
    DECOMPOSING = "decomposing"
    TASK_CREATED = "task_created"
    ESTIMATING = "estimating"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class PlanningUpdate:
    """
    Update event during planning process.
    
    Attributes:
        type: The type of update (phase or event)
        message: Human-readable message about the update
        progress: Overall progress from 0.0 to 1.0
        data: Optional data specific to the update type
        agent_name: Name of the agent generating the update
    """
    
    type: UpdateType
    message: str
    progress: float  # 0.0 to 1.0
    data: dict | None = None
    agent_name: str | None = None
    
    def __post_init__(self):
        """Validate progress is within bounds."""
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"Progress must be between 0.0 and 1.0, got {self.progress}")