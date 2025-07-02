"""Task status value object representing task lifecycle states."""

from enum import Enum


class TaskStatus(Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

    def can_transition_to(self, new_status: "TaskStatus") -> bool:
        """Check if transition to new status is valid."""
        valid_transitions = {
            TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
            TaskStatus.IN_PROGRESS: {
                TaskStatus.COMPLETED,
                TaskStatus.BLOCKED,
                TaskStatus.CANCELLED,
            },
            TaskStatus.BLOCKED: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
            TaskStatus.COMPLETED: set(),  # Terminal state
            TaskStatus.CANCELLED: set(),  # Terminal state
        }
        return new_status in valid_transitions.get(self, set())
