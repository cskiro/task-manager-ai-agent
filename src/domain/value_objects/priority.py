"""Priority value object for task prioritization."""

from enum import IntEnum


class Priority(IntEnum):
    """Task priority levels with numeric values for comparison."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

    @property
    def emoji(self) -> str:
        """Get emoji representation of priority."""
        return {
            Priority.LOW: "🟢",
            Priority.MEDIUM: "🟡",
            Priority.HIGH: "🟠",
            Priority.URGENT: "🔴",
            Priority.CRITICAL: "🚨",
        }[self]

    def __str__(self) -> str:
        """String representation with emoji."""
        return f"{self.emoji} {self.name}"
