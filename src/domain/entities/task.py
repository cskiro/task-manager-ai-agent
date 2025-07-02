"""Task entity representing a unit of work."""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from ..value_objects.priority import Priority
from ..value_objects.task_status import TaskStatus
from ..value_objects.time_estimate import TimeEstimate


@dataclass
class Task:
    """
    Task entity with rich domain behavior.

    Represents a unit of work with lifecycle management,
    dependencies, and time estimation.
    """

    id: UUID = field(default_factory=uuid4)
    title: str = ""
    description: str = ""
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[UUID] = field(default_factory=list)
    time_estimate: TimeEstimate | None = None

    def __post_init__(self):
        """Validate task after initialization."""
        if not self.title:
            raise ValueError("Task title cannot be empty")

    def can_start(self, completed_tasks: set[UUID]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)

    def start(self) -> None:
        """Mark task as started with validation."""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot start task in {self.status.value} status")
        if not self.can_start(set()):  # In real usage, pass actual completed tasks
            raise ValueError("Cannot start task with unsatisfied dependencies")
        self.status = TaskStatus.IN_PROGRESS

    def complete(self) -> None:
        """Mark task as completed."""
        if self.status != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete task in {self.status.value} status")
        self.status = TaskStatus.COMPLETED

    def block(self, reason: str = "") -> None:
        """Mark task as blocked."""
        if self.status != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Cannot block task in {self.status.value} status")
        self.status = TaskStatus.BLOCKED

    def cancel(self, reason: str = "") -> None:
        """Cancel the task."""
        if self.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}:
            raise ValueError(f"Cannot cancel task in {self.status.value} status")
        self.status = TaskStatus.CANCELLED

    def add_dependency(self, task_id: UUID) -> None:
        """Add a dependency to this task."""
        if task_id == self.id:
            raise ValueError("Task cannot depend on itself")
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)

    def remove_dependency(self, task_id: UUID) -> None:
        """Remove a dependency from this task."""
        if task_id in self.dependencies:
            self.dependencies.remove(task_id)

    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == TaskStatus.COMPLETED

    @property
    def is_blocked(self) -> bool:
        """Check if task is blocked."""
        return self.status == TaskStatus.BLOCKED

    @property
    def expected_hours(self) -> float:
        """Get expected hours for the task."""
        if self.time_estimate:
            return self.time_estimate.expected_hours
        return 0.0
