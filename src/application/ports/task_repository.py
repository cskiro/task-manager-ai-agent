"""Protocol for task repository."""

from typing import Protocol
from uuid import UUID

from src.domain.entities import Task


class TaskRepository(Protocol):
    """Protocol for task persistence."""

    async def save(self, task: Task) -> None:
        """Save a task."""
        ...

    async def get(self, task_id: UUID) -> Task | None:
        """Get a task by ID."""
        ...

    async def list_by_project(self, project_id: UUID) -> list[Task]:
        """List all tasks for a project."""
        ...

    async def update(self, task: Task) -> None:
        """Update an existing task."""
        ...

    async def delete(self, task_id: UUID) -> None:
        """Delete a task."""
        ...
