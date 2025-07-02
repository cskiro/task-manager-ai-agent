"""Simple JSON-based task repository implementation."""

import json
from pathlib import Path
from uuid import UUID

from src.domain.entities.task import Task
from src.domain.value_objects import Priority, TaskStatus, TimeEstimate


class JSONTaskRepository:
    """Simple JSON file-based task repository."""

    def __init__(self, file_path: Path):
        """Initialize repository with file path."""
        self.file_path = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensure the JSON file exists with empty structure."""
        if not self.file_path.exists() or not self.file_path.read_text().strip():
            self.file_path.write_text(json.dumps({"tasks": []}, indent=2))

    def _load_data(self) -> dict:
        """Load data from JSON file."""
        return json.loads(self.file_path.read_text())

    def _save_data(self, data: dict):
        """Save data to JSON file."""
        self.file_path.write_text(json.dumps(data, indent=2))

    def _task_to_dict(self, task: Task) -> dict:
        """Convert task to dictionary for JSON storage."""
        return {
            "id": str(task.id),
            "title": task.title,
            "description": task.description,
            "priority": task.priority.value,
            "status": task.status.value,
            "dependencies": [str(dep) for dep in task.dependencies],
            "time_estimate": (
                {
                    "optimistic": task.time_estimate.optimistic_hours,
                    "realistic": task.time_estimate.realistic_hours,
                    "pessimistic": task.time_estimate.pessimistic_hours,
                }
                if task.time_estimate
                else None
            ),
        }

    def _dict_to_task(self, data: dict) -> Task:
        """Convert dictionary to task."""
        time_estimate = None
        if data.get("time_estimate"):
            te = data["time_estimate"]
            time_estimate = TimeEstimate(
                optimistic_hours=te["optimistic"],
                realistic_hours=te["realistic"],
                pessimistic_hours=te["pessimistic"],
            )

        return Task(
            id=UUID(data["id"]),
            title=data["title"],
            description=data.get("description", ""),
            priority=Priority(data["priority"]),
            status=TaskStatus(data["status"]),
            dependencies=[UUID(dep) for dep in data.get("dependencies", [])],
            time_estimate=time_estimate,
        )

    async def save(self, task: Task):
        """Save a task to the repository."""
        data = self._load_data()

        # Check if task already exists
        task_dict = self._task_to_dict(task)
        existing_index = None

        for i, t in enumerate(data["tasks"]):
            if t["id"] == task_dict["id"]:
                existing_index = i
                break

        if existing_index is not None:
            # Update existing
            data["tasks"][existing_index] = task_dict
        else:
            # Add new
            data["tasks"].append(task_dict)

        self._save_data(data)

    async def get_by_id(self, task_id: UUID) -> Task | None:
        """Get a task by ID."""
        data = self._load_data()

        for task_data in data["tasks"]:
            if task_data["id"] == str(task_id):
                return self._dict_to_task(task_data)

        return None

    async def get_all(self) -> list[Task]:
        """Get all tasks."""
        data = self._load_data()
        return [self._dict_to_task(t) for t in data["tasks"]]

    async def update(self, task: Task):
        """Update a task (same as save for this implementation)."""
        await self.save(task)

    async def delete(self, task_id: UUID):
        """Delete a task by ID."""
        data = self._load_data()

        # Filter out the task to delete
        data["tasks"] = [t for t in data["tasks"] if t["id"] != str(task_id)]

        self._save_data(data)
