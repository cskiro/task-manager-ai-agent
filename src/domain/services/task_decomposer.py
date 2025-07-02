"""Task decomposition service."""

import re
from typing import Protocol

from src.domain.entities.task import Task
from src.domain.value_objects import Priority, TimeEstimate


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def complete(self, prompt: str, system: str = None) -> str:
        """Get completion from LLM."""
        ...


class TaskDecomposer:
    """Service for decomposing projects into tasks using AI."""

    async def decompose(self, project_description: str, llm_provider: LLMProvider) -> list[Task]:
        """
        Decompose a project description into tasks.

        Args:
            project_description: Natural language project description
            llm_provider: LLM provider for AI-powered decomposition

        Returns:
            List of tasks with dependencies and time estimates
        """
        if not project_description or not project_description.strip():
            return []

        # Create prompt for LLM
        prompt = f"Break down this project into tasks: {project_description}"

        # Get LLM response
        llm_response = await llm_provider.complete(prompt)

        # Parse tasks from response
        return self._parse_tasks_from_response(llm_response)

    def _parse_tasks_from_response(self, response: str) -> list[Task]:
        """Parse tasks from LLM response."""
        tasks = []
        lines = response.strip().split("\n")

        for i, line in enumerate(lines):
            # Simple parsing - extract task title from numbered list
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number prefix
                parts = line.split(".", 1)
                if len(parts) > 1:
                    title = parts[1].strip()

                    # Extract time estimate if present
                    time_estimate = self._extract_time_estimate(title)
                    if time_estimate:
                        # Remove time estimate from title
                        title = re.sub(r"\s*\(.*?\)\s*$", "", title)

                    # Determine priority based on keywords
                    priority = self._determine_priority(title)

                    task = Task(title=title, priority=priority, time_estimate=time_estimate)

                    # Add dependencies based on task order and type
                    self._add_dependencies(task, tasks, i)

                    tasks.append(task)

        return tasks

    def _extract_time_estimate(self, title: str) -> TimeEstimate:
        """Extract time estimate from task title."""
        match = re.search(r"\((\d+)-(\d+)\s*hours?\)", title)
        if match:
            min_hours = float(match.group(1))
            max_hours = float(match.group(2))
            # Simple PERT estimation
            realistic = (min_hours + max_hours) / 2
            return TimeEstimate(
                optimistic_hours=min_hours, realistic_hours=realistic, pessimistic_hours=max_hours
            )
        return None

    def _determine_priority(self, title: str) -> Priority:
        """Determine task priority based on keywords."""
        title_lower = title.lower()

        # High priority keywords
        if any(
            word in title_lower
            for word in ["setup", "set up", "authentication", "security", "database"]
        ):
            return Priority.HIGH

        # Low priority keywords
        if any(word in title_lower for word in ["documentation", "testing", "polish"]):
            return Priority.LOW

        return Priority.MEDIUM

    def _add_dependencies(self, task: Task, existing_tasks: list[Task], index: int) -> None:
        """Add dependencies based on task type and order."""
        if index == 0:
            return  # First task has no dependencies

        task_lower = task.title.lower()

        # Database/backend tasks depend on setup
        if any(word in task_lower for word in ["database", "crud", "api", "backend"]):
            for prev_task in existing_tasks:
                if "setup" in prev_task.title.lower() or "set up" in prev_task.title.lower():
                    task.add_dependency(prev_task.id)
                    break

        # Frontend tasks depend on API
        if "frontend" in task_lower or "ui" in task_lower:
            for prev_task in existing_tasks:
                if "api" in prev_task.title.lower() or "backend" in prev_task.title.lower():
                    task.add_dependency(prev_task.id)

        # Authentication often depends on database
        if "authentication" in task_lower or "auth" in task_lower:
            for prev_task in existing_tasks:
                if "database" in prev_task.title.lower():
                    task.add_dependency(prev_task.id)
