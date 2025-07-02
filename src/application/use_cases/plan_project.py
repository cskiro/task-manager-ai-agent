"""Use case for planning projects with AI assistance."""

from dataclasses import dataclass
from uuid import UUID

from src.domain.entities import Task
from src.domain.value_objects import Priority, TimeEstimate


@dataclass
class PlanProjectRequest:
    """Request for planning a project."""

    project_id: UUID
    description: str
    context: str | None = None


@dataclass
class PlanProjectResponse:
    """Response from planning a project."""

    project_id: UUID
    tasks: list[Task]
    summary: str


class PlanProjectUseCase:
    """
    Use case for AI-powered project planning.

    Orchestrates domain services and external dependencies
    through dependency injection.
    """

    def __init__(self, llm_provider, task_repository):
        """Initialize with dependencies."""
        self.llm_provider = llm_provider
        self.task_repository = task_repository

    async def execute(self, request: PlanProjectRequest) -> PlanProjectResponse:
        """Execute the project planning use case."""
        # Get AI decomposition
        prompt = f"Break down this project into tasks: {request.description}"
        if request.context:
            prompt += f"\nContext: {request.context}"

        llm_response = await self.llm_provider.complete(prompt)

        # Parse tasks from LLM response
        tasks = []
        lines = llm_response.strip().split("\n")

        for i, line in enumerate(lines):
            # Simple parsing - extract task title from numbered list
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number prefix
                parts = line.split(".", 1)
                if len(parts) > 1:
                    title = parts[1].strip()

                    # Extract time estimate if present (e.g., "(2-4 hours)")
                    time_estimate = None
                    if "(" in title and "hour" in title:
                        import re

                        match = re.search(r"\((\d+)-(\d+)\s*hours?\)", title)
                        if match:
                            min_hours = float(match.group(1))
                            max_hours = float(match.group(2))
                            # Simple PERT estimation
                            realistic = (min_hours + max_hours) / 2
                            time_estimate = TimeEstimate(
                                optimistic_hours=min_hours,
                                realistic_hours=realistic,
                                pessimistic_hours=max_hours,
                            )
                            # Remove time estimate from title
                            title = re.sub(r"\s*\(.*?\)\s*$", "", title)

                    task = Task(title=title, priority=Priority.MEDIUM, time_estimate=time_estimate)

                    # Add dependency on previous task if this is setup-related
                    if i > 0 and any(
                        word in title.lower()
                        for word in ["database", "crud", "frontend", "authentication"]
                    ):
                        # Find setup task
                        for prev_task in tasks:
                            if (
                                "set up" in prev_task.title.lower()
                                or "setup" in prev_task.title.lower()
                            ):
                                task.add_dependency(prev_task.id)
                                break

                    tasks.append(task)

        # Save tasks to repository
        for task in tasks:
            await self.task_repository.save(task)

        return PlanProjectResponse(
            project_id=request.project_id,
            tasks=tasks,
            summary=f"Created {len(tasks)} tasks for project",
        )
