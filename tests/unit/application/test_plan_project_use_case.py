"""Unit tests for PlanProjectUseCase."""

from uuid import uuid4

import pytest

from src.application.use_cases.plan_project import (
    PlanProjectRequest,
    PlanProjectResponse,
    PlanProjectUseCase,
)
from src.domain.entities import Task
from src.infrastructure.llm import MockLLMProvider


class MockTaskRepository:
    """Mock task repository for testing."""

    def __init__(self):
        self.saved_tasks = []

    async def save(self, task: Task) -> None:
        self.saved_tasks.append(task)


class TestPlanProjectUseCase:
    """Test cases for PlanProjectUseCase."""

    @pytest.mark.asyncio
    async def test_plan_simple_project(self):
        """Test planning a simple project with AI decomposition."""
        # Arrange
        project_id = uuid4()
        project_description = "Create a simple todo list web application"

        # Create request first to use in mock configuration
        request = PlanProjectRequest(
            project_id=project_id,
            description=project_description,
            context="Web development project using modern stack",
        )

        # Configure mock LLM provider with specific response
        mock_responses = {
            f"Break down this project into tasks: {project_description}\nContext: {request.context}": """
1. Set up project structure and dependencies
2. Create database schema for todos
3. Implement CRUD API endpoints
4. Build frontend UI
5. Add user authentication
"""
        }

        llm_provider = MockLLMProvider(responses=mock_responses)
        task_repository = MockTaskRepository()
        use_case = PlanProjectUseCase(llm_provider=llm_provider, task_repository=task_repository)

        # Act
        response = await use_case.execute(request)

        # Assert
        assert isinstance(response, PlanProjectResponse)
        assert response.project_id == project_id
        assert len(response.tasks) == 5
        assert response.summary == "Created 5 tasks for project"

        # Verify tasks were created with proper structure
        task_titles = [task.title for task in response.tasks]
        assert "Set up project structure and dependencies" in task_titles
        assert "Create database schema for todos" in task_titles
        assert "Implement CRUD API endpoints" in task_titles
        assert "Build frontend UI" in task_titles
        assert "Add user authentication" in task_titles

        # Verify tasks were saved to repository
        assert len(task_repository.saved_tasks) == 5

        # Verify task dependencies (setup should come first)
        setup_task = next(t for t in response.tasks if "Set up project" in t.title)
        db_task = next(t for t in response.tasks if "database schema" in t.title)
        assert setup_task.id in db_task.dependencies

    @pytest.mark.asyncio
    async def test_plan_project_with_time_estimates(self):
        """Test that AI-generated tasks include time estimates."""
        # Arrange
        llm_provider = MockLLMProvider()  # Will use default web project response
        task_repository = MockTaskRepository()
        use_case = PlanProjectUseCase(llm_provider=llm_provider, task_repository=task_repository)

        request = PlanProjectRequest(
            project_id=uuid4(), description="Break down a web application project"
        )

        # Act
        response = await use_case.execute(request)

        # Assert - at least one task should have time estimates
        tasks_with_estimates = [t for t in response.tasks if t.time_estimate is not None]
        assert len(tasks_with_estimates) > 0

        # Check that time estimates are properly structured
        first_estimate = tasks_with_estimates[0].time_estimate
        assert first_estimate.optimistic_hours > 0
        assert first_estimate.realistic_hours >= first_estimate.optimistic_hours
        assert first_estimate.pessimistic_hours >= first_estimate.realistic_hours
