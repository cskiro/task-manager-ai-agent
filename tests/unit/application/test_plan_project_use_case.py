"""Unit tests for PlanProjectUseCase."""
import pytest
from uuid import uuid4

from src.application.use_cases.plan_project import (
    PlanProjectUseCase,
    PlanProjectRequest,
    PlanProjectResponse
)
from src.application.ports.llm_provider import LLMProvider
from src.application.ports.task_repository import TaskRepository
from src.domain.entities import Task
from src.domain.value_objects import Priority, TimeEstimate


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
        
        # Mock LLM provider that returns predictable task decomposition
        class MockLLMProvider:
            async def complete(self, prompt: str, system: str = None) -> str:
                return """
                1. Set up project structure and dependencies
                2. Create database schema for todos
                3. Implement CRUD API endpoints
                4. Build frontend UI
                5. Add user authentication
                """
        
        # Use the mock task repository
        
        llm_provider = MockLLMProvider()
        task_repository = MockTaskRepository()
        use_case = PlanProjectUseCase(
            llm_provider=llm_provider,
            task_repository=task_repository
        )
        
        request = PlanProjectRequest(
            project_id=project_id,
            description=project_description,
            context="Web development project using modern stack"
        )
        
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
        class MockLLMProviderWithEstimates:
            async def complete(self, prompt: str, system: str = None) -> str:
                return """
                1. Set up project structure (2-4 hours)
                2. Implement core logic (8-16 hours)
                """
        
        llm_provider = MockLLMProviderWithEstimates()
        task_repository = MockTaskRepository()
        use_case = PlanProjectUseCase(
            llm_provider=llm_provider,
            task_repository=task_repository
        )
        
        request = PlanProjectRequest(
            project_id=uuid4(),
            description="Build a feature"
        )
        
        # Act
        response = await use_case.execute(request)
        
        # Assert
        setup_task = next(t for t in response.tasks if "Set up project" in t.title)
        assert setup_task.time_estimate is not None
        assert setup_task.time_estimate.optimistic_hours == 2.0
        assert setup_task.time_estimate.realistic_hours == 3.0
        assert setup_task.time_estimate.pessimistic_hours == 4.0