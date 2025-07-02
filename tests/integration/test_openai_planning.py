"""Integration tests with real OpenAI API."""
import os
import pytest
from uuid import uuid4
from dotenv import load_dotenv

from src.application.use_cases.plan_project import (
    PlanProjectUseCase,
    PlanProjectRequest
)
from src.infrastructure.llm.openai_provider import OpenAIProvider
from src.application.ports.task_repository import TaskRepository
from src.domain.entities import Task


# Load environment variables
load_dotenv()


class InMemoryTaskRepository:
    """Simple in-memory implementation for integration tests."""
    
    def __init__(self):
        self.tasks = {}
    
    async def save(self, task: Task) -> None:
        self.tasks[task.id] = task
    
    async def get(self, task_id: str) -> Task:
        return self.tasks.get(task_id)
    
    async def list_all(self) -> list[Task]:
        return list(self.tasks.values())


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
class TestOpenAIPlanningIntegration:
    """Integration tests with real OpenAI API."""
    
    @pytest.mark.asyncio
    async def test_plan_real_project_with_openai(self):
        """Test planning a real project using OpenAI."""
        # Arrange
        api_key = os.getenv("OPENAI_API_KEY")
        llm_provider = OpenAIProvider(api_key=api_key)
        task_repository = InMemoryTaskRepository()
        
        use_case = PlanProjectUseCase(
            llm_provider=llm_provider,
            task_repository=task_repository
        )
        
        request = PlanProjectRequest(
            project_id=uuid4(),
            description="Build a simple blog platform with user authentication and markdown support",
            context="Web application using Python backend and React frontend"
        )
        
        # Act
        response = await use_case.execute(request)
        
        # Assert
        assert response.project_id == request.project_id
        assert len(response.tasks) >= 3  # Should create multiple tasks
        assert response.summary.startswith("Created")
        
        # Verify task quality
        task_titles = [task.title for task in response.tasks]
        
        # Should have some essential tasks
        essential_keywords = ["auth", "database", "frontend", "api"]
        found_keywords = []
        for keyword in essential_keywords:
            if any(keyword.lower() in title.lower() for title in task_titles):
                found_keywords.append(keyword)
        
        assert len(found_keywords) >= 2, f"Expected essential tasks, found: {task_titles}"
        
        # Verify tasks were saved
        saved_tasks = await task_repository.list_all()
        assert len(saved_tasks) == len(response.tasks)
        
        # Verify dependencies make sense
        for task in response.tasks:
            # Dependencies should reference existing tasks
            for dep_id in task.dependencies:
                assert any(t.id == dep_id for t in response.tasks), \
                    f"Dependency {dep_id} not found in created tasks"
        
        # Print results for manual inspection
        print(f"\n=== OpenAI Planning Results ===")
        print(f"Created {len(response.tasks)} tasks:")
        for i, task in enumerate(response.tasks, 1):
            deps = f" (depends on: {', '.join(map(str, task.dependencies))})" if task.dependencies else ""
            estimate = f" [{task.time_estimate.expected_hours}h]" if task.time_estimate else ""
            print(f"{i}. {task.title}{estimate}{deps}")
    
    @pytest.mark.asyncio
    async def test_plan_technical_project_with_details(self):
        """Test planning a technical project with specific requirements."""
        # Arrange
        api_key = os.getenv("OPENAI_API_KEY")
        llm_provider = OpenAIProvider(api_key=api_key, model="gpt-4-turbo-preview")
        task_repository = InMemoryTaskRepository()
        
        use_case = PlanProjectUseCase(
            llm_provider=llm_provider,
            task_repository=task_repository
        )
        
        request = PlanProjectRequest(
            project_id=uuid4(),
            description="""
            Create a real-time collaborative code editor with the following features:
            - Multiple users can edit the same file simultaneously
            - Syntax highlighting for Python, JavaScript, and Go
            - Real-time cursor positions and selections
            - Built-in terminal for running code
            - File explorer with project structure
            """,
            context="Using WebSockets for real-time sync, Monaco editor for the frontend"
        )
        
        # Act
        response = await use_case.execute(request)
        
        # Assert
        assert len(response.tasks) >= 5  # Complex project should have many tasks
        
        # Check for specific technical tasks
        task_titles_lower = [task.title.lower() for task in response.tasks]
        
        expected_components = ["websocket", "editor", "syntax", "terminal", "file"]
        found_components = [
            comp for comp in expected_components 
            if any(comp in title for title in task_titles_lower)
        ]
        
        assert len(found_components) >= 3, \
            f"Expected technical components in tasks, found: {found_components}"
        
        # Verify time estimates are reasonable
        total_hours = sum(
            task.time_estimate.expected_hours 
            for task in response.tasks 
            if task.time_estimate
        )
        assert 40 <= total_hours <= 500, \
            f"Total estimate {total_hours}h seems unrealistic for this project"
        
        # Verify critical path exists
        tasks_with_no_deps = [t for t in response.tasks if not t.dependencies]
        assert len(tasks_with_no_deps) >= 1, "Should have at least one starting task"