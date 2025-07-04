"""Unit tests for streaming plan runner."""

import asyncio
from uuid import uuid4

import pytest

from src.application.use_cases.plan_project import PlanProjectRequest
from src.domain.entities import Task
from src.domain.value_objects import Priority, TaskStatus, TimeEstimate
from src.application.use_cases.streaming_runner import StreamingPlanRunner
from src.application.models import UpdateType, PlanningUpdate


class MockLLMProvider:
    """Mock LLM that simulates streaming responses."""
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Simulate LLM completion with delay."""
        await asyncio.sleep(0.01)  # Simulate network delay
        return "Task 1: Setup project\nTask 2: Implement features"
    
    async def structured_complete(self, prompt: str, response_model, **kwargs):
        """Return mock structured data."""
        await asyncio.sleep(0.01)
        # Return mock tasks based on the response model
        return {
            "tasks": [
                {
                    "title": "Setup project",
                    "description": "Initialize the project structure",
                    "priority": 3,
                    "dependencies": [],
                    "time_estimate": {"optimistic": 1.0, "realistic": 2.0, "pessimistic": 3.0}
                },
                {
                    "title": "Implement features",
                    "description": "Build core functionality",
                    "priority": 4,
                    "dependencies": [],
                    "time_estimate": {"optimistic": 4.0, "realistic": 8.0, "pessimistic": 12.0}
                }
            ]
        }


class TestStreamingPlanRunner:
    """Test streaming plan runner functionality."""
    
    @pytest.mark.asyncio
    async def test_streaming_updates_during_planning(self):
        """Test that planning emits progress updates."""
        # Arrange
        updates_received = []
        
        async def handle_update(update: PlanningUpdate):
            updates_received.append(update)
        
        runner = StreamingPlanRunner(
            llm_provider=MockLLMProvider(),
            task_repository=MockTaskRepository()
        )
        
        request = PlanProjectRequest(
            project_id=uuid4(),
            description="Build a blog platform",
            context="Using Python and React"
        )
        
        # Act
        result = await runner.execute(request, handle_update)
        
        # Assert
        assert len(updates_received) >= 5  # Multiple update types
        
        # Check update sequence
        update_types = [u.type for u in updates_received]
        assert update_types[0] == UpdateType.STARTED
        assert update_types[-1] == UpdateType.COMPLETED
        assert UpdateType.ANALYZING in update_types
        assert UpdateType.DECOMPOSING in update_types
        
        # Check progress increases
        progress_values = [u.progress for u in updates_received]
        assert progress_values[0] == 0.0
        assert progress_values[-1] == 1.0
        assert all(progress_values[i] <= progress_values[i+1] 
                  for i in range(len(progress_values)-1))
    
    @pytest.mark.asyncio
    async def test_task_creation_updates(self):
        """Test updates when individual tasks are created."""
        # Arrange
        task_updates = []
        
        async def handle_update(update: PlanningUpdate):
            if update.type == UpdateType.TASK_CREATED:
                task_updates.append(update)
        
        runner = StreamingPlanRunner(
            llm_provider=MockLLMProvider(),
            task_repository=MockTaskRepository()
        )
        
        request = PlanProjectRequest(
            project_id=uuid4(),
            description="Create REST API"
        )
        
        # Act
        await runner.execute(request, handle_update)
        
        # Assert
        assert len(task_updates) == 2  # Two tasks created
        assert all(u.data and "task" in u.data for u in task_updates)
        assert task_updates[0].message.startswith("Created task:")
    
    @pytest.mark.asyncio
    async def test_agent_identification_in_updates(self):
        """Test that updates identify which agent is working."""
        # Arrange
        agent_updates = []
        
        async def handle_update(update: PlanningUpdate):
            if update.agent_name:
                agent_updates.append(update)
        
        runner = StreamingPlanRunner(
            llm_provider=MockLLMProvider(),
            task_repository=MockTaskRepository()
        )
        
        request = PlanProjectRequest(
            project_id=uuid4(),
            description="Build mobile app"
        )
        
        # Act
        await runner.execute(request, handle_update)
        
        # Assert
        assert len(agent_updates) > 0
        agent_names = {u.agent_name for u in agent_updates}
        assert "Task Planner" in agent_names
    
    @pytest.mark.asyncio
    async def test_error_handling_with_updates(self):
        """Test that errors are reported through updates."""
        # Arrange
        updates = []
        
        async def handle_update(update: PlanningUpdate):
            updates.append(update)
        
        # LLM that fails
        class FailingLLM:
            async def complete(self, *args, **kwargs):
                raise RuntimeError("LLM connection failed")
            
            async def structured_complete(self, *args, **kwargs):
                raise RuntimeError("LLM connection failed")
        
        runner = StreamingPlanRunner(
            llm_provider=FailingLLM(),
            task_repository=MockTaskRepository()
        )
        
        request = PlanProjectRequest(
            project_id=uuid4(),
            description="Test project"
        )
        
        # Act
        with pytest.raises(RuntimeError):
            await runner.execute(request, handle_update)
        
        # Assert
        assert updates[-1].type == UpdateType.ERROR
        assert "LLM connection failed" in updates[-1].message
        assert updates[-1].progress < 1.0
    
    @pytest.mark.asyncio 
    async def test_cancellation_support(self):
        """Test that planning can be cancelled mid-stream."""
        # Arrange
        updates = []
        
        async def handle_update(update: PlanningUpdate):
            updates.append(update)
            # Cancel after analyzing
            if update.type == UpdateType.ANALYZING:
                raise asyncio.CancelledError()
        
        runner = StreamingPlanRunner(
            llm_provider=MockLLMProvider(),
            task_repository=MockTaskRepository()
        )
        
        request = PlanProjectRequest(
            project_id=uuid4(),
            description="Cancellable project"
        )
        
        # Act
        with pytest.raises(asyncio.CancelledError):
            await runner.execute(request, handle_update)
        
        # Assert
        assert len(updates) > 0
        assert updates[-1].type == UpdateType.ANALYZING
        # Should not reach COMPLETED
        assert not any(u.type == UpdateType.COMPLETED for u in updates)


class MockTaskRepository:
    """Mock task repository for tests."""
    
    async def save(self, task):
        """Mock save."""
        pass
    
    async def get_by_project(self, project_id):
        """Mock retrieval."""
        return []


