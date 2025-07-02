"""Tests for TaskPlanningAgent."""

import pytest
from unittest.mock import Mock, AsyncMock
from uuid import UUID

from src.domain.entities.agent import AgentState
from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.entities.task import Task
from src.domain.value_objects import Priority


class TestTaskPlanningAgent:
    """Test cases for TaskPlanningAgent."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = Mock()
        provider.complete = AsyncMock()
        provider.structured_complete = AsyncMock()
        return provider
    
    @pytest.fixture
    def mock_task_decomposer(self):
        """Create a mock task decomposer."""
        decomposer = Mock()
        decomposer.decompose = AsyncMock()
        return decomposer
    
    def test_task_planning_agent_creation(self, mock_llm_provider, mock_task_decomposer):
        """Test creating a TaskPlanningAgent."""
        agent = TaskPlanningAgent(
            name="TaskPlanner",
            llm_provider=mock_llm_provider,
            task_decomposer=mock_task_decomposer
        )
        
        assert agent.name == "TaskPlanner"
        assert agent.state == AgentState.IDLE
        assert agent.llm_provider == mock_llm_provider
        assert agent.task_decomposer == mock_task_decomposer
    
    @pytest.mark.asyncio
    async def test_agent_thinks_about_project(self, mock_llm_provider, mock_task_decomposer):
        """Test agent thinking phase for project planning."""
        agent = TaskPlanningAgent("Planner", mock_llm_provider, mock_task_decomposer)
        
        observation = {
            "input": "Build a todo app with user authentication",
            "iteration": 0
        }
        
        # Mock decomposer to return tasks
        mock_tasks = [
            Task(title="Setup project", priority=Priority.HIGH),
            Task(title="Implement auth", priority=Priority.HIGH),
            Task(title="Create UI", priority=Priority.MEDIUM)
        ]
        mock_task_decomposer.decompose.return_value = mock_tasks
        
        thought = await agent.think(observation)
        
        assert thought["action"] == "decompose_project"
        assert "tasks" in thought
        assert len(thought["tasks"]) == 3
        assert thought["is_complete"] is False
        
        # Verify decomposer was called
        mock_task_decomposer.decompose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_acts_to_create_plan(self, mock_llm_provider, mock_task_decomposer):
        """Test agent acting phase to create execution plan."""
        agent = TaskPlanningAgent("Planner", mock_llm_provider, mock_task_decomposer)
        
        # Setup some tasks
        mock_tasks = [
            Task(title="Task 1", priority=Priority.HIGH),
            Task(title="Task 2", priority=Priority.MEDIUM),
        ]
        
        action = {
            "action": "create_plan",
            "tasks": mock_tasks
        }
        
        result = await agent.act(action)
        
        assert "execution_plan" in result
        assert "total_tasks" in result
        assert result["total_tasks"] == 2
        assert result["status"] == "plan_created"
    
    @pytest.mark.asyncio
    async def test_agent_completes_planning_cycle(self, mock_llm_provider, mock_task_decomposer):
        """Test complete planning cycle from input to final plan."""
        agent = TaskPlanningAgent("Planner", mock_llm_provider, mock_task_decomposer)
        
        # Mock decomposer
        mock_tasks = [
            Task(title="Database setup", priority=Priority.HIGH),
            Task(title="API development", priority=Priority.HIGH),
            Task(title="Frontend", priority=Priority.MEDIUM)
        ]
        mock_task_decomposer.decompose.return_value = mock_tasks
        
        result = await agent.run("Create an e-commerce platform")
        
        assert "execution_plan" in result
        assert "tasks" in result
        assert len(result["tasks"]) == 3
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_handles_empty_project(self, mock_llm_provider, mock_task_decomposer):
        """Test agent handles empty or invalid project description."""
        agent = TaskPlanningAgent("Planner", mock_llm_provider, mock_task_decomposer)
        
        # Mock decomposer to return empty list
        mock_task_decomposer.decompose.return_value = []
        
        result = await agent.run("")
        
        assert "error" in result or len(result.get("tasks", [])) == 0
    
    def test_agent_has_project_context(self, mock_llm_provider, mock_task_decomposer):
        """Test agent maintains project context."""
        agent = TaskPlanningAgent("Planner", mock_llm_provider, mock_task_decomposer)
        
        assert hasattr(agent, 'project_context')
        assert agent.project_context == {}
        
        # Agent should be able to store project info
        agent.project_context["name"] = "Test Project"
        assert agent.project_context["name"] == "Test Project"