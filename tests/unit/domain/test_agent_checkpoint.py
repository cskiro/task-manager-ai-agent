"""Tests for agent checkpoint functionality."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from src.domain.entities.agent import AgentMemory, AgentState
from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.infrastructure.llm import MockLLMProvider


class TestAgentCheckpoint:
    """Test agent checkpoint save and restore functionality."""

    @pytest.mark.asyncio
    async def test_agent_can_save_checkpoint(self, tmp_path):
        """Agent should be able to save its state to a checkpoint."""
        # Arrange
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=MockLLMProvider(),
            task_decomposer=TaskDecomposer(),
        )
        
        # Add some state to the agent
        agent.state = AgentState.THINKING
        agent.memory.add_observation({"thought": "Planning a web app"})
        agent.memory.add_observation({"action": "Decomposing tasks"})
        
        checkpoint_path = tmp_path / "checkpoint.json"
        
        # Act
        await agent.save_checkpoint(checkpoint_path)
        
        # Assert
        assert checkpoint_path.exists()
        
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        
        assert checkpoint_data["agent_name"] == "TestAgent"
        assert checkpoint_data["state"] == "thinking"
        assert len(checkpoint_data["memory"]["short_term"]) == 2
        assert checkpoint_data["timestamp"] is not None

    @pytest.mark.asyncio
    async def test_agent_can_restore_from_checkpoint(self, tmp_path):
        """Agent should be able to restore its state from a checkpoint."""
        # Arrange
        original_agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=MockLLMProvider(),
            task_decomposer=TaskDecomposer(),
        )
        
        # Set up original state
        original_agent.state = AgentState.ACTING
        original_agent.memory.add_observation({"thought": "Building features"})
        original_agent.memory.add_observation({"tool": "calculator", "result": 42})
        
        checkpoint_path = tmp_path / "checkpoint.json"
        await original_agent.save_checkpoint(checkpoint_path)
        
        # Create new agent
        new_agent = TaskPlanningAgent(
            name="RestoredAgent",
            llm_provider=MockLLMProvider(),
            task_decomposer=TaskDecomposer(),
        )
        
        # Act
        await new_agent.restore_checkpoint(checkpoint_path)
        
        # Assert
        assert new_agent.name == "TestAgent"  # Name restored from checkpoint
        assert new_agent.state == AgentState.ACTING
        assert len(new_agent.memory.short_term) == 2
        assert new_agent.memory.short_term[0]["content"]["thought"] == "Building features"
        assert new_agent.memory.short_term[1]["content"]["tool"] == "calculator"

    @pytest.mark.asyncio
    async def test_checkpoint_includes_task_context(self, tmp_path):
        """Checkpoint should include current task context for resumption."""
        # Arrange
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=MockLLMProvider(),
            task_decomposer=TaskDecomposer(),
        )
        
        # Simulate working on a task
        task_context = {
            "original_request": "Build a REST API with authentication",
            "current_step": 3,
            "total_steps": 8,
            "tasks_completed": ["Setup project", "Design schema", "Create models"],
            "tasks_remaining": ["Add authentication", "Create endpoints", "Write tests"],
        }
        
        agent.set_task_context(task_context)
        
        checkpoint_path = tmp_path / "checkpoint.json"
        
        # Act
        await agent.save_checkpoint(checkpoint_path)
        
        # Assert
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        
        assert "task_context" in checkpoint_data
        assert checkpoint_data["task_context"]["current_step"] == 3
        assert len(checkpoint_data["task_context"]["tasks_completed"]) == 3

    @pytest.mark.asyncio
    async def test_checkpoint_versioning(self, tmp_path):
        """Checkpoints should include version for compatibility."""
        # Arrange
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=MockLLMProvider(),
            task_decomposer=TaskDecomposer(),
        )
        
        checkpoint_path = tmp_path / "checkpoint.json"
        
        # Act
        await agent.save_checkpoint(checkpoint_path)
        
        # Assert
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        
        assert "version" in checkpoint_data
        assert checkpoint_data["version"] == "1.0"
        assert "agent_type" in checkpoint_data
        assert checkpoint_data["agent_type"] == "TaskPlanningAgent"

    @pytest.mark.asyncio
    async def test_auto_checkpoint_on_milestone(self):
        """Agent should auto-checkpoint after completing major milestones."""
        # Arrange
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=MockLLMProvider(),
            task_decomposer=TaskDecomposer(),
            enable_auto_checkpoint=True,
        )
        
        # Act
        result = await agent.run("Build a simple web app")
        
        # Assert
        # Check that checkpoint was created
        checkpoint_dir = Path.home() / ".taskman" / "checkpoints"
        checkpoints = list(checkpoint_dir.glob(f"{agent.name}_*.json"))
        
        assert len(checkpoints) > 0
        
        # Verify checkpoint contains milestone data
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        with open(latest_checkpoint) as f:
            data = json.load(f)
        
        assert data["milestone"] == "task_decomposition_complete"
        assert "task_context" in data or "current_tasks" in data.get("task_context", {})

    @pytest.mark.asyncio
    async def test_resume_interrupted_session(self, tmp_path):
        """Should be able to resume an interrupted agent session."""
        # Arrange
        # Create checkpoint from interrupted session
        checkpoint_data = {
            "version": "1.0",
            "agent_type": "TaskPlanningAgent",
            "agent_name": "ProjectPlanner",
            "state": "acting",
            "timestamp": datetime.now().isoformat(),
            "task_context": {
                "original_request": "Create a mobile app",
                "current_step": 5,
                "total_steps": 10,
                "tasks_completed": [
                    "Setup React Native project",
                    "Design UI mockups",
                    "Implement navigation",
                    "Create data models",
                    "Build authentication flow"
                ],
                "tasks_remaining": [
                    "Integrate API",
                    "Add offline support",
                    "Implement push notifications",
                    "Write tests",
                    "Deploy to stores"
                ]
            },
            "memory": {
                "short_term": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "content": {"thought": "Authentication flow complete, moving to API integration"}
                    }
                ]
            },
            "reasoning_traces": [
                {"step": 1, "thought": "Mobile app needs React Native for cross-platform"},
                {"step": 2, "thought": "Authentication is critical, implement first"},
            ]
        }
        
        checkpoint_path = tmp_path / "interrupted_session.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)
        
        # Create agent and restore
        agent = TaskPlanningAgent(
            name="NewAgent",
            llm_provider=MockLLMProvider(),
            task_decomposer=TaskDecomposer(),
        )
        
        # Act
        await agent.restore_checkpoint(checkpoint_path)
        resume_info = agent.get_resume_summary()
        
        # Assert
        assert agent.name == "ProjectPlanner"
        assert resume_info["progress_percentage"] == 50  # 5 of 10 tasks
        assert resume_info["last_completed"] == "Build authentication flow"
        assert resume_info["next_task"] == "Integrate API"
        assert resume_info["estimated_remaining_hours"] > 0

    def test_checkpoint_file_rotation(self, tmp_path):
        """Should rotate checkpoint files to prevent unlimited growth."""
        # Arrange
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=MockLLMProvider(),
            task_decomposer=TaskDecomposer(),
            checkpoint_dir=tmp_path,
            max_checkpoints=3,
        )
        
        # Act - Create multiple checkpoints
        for i in range(5):
            agent.create_checkpoint(milestone=f"step_{i}")
        
        # Assert
        checkpoints = list(tmp_path.glob("*.json"))
        assert len(checkpoints) == 3  # Only keep latest 3
        
        # Verify we kept the most recent ones by checking milestone content
        milestones = []
        for checkpoint_file in checkpoints:
            with open(checkpoint_file) as f:
                data = json.load(f)
                milestones.append(data.get("milestone", ""))
        
        # Should have kept the last 3 milestones: step_2, step_3, step_4
        assert "step_2" in milestones
        assert "step_3" in milestones
        assert "step_4" in milestones
        assert "step_0" not in milestones  # Oldest should be removed
        assert "step_1" not in milestones  # Second oldest should be removed