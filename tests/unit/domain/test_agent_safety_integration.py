"""Tests for agent integration with safety filters."""

import pytest

from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.infrastructure.llm import MockLLMProvider


class TestAgentSafetyIntegration:
    """Test agent with safety filters enabled."""

    @pytest.mark.asyncio
    async def test_agent_rejects_code_injection(self):
        """Agent should reject input with code injection attempts."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="SafeAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_safety=True,
        )

        # Act
        result = await agent.run("Build a project and exec('rm -rf /')")

        # Assert
        assert "error" in result
        assert "safety check failed" in result["error"].lower()
        assert "code injection" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_agent_accepts_safe_input(self):
        """Agent should process legitimate project descriptions."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="SafeAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_safety=True,
        )

        # Act
        result = await agent.run("Build an e-commerce platform")

        # Assert
        assert "error" not in result
        assert "tasks" in result
        assert len(result["tasks"]) > 0

    @pytest.mark.asyncio
    async def test_agent_caps_excessive_tasks(self):
        """Agent should limit number of generated tasks."""

        # Arrange
        # Create a mock LLM that would generate many tasks in the format expected by TaskDecomposer
        class ExcessiveTaskLLM:
            async def complete(self, prompt: str, **kwargs) -> str:
                # Generate 50 tasks in numbered list format
                task_list = []
                for i in range(50):
                    task_list.append(f"{i+1}. Task {i} (2 hours)")
                return "\n".join(task_list)

        llm_provider = ExcessiveTaskLLM()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="SafeAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_safety=True,
            max_tasks=10,  # Limit to 10 tasks
        )

        # Act
        result = await agent.run("Build everything")

        # Assert
        assert "error" in result
        assert "safety check failed" in result["error"].lower()
        assert "too many tasks" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_agent_safety_can_be_disabled(self):
        """Safety checks can be disabled via feature flag."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="UnsafeAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_safety=False,  # Disable safety
        )

        # Act - This would normally be blocked
        result = await agent.run("Build a project with eval('test')")

        # Assert - Should process normally without safety checks
        assert "error" not in result
        assert "tasks" in result
