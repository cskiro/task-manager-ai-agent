"""Tests for Chain of Thought (CoT) reasoning capability."""

import pytest

from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.infrastructure.llm import MockLLMProvider


class TestCoTReasoning:
    """Test Chain of Thought reasoning in agents."""

    @pytest.mark.asyncio
    async def test_agent_generates_reasoning_steps(self):
        """Test that agent generates step-by-step reasoning."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_cot=True,
        )

        # Act
        result = await agent.run("Build an e-commerce website")

        # Assert
        assert "reasoning_steps" in result
        assert isinstance(result["reasoning_steps"], list)
        assert len(result["reasoning_steps"]) > 0

        # Each step should have structure
        first_step = result["reasoning_steps"][0]
        assert "step" in first_step
        assert "thought" in first_step
        assert "conclusion" in first_step

    @pytest.mark.asyncio
    async def test_reasoning_steps_stored_in_memory(self):
        """Test that reasoning steps are stored in agent memory."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_cot=True,
        )

        # Act
        await agent.run("Build a mobile app")

        # Assert
        # Check that memory contains reasoning steps
        memory_items = agent.memory.short_term
        reasoning_memories = [
            m for m in memory_items if m["content"].get("type") == "reasoning_step"
        ]

        assert len(reasoning_memories) > 0
        assert reasoning_memories[0]["content"]["step"] == 1
        assert "thought" in reasoning_memories[0]["content"]

    @pytest.mark.asyncio
    async def test_cot_improves_task_decomposition(self):
        """Test that CoT reasoning improves task quality."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_cot=True,  # Enable Chain of Thought
        )

        # Act
        result = await agent.run("Create a complex distributed system")

        # Assert
        # With CoT, we should get more detailed analysis
        reasoning_steps = result.get("reasoning_steps", [])
        assert any("architecture" in step["thought"].lower() for step in reasoning_steps)
        assert any("dependencies" in step["thought"].lower() for step in reasoning_steps)
        assert any(
            "scale" in step["thought"].lower() or "scaling" in step["thought"].lower()
            for step in reasoning_steps
        )

    @pytest.mark.asyncio
    async def test_cot_reasoning_format(self):
        """Test that CoT reasoning follows expected format."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_cot=True,
        )

        # Act
        result = await agent.run("Build a real-time chat application")

        # Assert
        reasoning_steps = result["reasoning_steps"]

        # Should have multiple steps
        assert len(reasoning_steps) >= 3

        # Each step should be numbered sequentially
        for i, step in enumerate(reasoning_steps):
            assert step["step"] == i + 1
            assert isinstance(step["thought"], str)
            assert len(step["thought"]) > 10  # Non-trivial thought
            assert isinstance(step["conclusion"], str)
