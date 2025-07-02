"""Tests for agent integration with calculator tool."""

import pytest
from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.domain.tools.calculator import CalculatorTool
from src.infrastructure.llm import MockLLMProvider


class TestAgentCalculatorIntegration:
    """Test agent using calculator tool for estimations."""
    
    @pytest.mark.asyncio
    async def test_agent_uses_calculator_for_estimations(self):
        """Test that agent can use calculator for time/cost estimations."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="EstimationAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_tools=True,
            enable_estimation=True  # New flag for estimation features
        )
        
        # Register calculator tool
        calculator = CalculatorTool(name="calculator")
        agent.register_tool(calculator)
        
        # Act
        result = await agent.run("Build a web app with 5 developers")
        
        # Assert
        # Check that agent has estimation calculations
        assert "estimations" in result
        estimations = result["estimations"]
        
        # Should have total hours calculation
        assert "total_hours" in estimations
        assert isinstance(estimations["total_hours"], (int, float))
        
        # Should have cost estimation if hourly rate provided
        assert "total_cost" in estimations or "cost_calculation" in estimations
    
    @pytest.mark.asyncio
    async def test_agent_calculates_parallel_task_time(self):
        """Test agent calculates time savings from parallel execution."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="ParallelAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_tools=True,
            enable_estimation=True
        )
        
        calculator = CalculatorTool(name="calculator")
        agent.register_tool(calculator)
        
        # Act
        result = await agent.run("Build microservices that can be developed in parallel")
        
        # Assert
        execution_plan = result.get("execution_plan", [])
        
        # Find parallel steps
        parallel_steps = [step for step in execution_plan if step.get("can_parallel")]
        
        if parallel_steps:
            # Check time calculations for parallel execution
            assert "parallel_time_saved" in result or any(
                "parallel" in str(m["content"]).lower() 
                for m in agent.memory.short_term
            )