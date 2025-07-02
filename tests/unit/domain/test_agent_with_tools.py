"""Tests for agent tool integration."""

import pytest
from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.domain.tools.web_search import WebSearchTool
from src.infrastructure.llm import MockLLMProvider


class TestAgentWithTools:
    """Test agent integration with tools."""
    
    @pytest.mark.asyncio
    async def test_agent_can_register_tools(self):
        """Test that agent can register and access tools."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer
        )
        
        # Create and register tool
        web_search = WebSearchTool(name="web_search")
        agent.register_tool(web_search)
        
        # Assert
        assert "web_search" in agent.tools
        assert agent.tools["web_search"] == web_search
    
    @pytest.mark.asyncio
    async def test_agent_uses_web_search_for_research(self):
        """Test that agent can use web search during planning."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_tools=True
        )
        
        # Register web search tool
        web_search = WebSearchTool(name="web_search")
        agent.register_tool(web_search)
        
        # Act
        result = await agent.run("Build a modern React application with best practices")
        
        # Assert
        # Check that agent used the tool
        tool_uses = [m for m in agent.memory.short_term if m["content"].get("tool_used")]
        assert len(tool_uses) > 0
        assert tool_uses[0]["content"]["tool_used"] == "web_search"
        assert "query" in tool_uses[0]["content"]
    
    @pytest.mark.asyncio
    async def test_agent_tool_results_influence_planning(self):
        """Test that tool results influence task planning."""
        # Arrange
        llm_provider = MockLLMProvider()
        task_decomposer = TaskDecomposer()
        agent = TaskPlanningAgent(
            name="TestAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_tools=True
        )
        
        # Register web search tool
        web_search = WebSearchTool(name="web_search")
        agent.register_tool(web_search)
        
        # Act
        result = await agent.run("Build a Python task management system")
        
        # Assert
        # With web search, agent should have researched best practices
        research_found = any(
            "research" in str(m["content"]).lower() or 
            "best practices" in str(m["content"]).lower()
            for m in agent.memory.short_term
        )
        assert research_found