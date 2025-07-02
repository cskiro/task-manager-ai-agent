"""Tests for agent domain entities."""

import pytest
from typing import Dict, Any
from abc import ABC

from src.domain.entities.agent import BaseAgent, AgentState


class TestBaseAgent:
    """Test cases for BaseAgent abstract class."""
    
    def test_cannot_instantiate_abstract_base_agent(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent("TestAgent")
    
    def test_agent_has_required_attributes(self):
        """Test that agent has all required attributes."""
        # Create a concrete implementation for testing
        class ConcreteAgent(BaseAgent):
            async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
                return {"thought": "test"}
            
            async def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "test"}
        
        agent = ConcreteAgent("TestAgent")
        
        assert agent.name == "TestAgent"
        assert agent.state == AgentState.IDLE
        assert hasattr(agent, 'memory')
        assert hasattr(agent, 'think')
        assert hasattr(agent, 'act')
        assert hasattr(agent, 'run')
    
    @pytest.mark.asyncio
    async def test_agent_run_basic_cycle(self):
        """Test basic think-act cycle."""
        class SimpleAgent(BaseAgent):
            async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
                if observation.get("iteration", 0) == 0:
                    return {
                        "thought": "I need to process this",
                        "action": "process",
                        "is_complete": False
                    }
                else:
                    return {
                        "thought": "Task complete",
                        "is_complete": True,
                        "result": "Done"
                    }
            
            async def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "processed"}
        
        agent = SimpleAgent("TestAgent")
        result = await agent.run("Test input")
        
        assert result == "Done"
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_state_transitions(self):
        """Test agent state transitions during execution."""
        states_observed = []
        
        class StateTrackingAgent(BaseAgent):
            async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
                states_observed.append(self.state)
                return {"thought": "Done", "is_complete": True, "result": "Complete"}
            
            async def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
                states_observed.append(self.state)
                return {"status": "ok"}
        
        agent = StateTrackingAgent("StateAgent")
        assert agent.state == AgentState.IDLE
        
        await agent.run("Test")
        
        assert AgentState.THINKING in states_observed
        assert agent.state == AgentState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_agent_memory_stores_observations(self):
        """Test that agent memory stores observations."""
        class MemoryAgent(BaseAgent):
            async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "thought": "Remembering",
                    "is_complete": True,
                    "result": f"Memory has {len(self.memory.short_term)} items"
                }
            
            async def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
                return {"status": "ok"}
        
        agent = MemoryAgent("MemoryAgent")
        result = await agent.run("Remember this")
        
        assert len(agent.memory.short_term) > 0
        assert "Memory has" in result


class TestAgentState:
    """Test cases for AgentState enum."""
    
    def test_agent_states_exist(self):
        """Test that all required agent states exist."""
        assert AgentState.IDLE
        assert AgentState.THINKING
        assert AgentState.ACTING
        assert AgentState.COMPLETED
        assert AgentState.ERROR
    
    def test_agent_state_values(self):
        """Test agent state string values."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.THINKING.value == "thinking"
        assert AgentState.ACTING.value == "acting"
        assert AgentState.COMPLETED.value == "completed"
        assert AgentState.ERROR.value == "error"