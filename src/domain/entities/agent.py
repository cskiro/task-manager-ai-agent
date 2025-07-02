"""Agent domain entities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentMemory:
    """Simple memory storage for agents."""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_observation(self, observation: Dict[str, Any]) -> None:
        """Add observation to short-term memory."""
        self.short_term.append({
            "timestamp": datetime.now().isoformat(),
            "content": observation
        })
        # Keep only last 10 observations
        if len(self.short_term) > 10:
            self.short_term.pop(0)


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.max_iterations = 10
    
    @abstractmethod
    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation and decide next action."""
        pass
    
    @abstractmethod
    async def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decided action."""
        pass
    
    async def run(self, initial_input: str) -> Any:
        """Main execution loop."""
        self.state = AgentState.THINKING
        observation = {"input": initial_input, "iteration": 0}
        
        for i in range(self.max_iterations):
            try:
                # Think phase
                thought = await self.think(observation)
                self.memory.add_observation(thought)
                
                # Check if complete
                if thought.get("is_complete", False):
                    self.state = AgentState.COMPLETED
                    return thought.get("result")
                
                # Act phase
                self.state = AgentState.ACTING
                action_result = await self.act(thought)
                
                # Update observation
                observation = {
                    "previous_thought": thought,
                    "action_result": action_result,
                    "iteration": i + 1
                }
                
                self.state = AgentState.THINKING
                
            except Exception as e:
                self.state = AgentState.ERROR
                return {"error": str(e)}
        
        return {"error": "Max iterations reached"}