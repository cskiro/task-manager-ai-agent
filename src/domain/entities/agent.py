"""Agent domain entities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


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

    short_term: list[dict[str, Any]] = field(default_factory=list)

    def add_observation(self, observation: dict[str, Any]) -> None:
        """Add observation to short-term memory."""
        self.short_term.append({"timestamp": datetime.now().isoformat(), "content": observation})
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
        self.tools = {}
        self.task_context: Optional[dict[str, Any]] = None
        self.reasoning_traces: list[dict[str, Any]] = []

    @abstractmethod
    async def think(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Process observation and decide next action."""
        pass

    @abstractmethod
    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
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
                    # If there's an error, return it at the top level
                    if "error" in thought:
                        return {"error": thought["error"]}
                    return thought.get("result")

                # Act phase
                self.state = AgentState.ACTING
                action_result = await self.act(thought)

                # Update observation
                observation = {
                    "previous_thought": thought,
                    "action_result": action_result,
                    "iteration": i + 1,
                }

                self.state = AgentState.THINKING

            except Exception as e:
                self.state = AgentState.ERROR
                return {"error": str(e)}

        return {"error": "Max iterations reached"}

    def register_tool(self, tool) -> None:
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool
    
    def set_task_context(self, context: dict[str, Any]) -> None:
        """Set the current task context for checkpointing."""
        self.task_context = context
    
    def get_resume_summary(self) -> dict[str, Any]:
        """Get summary of current progress for resuming."""
        if not self.task_context:
            return {"message": "No task context available"}
        
        completed = len(self.task_context.get("tasks_completed", []))
        total = self.task_context.get("total_steps", 0)
        
        return {
            "progress_percentage": int((completed / total * 100)) if total > 0 else 0,
            "last_completed": self.task_context.get("tasks_completed", ["None"])[-1],
            "next_task": self.task_context.get("tasks_remaining", ["None"])[0],
            "estimated_remaining_hours": len(self.task_context.get("tasks_remaining", [])) * 4,
        }
