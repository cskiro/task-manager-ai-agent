"""Service for managing agent checkpoints."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.domain.entities.checkpoint import AgentCheckpoint


class CheckpointService:
    """Handles checkpoint persistence and recovery."""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None, max_checkpoints: int = 5):
        """Initialize checkpoint service.
        
        Args:
            checkpoint_dir: Directory to store checkpoints (defaults to ~/.taskman/checkpoints)
            max_checkpoints: Maximum number of checkpoints to keep per agent
        """
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = Path.home() / ".taskman" / "checkpoints"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
    
    async def save_checkpoint(self, checkpoint: AgentCheckpoint, filepath: Optional[Path] = None) -> Path:
        """Save checkpoint to file.
        
        Args:
            checkpoint: The checkpoint to save
            filepath: Optional specific filepath, otherwise auto-generated
            
        Returns:
            Path to the saved checkpoint file
        """
        if filepath is None:
            # Generate filename with timestamp including microseconds
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{checkpoint.agent_name}_{timestamp}.json"
            filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        with open(filepath, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        
        # Rotate old checkpoints if needed
        if filepath.parent == self.checkpoint_dir:
            await self._rotate_checkpoints(checkpoint.agent_name)
        
        return filepath
    
    async def load_checkpoint(self, filepath: Path) -> AgentCheckpoint:
        """Load checkpoint from file.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Loaded checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        with open(filepath) as f:
            data = json.load(f)
        
        return AgentCheckpoint.from_dict(data)
    
    async def get_latest_checkpoint(self, agent_name: str) -> Optional[Path]:
        """Get the most recent checkpoint for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        pattern = f"{agent_name}_*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        # Sort by modification time (most recent first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    async def _rotate_checkpoints(self, agent_name: str) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit.
        
        Args:
            agent_name: Name of the agent
        """
        pattern = f"{agent_name}_*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest checkpoints
        num_to_remove = len(checkpoints) - self.max_checkpoints
        for checkpoint in checkpoints[:num_to_remove]:
            checkpoint.unlink()
    
    def create_checkpoint_from_agent(self, agent: Any) -> AgentCheckpoint:
        """Create a checkpoint from current agent state.
        
        Args:
            agent: The agent to checkpoint
            
        Returns:
            AgentCheckpoint with current agent state
        """
        # Extract memory data
        memory_data = {
            "short_term": agent.memory.short_term
        }
        
        # Extract task context if available
        task_context = getattr(agent, "task_context", None)
        
        # Extract reasoning traces if available
        reasoning_traces = getattr(agent, "reasoning_traces", [])
        
        return AgentCheckpoint(
            agent_type=agent.__class__.__name__,
            agent_name=agent.name,
            state=agent.state.value,
            memory=memory_data,
            task_context=task_context,
            reasoning_traces=reasoning_traces,
        )