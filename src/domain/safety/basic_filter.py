"""Basic safety filter implementation."""

from typing import Dict, Any
from src.domain.safety.base import ValidationResult


class BasicSafetyFilter:
    """Basic safety checks for task manager agent."""
    
    def __init__(
        self,
        max_tasks: int = 20,
        max_tool_calls: int = 10,
        max_input_length: int = 5000
    ):
        self.max_tasks = max_tasks
        self.max_tool_calls = max_tool_calls
        self.max_input_length = max_input_length
        self.tool_call_count = 0
    
    async def validate_input(self, text: str) -> ValidationResult:
        """
        Check if input is safe to process.
        
        Validates against:
        - Empty input
        - Excessive length
        - Code injection attempts
        """
        # Check for empty input
        if not text or not text.strip():
            return ValidationResult(
                valid=False,
                reason="Input cannot be empty"
            )
        
        # Check length
        if len(text) > self.max_input_length:
            return ValidationResult(
                valid=False,
                reason=f"Input too long: {len(text)} characters (max: {self.max_input_length})"
            )
        
        # Check for code injection patterns
        dangerous_patterns = ['exec(', 'eval(', '__import__']
        text_lower = text.lower()
        
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                return ValidationResult(
                    valid=False,
                    reason="Potential code injection detected"
                )
        
        return ValidationResult(valid=True)
    
    async def validate_output(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Check if output is safe to return.
        
        Validates against:
        - Excessive task generation
        """
        # Check task count
        tasks = content.get("tasks", [])
        if len(tasks) > self.max_tasks:
            return ValidationResult(
                valid=False,
                reason=f"Too many tasks: {len(tasks)} (max: {self.max_tasks})"
            )
        
        return ValidationResult(valid=True)
    
    def check_tool_call_limit(self) -> bool:
        """
        Check if tool call limit has been reached.
        
        Returns:
            True if under limit, False if limit reached
        """
        if self.tool_call_count < self.max_tool_calls:
            self.tool_call_count += 1
            return True
        return False
    
    def reset_tool_call_count(self) -> None:
        """Reset the tool call counter for a new session."""
        self.tool_call_count = 0