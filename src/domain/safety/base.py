"""Base protocol and types for safety filters."""

from typing import Protocol, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a safety validation check."""
    valid: bool
    reason: Optional[str] = None


class SafetyFilter(Protocol):
    """Protocol for safety validation."""
    
    async def validate_input(self, text: str) -> ValidationResult:
        """
        Check if input is safe to process.
        
        Args:
            text: Input text to validate
            
        Returns:
            ValidationResult indicating if input is safe
        """
        ...
    
    async def validate_output(self, content: Dict[str, Any]) -> ValidationResult:
        """
        Check if output is safe to return.
        
        Args:
            content: Output content to validate
            
        Returns:
            ValidationResult indicating if output is safe
        """
        ...