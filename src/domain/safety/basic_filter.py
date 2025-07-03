"""Basic safety filter implementation."""

from datetime import datetime
from typing import Any

from src.domain.safety.base import ValidationResult
from src.domain.safety.pii_detector import PIIDetector


class BasicSafetyFilter:
    """Basic safety checks for task manager agent."""

    def __init__(self, max_tasks: int = 20, max_tool_calls: int = 10, max_input_length: int = 5000):
        self.max_tasks = max_tasks
        self.max_tool_calls = max_tool_calls
        self.max_input_length = max_input_length
        self.tool_call_count = 0
        self.pii_detector = PIIDetector()
        self.audit_trail: list[dict[str, Any]] = []

    async def validate_input(self, text: str) -> ValidationResult:
        """
        Check if input is safe to process.

        Validates against:
        - Empty input
        - Excessive length
        - Code injection attempts
        - PII (Personally Identifiable Information)
        """
        # Check for empty input
        if not text or not text.strip():
            result = ValidationResult(valid=False, reason="Input cannot be empty")
            self._record_refusal("input", result.reason)
            return result

        # Check length
        if len(text) > self.max_input_length:
            result = ValidationResult(
                valid=False,
                reason=f"Input too long: {len(text)} characters (max: {self.max_input_length})",
            )
            self._record_refusal("input", result.reason)
            return result

        # Check for code injection patterns
        dangerous_patterns = ["exec(", "eval(", "__import__"]
        text_lower = text.lower()

        for pattern in dangerous_patterns:
            if pattern in text_lower:
                result = ValidationResult(valid=False, reason="Potential code injection detected")
                self._record_refusal("input", result.reason)
                return result

        # Check for PII
        if self.pii_detector.contains_pii(text):
            pii_types = self.pii_detector.find_pii_types(text)
            result = ValidationResult(
                valid=False,
                reason=f"Contains PII: {', '.join(pii_types)}"
            )
            self._record_refusal("input", result.reason)
            return result

        return ValidationResult(valid=True)

    async def validate_output(self, content: dict[str, Any]) -> ValidationResult:
        """
        Check if output is safe to return.

        Validates against:
        - Excessive task generation
        - PII in task descriptions
        """
        # Check task count
        tasks = content.get("tasks", [])
        if len(tasks) > self.max_tasks:
            result = ValidationResult(
                valid=False, reason=f"Too many tasks: {len(tasks)} (max: {self.max_tasks})"
            )
            self._record_refusal("output", result.reason)
            return result

        # Check for PII in task titles/descriptions
        for task in tasks:
            # Handle both Task objects and dictionaries
            if hasattr(task, 'title'):
                # Task object
                task_text = task.title + " " + task.description
            else:
                # Dictionary
                task_text = task.get("title", "") + " " + task.get("description", "")
            
            if self.pii_detector.contains_pii(task_text):
                pii_types = self.pii_detector.find_pii_types(task_text)
                result = ValidationResult(
                    valid=False,
                    reason=f"Contains PII: {', '.join(pii_types)}"
                )
                self._record_refusal("output", result.reason)
                return result

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

    def _record_refusal(self, validation_type: str, reason: str) -> None:
        """
        Record a refused request in the audit trail.
        
        Args:
            validation_type: Type of validation (input/output)
            reason: Reason for refusal
        """
        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "type": validation_type,
            "reason": reason
        })

    def get_audit_trail(self) -> list[dict[str, Any]]:
        """
        Get the audit trail of refused requests.
        
        Returns:
            List of audit trail entries
        """
        return self.audit_trail

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []
