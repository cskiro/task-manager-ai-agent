"""Mock LLM provider for testing."""

import json

from pydantic import BaseModel


class MockLLMProvider:
    """
    Mock LLM provider with predictable responses for testing.

    Supports configurable responses for different prompts.
    """

    def __init__(self, responses: dict[str, str] | None = None):
        """
        Initialize with optional predefined responses.

        Args:
            responses: Dict mapping prompt patterns to responses
        """
        self.responses = responses or {}
        self.default_responses = {
            "break down": self._default_task_breakdown,
            "estimate": self._default_time_estimate,
            "dependencies": self._default_dependencies,
        }
        self.call_history: list[dict[str, str]] = []

    async def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Get completion from mock LLM."""
        # Log the call
        self.call_history.append({"prompt": prompt, "system": system or "", "kwargs": str(kwargs)})

        # Check for exact match in configured responses
        if prompt in self.responses:
            return self.responses[prompt]

        # Check for pattern matches in default responses
        prompt_lower = prompt.lower()
        for pattern, response_func in self.default_responses.items():
            if pattern in prompt_lower:
                return response_func(prompt)

        # Default generic response
        return "1. Analyze requirements\n2. Design solution\n3. Implement features\n4. Test thoroughly\n5. Deploy"

    async def structured_complete(
        self, prompt: str, response_model: type[BaseModel], **kwargs
    ) -> BaseModel | dict:
        """Get structured response from mock LLM."""
        # For task planning, return structured task data
        if "break down" in prompt.lower() and "tasks" in prompt.lower():
            return {
                "tasks": [
                    {
                        "title": "Set up project structure",
                        "description": "Initialize the blog API project with folder structure and dependencies",
                        "priority": 3,
                        "time_estimate": {"optimistic": 1.0, "realistic": 2.0, "pessimistic": 3.0}
                    },
                    {
                        "title": "Design database schema",
                        "description": "Create models for posts, comments, and users",
                        "priority": 4,
                        "time_estimate": {"optimistic": 2.0, "realistic": 4.0, "pessimistic": 6.0}
                    },
                    {
                        "title": "Implement authentication",
                        "description": "Add JWT-based authentication for API endpoints",
                        "priority": 5,
                        "time_estimate": {"optimistic": 3.0, "realistic": 5.0, "pessimistic": 8.0}
                    },
                    {
                        "title": "Create CRUD endpoints",
                        "description": "Build RESTful endpoints for blog posts and comments",
                        "priority": 4,
                        "time_estimate": {"optimistic": 4.0, "realistic": 8.0, "pessimistic": 12.0}
                    },
                    {
                        "title": "Add tests and documentation",
                        "description": "Write unit tests and API documentation",
                        "priority": 3,
                        "time_estimate": {"optimistic": 2.0, "realistic": 4.0, "pessimistic": 6.0}
                    }
                ]
            }
        
        # For other cases, try the old approach
        text_response = await self.complete(prompt, **kwargs)
        try:
            data = json.loads(text_response)
            if isinstance(response_model, type) and issubclass(response_model, BaseModel):
                return response_model(**data)
            return data
        except Exception:
            # Return empty dict for dict type, or default instance for BaseModel
            if isinstance(response_model, type) and issubclass(response_model, BaseModel):
                return response_model()
            return {}

    def _default_task_breakdown(self, prompt: str) -> str:
        """Generate task breakdown based on project type."""
        if "web" in prompt.lower() or "app" in prompt.lower():
            return """
1. Set up project structure and dependencies (2-4 hours)
2. Design database schema and models (4-8 hours)
3. Implement backend API endpoints (8-16 hours)
4. Create frontend user interface (8-16 hours)
5. Add authentication and authorization (4-8 hours)
6. Implement data validation and error handling (4-6 hours)
7. Write comprehensive tests (6-10 hours)
8. Set up deployment pipeline (3-5 hours)
"""
        elif "cli" in prompt.lower() or "command" in prompt.lower():
            return """
1. Set up project structure (1-2 hours)
2. Design command interface and arguments (2-4 hours)
3. Implement core functionality (4-8 hours)
4. Add configuration management (2-4 hours)
5. Create help documentation (2-3 hours)
6. Write unit and integration tests (4-6 hours)
"""
        elif "library" in prompt.lower() or "package" in prompt.lower():
            return """
1. Design API and interfaces (2-4 hours)
2. Implement core functionality (6-12 hours)
3. Add comprehensive docstrings (2-4 hours)
4. Create usage examples (2-3 hours)
5. Write extensive test suite (4-8 hours)
6. Set up CI/CD pipeline (2-3 hours)
"""
        else:
            return """
1. Analyze requirements and constraints (2-4 hours)
2. Design system architecture (4-6 hours)
3. Implement core components (8-16 hours)
4. Add integration points (4-8 hours)
5. Create documentation (3-5 hours)
6. Test and validate solution (4-8 hours)
"""

    def _default_time_estimate(self, prompt: str) -> str:
        """Generate time estimates."""
        return """
Based on the complexity, here are the time estimates:
- Optimistic: 20 hours (if everything goes smoothly)
- Realistic: 35 hours (accounting for normal challenges)
- Pessimistic: 50 hours (if significant issues arise)
"""

    def _default_dependencies(self, prompt: str) -> str:
        """Generate task dependencies."""
        return """
Task dependencies:
- Task 2 depends on Task 1 (setup must be complete)
- Tasks 3 and 4 can be done in parallel after Task 2
- Task 5 depends on Tasks 3 and 4
- Task 6 depends on Task 5
- Task 7 can start after Task 3
"""

    def get_call_count(self) -> int:
        """Get number of times the mock was called."""
        return len(self.call_history)

    def get_last_call(self) -> dict[str, str] | None:
        """Get the last call made to the mock."""
        return self.call_history[-1] if self.call_history else None

    def reset(self) -> None:
        """Reset the call history."""
        self.call_history.clear()
