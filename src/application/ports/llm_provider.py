"""Protocol for LLM service providers."""

from typing import Protocol

from pydantic import BaseModel


class LLMProvider(Protocol):
    """Protocol for LLM service providers."""

    async def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Get completion from LLM."""
        ...

    async def structured_complete(
        self, prompt: str, response_model: type[BaseModel], **kwargs
    ) -> BaseModel:
        """Get structured response from LLM."""
        ...
