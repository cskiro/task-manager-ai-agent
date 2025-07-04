"""OpenAI implementation of LLM provider."""

import os

from openai import AsyncOpenAI
from pydantic import BaseModel


class OpenAIProvider:
    """OpenAI implementation of the LLM provider protocol."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "o3",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use for completions
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """
        Get completion from OpenAI.

        Args:
            prompt: User prompt
            system: System message (optional)
            **kwargs: Additional parameters passed to OpenAI

        Returns:
            Completion text
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        # Merge kwargs with defaults
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

        try:
            response = await self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            # Provide helpful error messages for common issues
            if "model_not_found" in error_msg or "does not exist" in error_msg:
                model_hint = f"Model '{self.model}' is not available. "
                if self.model == "o3":
                    model_hint += "o3 is a newer model that may require special access. Try using --model gpt-3.5-turbo or --model gpt-4 instead."
                else:
                    model_hint += "Try using --model gpt-3.5-turbo which is widely available."
                raise RuntimeError(f"OpenAI API error: {model_hint}") from e
            elif "insufficient_quota" in error_msg or "exceeded your current quota" in error_msg:
                raise RuntimeError(
                    f"OpenAI API quota exceeded. Please check your billing at https://platform.openai.com/account/billing\n"
                    f"You can use --mock flag to test without API calls."
                ) from e
            # Log error in production
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e

    async def structured_complete(
        self, prompt: str, response_model: type[BaseModel] | type[dict], **kwargs
    ) -> BaseModel | dict:
        """
        Get structured response from OpenAI.

        This is a simplified implementation. In production, you might use
        OpenAI's function calling or a library like instructor.

        Args:
            prompt: User prompt
            response_model: Pydantic model or dict for response structure
            **kwargs: Additional parameters

        Returns:
            Parsed response as Pydantic model or dict
        """
        # Add instructions for structured output
        if response_model is dict:
            # Handle dict type specially
            structured_prompt = f"""{prompt}

Please respond with valid JSON.

Respond ONLY with the JSON, no additional text."""
        else:
            # Handle Pydantic models
            structured_prompt = f"""{prompt}

Please respond with valid JSON that matches this structure:
{response_model.model_json_schema()}

Respond ONLY with the JSON, no additional text."""

        response_text = await self.complete(structured_prompt, **kwargs)

        # Try to parse the response
        try:
            # Simple extraction of JSON from response
            import json
            import re

            # Find JSON in response (between ```json and ``` or just raw JSON)
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_str = response_text.strip()

            data = json.loads(json_str)
            
            # Return based on response model type
            if response_model is dict:
                return data
            else:
                return response_model(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse structured response: {str(e)}") from e

    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIProvider(model={self.model})"
