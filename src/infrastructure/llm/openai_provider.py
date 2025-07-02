"""OpenAI implementation of LLM provider."""
import os
from typing import Optional, Any
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.application.ports.llm_provider import LLMProvider


class OpenAIProvider:
    """OpenAI implementation of the LLM provider protocol."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 2000
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
    
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
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
            **kwargs
        }
        
        try:
            response = await self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            # Log error in production
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    async def structured_complete(
        self,
        prompt: str,
        response_model: type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """
        Get structured response from OpenAI.
        
        This is a simplified implementation. In production, you might use
        OpenAI's function calling or a library like instructor.
        
        Args:
            prompt: User prompt
            response_model: Pydantic model for response structure
            **kwargs: Additional parameters
        
        Returns:
            Parsed response as Pydantic model
        """
        # Add instructions for structured output
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
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_str = response_text.strip()
            
            data = json.loads(json_str)
            return response_model(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse structured response: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"OpenAIProvider(model={self.model})"