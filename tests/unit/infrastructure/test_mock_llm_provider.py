"""Unit tests for MockLLMProvider."""
import pytest

from src.infrastructure.llm import MockLLMProvider


class TestMockLLMProvider:
    """Test cases for MockLLMProvider."""
    
    @pytest.mark.asyncio
    async def test_default_response(self):
        """Test default response when no pattern matches."""
        provider = MockLLMProvider()
        
        response = await provider.complete("Random prompt")
        
        assert "Analyze requirements" in response
        assert "Deploy" in response
    
    @pytest.mark.asyncio
    async def test_configured_response(self):
        """Test using configured responses."""
        responses = {
            "What is 2+2?": "4",
            "Hello": "Hi there!"
        }
        provider = MockLLMProvider(responses=responses)
        
        assert await provider.complete("What is 2+2?") == "4"
        assert await provider.complete("Hello") == "Hi there!"
    
    @pytest.mark.asyncio
    async def test_web_project_breakdown(self):
        """Test task breakdown for web projects."""
        provider = MockLLMProvider()
        
        response = await provider.complete("Break down a web application project")
        
        assert "database schema" in response
        assert "API endpoints" in response
        assert "frontend" in response
        assert "authentication" in response
    
    @pytest.mark.asyncio
    async def test_cli_project_breakdown(self):
        """Test task breakdown for CLI projects."""
        provider = MockLLMProvider()
        
        response = await provider.complete("Break down a CLI tool project")
        
        assert "command interface" in response
        assert "core functionality" in response
        assert "configuration management" in response
    
    @pytest.mark.asyncio
    async def test_call_history(self):
        """Test that call history is tracked."""
        provider = MockLLMProvider()
        
        await provider.complete("First prompt", system="System message")
        await provider.complete("Second prompt")
        
        assert provider.get_call_count() == 2
        
        last_call = provider.get_last_call()
        assert last_call["prompt"] == "Second prompt"
        assert last_call["system"] == ""
        
        first_call = provider.call_history[0]
        assert first_call["prompt"] == "First prompt"
        assert first_call["system"] == "System message"
    
    @pytest.mark.asyncio
    async def test_reset_history(self):
        """Test resetting call history."""
        provider = MockLLMProvider()
        
        await provider.complete("Test prompt")
        assert provider.get_call_count() == 1
        
        provider.reset()
        assert provider.get_call_count() == 0
        assert provider.get_last_call() is None
    
    @pytest.mark.asyncio
    async def test_time_estimates_in_breakdown(self):
        """Test that time estimates are included in breakdowns."""
        provider = MockLLMProvider()
        
        response = await provider.complete("Break down a web app")
        
        assert "hours)" in response
        assert "2-4 hours" in response or "4-8 hours" in response