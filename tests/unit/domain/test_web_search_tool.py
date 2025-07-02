"""Tests for web search tool functionality."""

import pytest
from typing import Protocol, runtime_checkable
from src.domain.tools.base import Tool, ToolResult


@runtime_checkable
class WebSearchTool(Protocol):
    """Protocol for web search tools."""
    
    async def search(self, query: str, max_results: int = 5) -> ToolResult:
        """Search the web and return results."""
        ...


class TestWebSearchTool:
    """Test web search tool functionality."""
    
    @pytest.mark.asyncio
    async def test_web_search_tool_exists(self):
        """Test that WebSearchTool can be imported and instantiated."""
        from src.domain.tools.web_search import WebSearchTool
        
        tool = WebSearchTool(name="web_search")
        assert tool.name == "web_search"
        assert hasattr(tool, "search")
    
    @pytest.mark.asyncio
    async def test_web_search_returns_results(self):
        """Test that web search returns structured results."""
        from src.domain.tools.web_search import WebSearchTool
        
        tool = WebSearchTool(name="web_search")
        result = await tool.search("Python task management best practices")
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "results" in result.data
        assert isinstance(result.data["results"], list)
        assert len(result.data["results"]) > 0
    
    @pytest.mark.asyncio
    async def test_web_search_result_structure(self):
        """Test that search results have expected structure."""
        from src.domain.tools.web_search import WebSearchTool
        
        tool = WebSearchTool(name="web_search")
        result = await tool.search("software architecture patterns")
        
        # Check first result structure
        first_result = result.data["results"][0]
        assert "title" in first_result
        assert "url" in first_result
        assert "snippet" in first_result
        assert isinstance(first_result["title"], str)
        assert isinstance(first_result["url"], str)
        assert isinstance(first_result["snippet"], str)
    
    @pytest.mark.asyncio
    async def test_web_search_respects_max_results(self):
        """Test that web search respects max_results parameter."""
        from src.domain.tools.web_search import WebSearchTool
        
        tool = WebSearchTool(name="web_search")
        result = await tool.search("agile project management", max_results=3)
        
        assert len(result.data["results"]) <= 3
    
    @pytest.mark.asyncio
    async def test_web_search_handles_no_results(self):
        """Test that web search handles queries with no results."""
        from src.domain.tools.web_search import WebSearchTool
        
        tool = WebSearchTool(name="web_search")
        # Very specific query unlikely to have results
        result = await tool.search("xyzabc123nonsense987query")
        
        assert result.success is True
        assert result.data["results"] == []
        assert "No results found" in result.message
    
    @pytest.mark.asyncio
    async def test_web_search_error_handling(self):
        """Test that web search handles errors gracefully."""
        from src.domain.tools.web_search import WebSearchTool
        
        # Force an error by using a query that triggers exception in mock
        tool = WebSearchTool(name="web_search", api_key=None)
        # Override search provider to force error
        class ErrorProvider:
            async def search(self, query, max_results):
                raise Exception("Network error")
        
        tool.search_provider = ErrorProvider()
        result = await tool.search("test query")
        
        assert result.success is False
        assert result.error is not None
        assert "Search failed" in result.error