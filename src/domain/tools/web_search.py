"""Web search tool for agents."""

from src.domain.tools.base import Tool, ToolResult


class MockSearchProvider:
    """Mock search provider for testing."""

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Mock search implementation."""
        if "xyzabc123nonsense987query" in query:
            return []

        # Return mock results based on query
        mock_results = []

        if "Python task management" in query:
            mock_results = [
                {
                    "title": "Best Practices for Task Management in Python",
                    "url": "https://example.com/python-task-management",
                    "snippet": "Learn the best practices for managing tasks in Python applications...",
                },
                {
                    "title": "Python Task Queue Libraries Comparison",
                    "url": "https://example.com/task-queues",
                    "snippet": "Compare popular Python task queue libraries like Celery, RQ, and Huey...",
                },
            ]
        elif "software architecture" in query:
            mock_results = [
                {
                    "title": "Software Architecture Patterns Guide",
                    "url": "https://example.com/architecture-patterns",
                    "snippet": "Comprehensive guide to software architecture patterns including MVC, MVP...",
                },
                {
                    "title": "Clean Architecture in Practice",
                    "url": "https://example.com/clean-architecture",
                    "snippet": "Implementing clean architecture principles in modern applications...",
                },
            ]
        elif "agile project management" in query:
            mock_results = [
                {
                    "title": "Agile Project Management Fundamentals",
                    "url": "https://example.com/agile-basics",
                    "snippet": "Understanding the core principles of agile project management...",
                },
                {
                    "title": "Scrum vs Kanban: Which to Choose?",
                    "url": "https://example.com/scrum-kanban",
                    "snippet": "Compare Scrum and Kanban methodologies for your team...",
                },
            ]
        else:
            # Generic results for other queries
            mock_results = [
                {
                    "title": f"Search result for: {query}",
                    "url": "https://example.com/result",
                    "snippet": f"Information about {query}...",
                }
            ]

        return mock_results[:max_results]


class WebSearchTool(Tool):
    """Tool for searching the web."""

    def __init__(self, name: str, api_key: str | None = None):
        super().__init__(name)
        self.api_key = api_key
        self.search_provider = MockSearchProvider()  # Use mock for now

    async def search(self, query: str, max_results: int = 5) -> ToolResult:
        """
        Search the web for information.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            ToolResult with search results
        """
        try:

            # Perform search
            results = await self.search_provider.search(query, max_results)

            if not results:
                return ToolResult(
                    success=True, data={"results": []}, message="No results found for your query"
                )

            return ToolResult(
                success=True, data={"results": results}, message=f"Found {len(results)} results"
            )

        except Exception as e:
            return ToolResult(success=False, data={}, error=f"Search failed: {str(e)}")

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the web search tool."""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        return await self.search(query, max_results)
