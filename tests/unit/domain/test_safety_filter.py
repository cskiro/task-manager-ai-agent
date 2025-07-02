"""Tests for safety filter implementation."""

import pytest
from src.domain.safety.base import SafetyFilter, ValidationResult
from src.domain.safety.basic_filter import BasicSafetyFilter


class TestBasicSafetyFilter:
    """Test basic safety filter functionality."""
    
    @pytest.mark.asyncio
    async def test_prevents_excessive_task_generation(self):
        """Safety filter should cap task count."""
        # Arrange
        filter = BasicSafetyFilter(max_tasks=5)
        output = {
            "tasks": [{"title": f"Task {i}"} for i in range(10)]  # 10 tasks
        }
        
        # Act
        result = await filter.validate_output(output)
        
        # Assert
        assert not result.valid
        assert "Too many tasks" in result.reason
        assert "10" in result.reason
    
    @pytest.mark.asyncio
    async def test_allows_reasonable_task_count(self):
        """Safety filter should allow normal task counts."""
        # Arrange
        filter = BasicSafetyFilter(max_tasks=20)
        output = {
            "tasks": [{"title": f"Task {i}"} for i in range(5)]
        }
        
        # Act
        result = await filter.validate_output(output)
        
        # Assert
        assert result.valid
        assert result.reason is None
    
    @pytest.mark.asyncio
    async def test_prevents_code_injection_attempts(self):
        """Safety filter should block obvious injection attempts."""
        # Arrange
        filter = BasicSafetyFilter()
        dangerous_inputs = [
            "Build a project and exec('rm -rf /')",
            "Create tasks then eval('malicious code')",
            "__import__('os').system('bad command')",
            "Normal project with exec() in the middle"
        ]
        
        # Act & Assert
        for dangerous_input in dangerous_inputs:
            result = await filter.validate_input(dangerous_input)
            assert not result.valid
            assert "code injection" in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_allows_normal_project_descriptions(self):
        """Safety filter should not block legitimate input."""
        # Arrange
        filter = BasicSafetyFilter()
        safe_inputs = [
            "Build an e-commerce platform with payment integration",
            "Create a mobile app for task management", 
            "Develop a web API with authentication",
            "Execute the marketing campaign plan",  # 'execute' is ok
            "Evaluate user feedback and iterate"    # 'evaluate' is ok
        ]
        
        # Act & Assert
        for safe_input in safe_inputs:
            result = await filter.validate_input(safe_input)
            assert result.valid
            assert result.reason is None
    
    @pytest.mark.asyncio
    async def test_tracks_tool_call_limits(self):
        """Safety filter should limit tool calls per session."""
        # Arrange
        filter = BasicSafetyFilter(max_tool_calls=3)
        
        # Act & Assert
        # First 3 calls should succeed
        for i in range(3):
            assert filter.check_tool_call_limit()
        
        # 4th call should fail
        assert not filter.check_tool_call_limit()
    
    @pytest.mark.asyncio
    async def test_empty_input_validation(self):
        """Safety filter should handle empty inputs gracefully."""
        # Arrange
        filter = BasicSafetyFilter()
        
        # Act
        result = await filter.validate_input("")
        
        # Assert
        assert not result.valid
        assert "empty" in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_excessively_long_input(self):
        """Safety filter should cap input length."""
        # Arrange
        filter = BasicSafetyFilter(max_input_length=1000)
        long_input = "a" * 2000
        
        # Act
        result = await filter.validate_input(long_input)
        
        # Assert
        assert not result.valid
        assert "too long" in result.reason.lower()