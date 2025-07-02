"""Tests for calculator tool functionality."""

import math

import pytest

from src.domain.tools.base import ToolResult


class TestCalculatorTool:
    """Test calculator tool functionality."""

    @pytest.mark.asyncio
    async def test_calculator_tool_exists(self):
        """Test that CalculatorTool can be imported and instantiated."""
        from src.domain.tools.calculator import CalculatorTool

        tool = CalculatorTool(name="calculator")
        assert tool.name == "calculator"
        assert hasattr(tool, "calculate")

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        from src.domain.tools.calculator import CalculatorTool

        tool = CalculatorTool(name="calculator")

        # Addition
        result = await tool.calculate("2 + 2")
        assert result.success is True
        assert result.data["result"] == 4.0

        # Subtraction
        result = await tool.calculate("10 - 3")
        assert result.data["result"] == 7.0

        # Multiplication
        result = await tool.calculate("5 * 6")
        assert result.data["result"] == 30.0

        # Division
        result = await tool.calculate("15 / 3")
        assert result.data["result"] == 5.0

    @pytest.mark.asyncio
    async def test_complex_expressions(self):
        """Test complex mathematical expressions."""
        from src.domain.tools.calculator import CalculatorTool

        tool = CalculatorTool(name="calculator")

        # Order of operations
        result = await tool.calculate("2 + 3 * 4")
        assert result.data["result"] == 14.0

        # Parentheses
        result = await tool.calculate("(2 + 3) * 4")
        assert result.data["result"] == 20.0

        # Decimals
        result = await tool.calculate("3.14 * 2")
        assert result.data["result"] == 6.28

    @pytest.mark.asyncio
    async def test_math_functions(self):
        """Test math module functions."""
        from src.domain.tools.calculator import CalculatorTool

        tool = CalculatorTool(name="calculator")

        # Square root
        result = await tool.calculate("sqrt(16)")
        assert result.data["result"] == 4.0

        # Power
        result = await tool.calculate("pow(2, 3)")
        assert result.data["result"] == 8.0

        # Trigonometry
        result = await tool.calculate("sin(0)")
        assert result.data["result"] == 0.0

        # Constants
        result = await tool.calculate("pi * 2")
        assert abs(result.data["result"] - (math.pi * 2)) < 0.0001

    @pytest.mark.asyncio
    async def test_invalid_expressions(self):
        """Test handling of invalid expressions."""
        from src.domain.tools.calculator import CalculatorTool

        tool = CalculatorTool(name="calculator")

        # Syntax error
        result = await tool.calculate("2 +* 2")
        assert result.success is False
        assert "error" in result.error.lower()

        # Division by zero
        result = await tool.calculate("1 / 0")
        assert result.success is False
        assert "zero" in result.error.lower()

        # Invalid function
        result = await tool.calculate("invalid_func(5)")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_security_restrictions(self):
        """Test that dangerous operations are blocked."""
        from src.domain.tools.calculator import CalculatorTool

        tool = CalculatorTool(name="calculator")

        # No imports allowed
        result = await tool.calculate("__import__('os')")
        assert result.success is False
        assert "not allowed" in result.error.lower() or "error" in result.error.lower()

        # No builtin functions except math
        result = await tool.calculate("eval('2+2')")
        assert result.success is False

        # No attribute access
        result = await tool.calculate("().__class__")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_method(self):
        """Test the execute method follows Tool interface."""
        from src.domain.tools.calculator import CalculatorTool

        tool = CalculatorTool(name="calculator")

        result = await tool.execute(expression="5 + 5")
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.data["result"] == 10.0
