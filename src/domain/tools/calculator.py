"""Calculator tool for mathematical operations."""

import math
import ast
import operator
from typing import Any, Dict
from src.domain.tools.base import Tool, ToolResult


class CalculatorTool(Tool):
    """Tool for performing mathematical calculations safely."""
    
    def __init__(self, name: str):
        super().__init__(name)
        # Safe math functions to expose
        self.allowed_names = {
            # Math constants
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': math.inf,
            'nan': math.nan,
            
            # Basic functions
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            
            # Math module functions
            'sqrt': math.sqrt,
            'pow': math.pow,
            'exp': math.exp,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            
            # Trigonometric functions
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,
            
            # Hyperbolic functions
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            
            # Other math functions
            'ceil': math.ceil,
            'floor': math.floor,
            'factorial': math.factorial,
            'gcd': math.gcd,
            'degrees': math.degrees,
            'radians': math.radians,
        }
    
    async def calculate(self, expression: str) -> ToolResult:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            ToolResult with the calculation result or error
        """
        try:
            # Remove whitespace for cleaner parsing
            expression = expression.strip()
            
            # Check for empty expression
            if not expression:
                return ToolResult(
                    success=False,
                    data={},
                    error="Empty expression provided"
                )
            
            # Parse the expression into an AST
            try:
                tree = ast.parse(expression, mode='eval')
            except SyntaxError as e:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Syntax error in expression: {str(e)}"
                )
            
            # Validate the AST for safety
            if not self._is_safe_ast(tree):
                return ToolResult(
                    success=False,
                    data={},
                    error="Expression contains operations that are not allowed"
                )
            
            # Evaluate the expression with restricted namespace
            try:
                result = eval(
                    compile(tree, '<calculator>', 'eval'),
                    {"__builtins__": {}},  # No builtins
                    self.allowed_names      # Only allowed functions
                )
            except ZeroDivisionError:
                return ToolResult(
                    success=False,
                    data={},
                    error="Division by zero error"
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Calculation error: {str(e)}"
                )
            
            # Convert result to float for consistency
            try:
                result = float(result)
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Result cannot be converted to number: {result}"
                )
            
            return ToolResult(
                success=True,
                data={
                    "result": result,
                    "expression": expression
                },
                message=f"Calculated: {expression} = {result}"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                error=f"Unexpected error: {str(e)}"
            )
    
    def _is_safe_ast(self, tree: ast.AST) -> bool:
        """
        Validate that the AST only contains safe operations.
        
        Args:
            tree: AST to validate
            
        Returns:
            True if safe, False otherwise
        """
        # Allowed node types for mathematical expressions
        allowed_nodes = {
            ast.Module,
            ast.Expr,
            ast.Expression,
            ast.Load,
            ast.Store,
            ast.Del,
            
            # Literals
            ast.Constant,  # For all literal values in Python 3.8+
            
            # Operations
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.BoolOp,
            
            # Operators
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.LShift,
            ast.RShift,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.MatMult,
            
            # Unary operators
            ast.UAdd,
            ast.USub,
            ast.Not,
            ast.Invert,
            
            # Comparison operators
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
            
            # Boolean operators
            ast.And,
            ast.Or,
            
            # Function calls (for math functions)
            ast.Call,
            ast.Name,
            ast.Attribute,
            
            # Other safe nodes
            ast.IfExp,  # Ternary operator
            ast.Tuple,
            ast.List,
        }
        
        for node in ast.walk(tree):
            # Check if node type is allowed
            if type(node) not in allowed_nodes:
                return False
            
            # Additional checks for specific node types
            if isinstance(node, ast.Name):
                # Only allow names that are in our allowed list
                if node.id not in self.allowed_names:
                    return False
            
            elif isinstance(node, ast.Attribute):
                # No attribute access allowed (prevents things like ().__class__)
                return False
            
            elif isinstance(node, ast.Call):
                # Only allow calls to our allowed functions
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self.allowed_names:
                        return False
                else:
                    # No other types of calls allowed
                    return False
        
        return True
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the calculator tool."""
        expression = kwargs.get("expression", "")
        return await self.calculate(expression)