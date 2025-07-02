#!/usr/bin/env python3
"""Demo script to showcase safety filter functionality."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.infrastructure.llm import MockLLMProvider
from rich.console import Console
from rich.panel import Panel

console = Console()


async def demo_safety():
    """Demonstrate safety filter capabilities."""
    console.print("\n[bold cyan]🛡️  Safety Filter Demo[/bold cyan]\n")
    
    # Setup
    llm_provider = MockLLMProvider()
    task_decomposer = TaskDecomposer()
    
    # Test cases
    test_cases = [
        {
            "name": "Normal project description",
            "input": "Build an e-commerce platform with payment integration",
            "expected": "success"
        },
        {
            "name": "Code injection attempt",
            "input": "Build a project and exec('malicious code')",
            "expected": "blocked"
        },
        {
            "name": "Empty input",
            "input": "",
            "expected": "blocked"
        },
        {
            "name": "Excessive length",
            "input": "Build " + "a very complex system " * 500,  # Very long input
            "expected": "blocked"
        }
    ]
    
    for test in test_cases:
        console.print(f"\n[bold]Test: {test['name']}[/bold]")
        console.print(f"Input: {test['input'][:50]}{'...' if len(test['input']) > 50 else ''}")
        
        # Create agent with safety enabled
        agent = TaskPlanningAgent(
            name="SafeAgent",
            llm_provider=llm_provider,
            task_decomposer=task_decomposer,
            enable_safety=True
        )
        
        # Run test
        result = await agent.run(test['input'])
        
        # Check result
        if "error" in result:
            console.print(f"[red]❌ Blocked:[/red] {result['error']}")
            status = "blocked"
        else:
            console.print(f"[green]✅ Allowed:[/green] Generated {len(result.get('tasks', []))} tasks")
            status = "success"
        
        # Verify expectation
        if status == test["expected"]:
            console.print("[dim]✓ Expected behavior[/dim]")
        else:
            console.print(f"[bold red]✗ Unexpected! Expected {test['expected']} but got {status}[/bold red]")
    
    # Demo with safety disabled
    console.print("\n[bold]Comparison: Safety Disabled[/bold]")
    
    unsafe_agent = TaskPlanningAgent(
        name="UnsafeAgent",
        llm_provider=llm_provider,
        task_decomposer=task_decomposer,
        enable_safety=False  # Disabled
    )
    
    dangerous_input = "Build a project with eval('test')"
    console.print(f"Input: {dangerous_input}")
    
    result = await unsafe_agent.run(dangerous_input)
    
    if "error" not in result:
        console.print("[yellow]⚠️  Processed without safety checks[/yellow]")
        console.print(f"Generated {len(result.get('tasks', []))} tasks")
    
    # Summary
    panel = Panel(
        "[green]Safety filters protect against:[/green]\n"
        "• Code injection attempts (exec, eval, __import__)\n"
        "• Empty or malformed input\n"
        "• Excessively long input (DoS prevention)\n"
        "• Excessive task generation\n"
        "• Resource exhaustion from too many tool calls",
        title="🛡️ Safety Filter Summary",
        border_style="blue"
    )
    console.print("\n")
    console.print(panel)


if __name__ == "__main__":
    asyncio.run(demo_safety())