#!/usr/bin/env python3
"""Demo script to showcase calculator tool for project estimations."""

import asyncio
from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.domain.tools.calculator import CalculatorTool
from src.infrastructure.llm import MockLLMProvider
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def demo_calculator():
    """Demonstrate calculator tool for project estimations."""
    console.print("\n[bold cyan]🧮 Calculator Tool Demo - Project Estimation[/bold cyan]\n")
    
    # First, show standalone calculator functionality
    console.print("[bold]1. Standalone Calculator Examples:[/bold]\n")
    
    calculator = CalculatorTool(name="calculator")
    
    examples = [
        ("Basic math", "2 + 2 * 3"),
        ("Complex expression", "(10 + 5) * 3 - 20 / 4"),
        ("Math functions", "sqrt(16) + pow(2, 3)"),
        ("Trigonometry", "sin(pi/2) + cos(0)"),
        ("Time estimation", "8 * 5 * 4"),  # 8 hours/day * 5 days/week * 4 weeks
    ]
    
    for desc, expr in examples:
        result = await calculator.calculate(expr)
        if result.success:
            console.print(f"  {desc}: {expr} = [green]{result.data['result']}[/green]")
        else:
            console.print(f"  {desc}: {expr} = [red]Error: {result.error}[/red]")
    
    console.print("\n[bold]2. Agent with Calculator for Project Estimation:[/bold]\n")
    
    # Create agent with estimation capabilities
    llm_provider = MockLLMProvider()
    task_decomposer = TaskDecomposer()
    agent = TaskPlanningAgent(
        name="EstimationAgent",
        llm_provider=llm_provider,
        task_decomposer=task_decomposer,
        enable_cot=True,
        enable_tools=True,
        enable_estimation=True
    )
    
    # Register calculator
    agent.register_tool(calculator)
    
    # Test project
    project = "Build an e-commerce platform with payment integration"
    
    console.print(f"[bold]Project:[/bold] {project}\n")
    
    # Run agent
    with console.status("[green]Agent is calculating project estimations..."):
        result = await agent.run(project)
    
    # Display tasks
    tasks = result.get("tasks", [])
    console.print(f"[bold green]Generated {len(tasks)} tasks[/bold green]\n")
    
    # Create task table with hours
    table = Table(title="Project Tasks with Time Estimates")
    table.add_column("Task", style="cyan")
    table.add_column("Priority", style="yellow")
    table.add_column("Hours", style="green")
    
    for task in tasks:
        table.add_row(
            task.title,
            task.priority.emoji,
            f"{task.expected_hours:.1f}h"
        )
    
    console.print(table)
    
    # Display estimations
    estimations = result.get("estimations", {})
    if estimations:
        console.print("\n[bold]Project Estimations:[/bold]\n")
        
        est_panel = Panel(
            f"[yellow]Total Hours:[/yellow] {estimations.get('total_hours', 0):.1f}h\n"
            f"[yellow]With 20% Buffer:[/yellow] {estimations.get('hours_with_buffer', 0):.1f}h\n"
            f"[yellow]Estimated Cost:[/yellow] {estimations.get('cost_calculation', 'N/A')}\n"
            f"[yellow]Time Saved (Parallel):[/yellow] {estimations.get('parallel_time_saved', 0):.1f}h",
            title="📊 Calculations",
            border_style="blue"
        )
        console.print(est_panel)
        
        # Show execution plan with parallel tasks
        execution_plan = result.get("execution_plan", [])
        if execution_plan:
            console.print("\n[bold]Execution Timeline:[/bold]\n")
            
            for step in execution_plan:
                step_num = step['step']
                can_parallel = step.get('can_parallel', False)
                tasks_in_step = step['tasks']
                
                if can_parallel and len(tasks_in_step) > 1:
                    console.print(f"Step {step_num}: [green]Parallel Execution[/green]")
                    max_time = max(t['estimated_hours'] for t in tasks_in_step)
                    console.print(f"  Duration: {max_time:.1f}h (parallel)")
                else:
                    console.print(f"Step {step_num}: Sequential Execution")
                    total_time = sum(t['estimated_hours'] for t in tasks_in_step)
                    console.print(f"  Duration: {total_time:.1f}h")
                
                for task in tasks_in_step:
                    console.print(f"  - {task['title']} ({task['estimated_hours']:.1f}h)")
                console.print()
    
    # Show calculation details
    console.print("\n[bold]Calculation Benefits:[/bold]")
    console.print("✓ Accurate project timeline based on task dependencies")
    console.print("✓ Cost estimation for budget planning")
    console.print("✓ Identified parallel execution opportunities")
    console.print("✓ Applied industry-standard 20% buffer")


if __name__ == "__main__":
    asyncio.run(demo_calculator())