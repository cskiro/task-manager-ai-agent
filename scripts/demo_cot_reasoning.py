#!/usr/bin/env python3
"""Demo script to showcase Chain of Thought reasoning."""

import asyncio
from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.infrastructure.llm import MockLLMProvider
from rich.console import Console
from rich.panel import Panel

console = Console()


async def demo_cot_reasoning():
    """Demonstrate Chain of Thought reasoning."""
    console.print("\n[bold cyan]🧠 Chain of Thought Reasoning Demo[/bold cyan]\n")
    
    # Create agent with CoT enabled
    llm_provider = MockLLMProvider()
    task_decomposer = TaskDecomposer()
    agent = TaskPlanningAgent(
        name="CoT_Agent",
        llm_provider=llm_provider,
        task_decomposer=task_decomposer,
        enable_cot=True
    )
    
    # Test project
    project = "Build a real-time collaborative document editor with websockets"
    
    console.print(f"[bold]Project:[/bold] {project}\n")
    
    # Run agent
    with console.status("[green]Agent is thinking..."):
        result = await agent.run(project)
    
    # Display reasoning steps
    reasoning_steps = result.get("reasoning_steps", [])
    if reasoning_steps:
        console.print("[bold cyan]Chain of Thought Process:[/bold cyan]\n")
        
        for step in reasoning_steps:
            panel = Panel(
                f"[yellow]Thought:[/yellow] {step['thought']}\n\n"
                f"[green]Conclusion:[/green] {step['conclusion']}",
                title=f"Step {step['step']}",
                border_style="blue"
            )
            console.print(panel)
            console.print()
    
    # Display generated tasks
    tasks = result.get("tasks", [])
    console.print(f"\n[bold green]Generated {len(tasks)} tasks based on reasoning[/bold green]\n")
    
    for i, task in enumerate(tasks, 1):
        console.print(f"{i}. {task.title} {task.priority.emoji}")
        if task.time_estimate:
            console.print(f"   Est: {task.expected_hours:.1f} hours")
        if task.dependencies:
            console.print(f"   Deps: {len(task.dependencies)} dependencies")
        console.print()
    
    # Show execution plan
    execution_plan = result.get("execution_plan", [])
    if execution_plan:
        console.print("\n[bold]Execution Plan:[/bold]")
        for step in execution_plan:
            console.print(f"\nStep {step['step']}:")
            for task in step['tasks']:
                console.print(f"  - {task['title']} {task['priority']}")
            if step.get('can_parallel'):
                console.print("  [dim](Can be done in parallel)[/dim]")


if __name__ == "__main__":
    asyncio.run(demo_cot_reasoning())