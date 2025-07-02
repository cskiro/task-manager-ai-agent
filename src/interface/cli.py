"""Command-line interface for task manager AI agent."""
import os
import asyncio
from typing import Optional
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from dotenv import load_dotenv

from src.application.use_cases.plan_project import (
    PlanProjectUseCase,
    PlanProjectRequest
)
from src.infrastructure.llm import OpenAIProvider, MockLLMProvider

# Load environment variables
load_dotenv()

# Initialize Typer app and Rich console
app = typer.Typer(
    name="taskman",
    help="AI-powered task manager for breaking down complex projects"
)
console = Console()


class InMemoryTaskRepository:
    """Simple in-memory task storage for CLI."""
    def __init__(self):
        self.tasks = {}
    
    async def save(self, task):
        self.tasks[task.id] = task


@app.command()
def plan(
    description: str = typer.Argument(..., help="Project description"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
    mock: bool = typer.Option(False, "--mock", help="Use mock LLM for testing"),
    model: str = typer.Option("gpt-4-turbo-preview", "--model", "-m", help="OpenAI model to use")
):
    """Plan a project by breaking it down into tasks using AI."""
    asyncio.run(_plan_project(description, context, mock, model))


async def _plan_project(description: str, context: Optional[str], mock: bool, model: str):
    """Async implementation of project planning."""
    # Show what we're doing
    console.print(f"\n[bold blue]Planning project:[/bold blue] {description}")
    if context:
        console.print(f"[dim]Context: {context}[/dim]")
    
    # Initialize dependencies
    if mock:
        console.print("[yellow]Using mock LLM provider[/yellow]")
        llm_provider = MockLLMProvider()
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY not found in environment[/red]")
            console.print("Please set your OpenAI API key in .env file or environment")
            raise typer.Exit(1)
        
        llm_provider = OpenAIProvider(api_key=api_key, model=model)
        console.print(f"[green]Using OpenAI {model}[/green]")
    
    task_repository = InMemoryTaskRepository()
    use_case = PlanProjectUseCase(llm_provider, task_repository)
    
    # Execute planning
    with console.status("[bold green]AI is thinking..."):
        request = PlanProjectRequest(
            project_id=uuid4(),
            description=description,
            context=context
        )
        response = await use_case.execute(request)
    
    # Display results
    console.print(f"\n[bold green]✓[/bold green] {response.summary}\n")
    
    # Create task table
    table = Table(title="Project Tasks", show_lines=True)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Task", style="white")
    table.add_column("Priority", style="yellow", width=8)
    table.add_column("Est. Hours", style="green", width=10)
    table.add_column("Dependencies", style="dim")
    
    # Build dependency map for display
    task_map = {task.id: task for task in response.tasks}
    
    for i, task in enumerate(response.tasks, 1):
        # Format dependencies
        deps = []
        for dep_id in task.dependencies:
            dep_task = task_map.get(dep_id)
            if dep_task:
                dep_idx = response.tasks.index(dep_task) + 1
                deps.append(f"#{dep_idx}")
        
        # Format time estimate
        time_est = f"{task.expected_hours:.1f}h" if task.time_estimate else "-"
        
        table.add_row(
            str(i),
            task.title,
            task.priority.emoji,
            time_est,
            ", ".join(deps) if deps else "-"
        )
    
    console.print(table)
    
    # Show task hierarchy if there are dependencies
    if any(task.dependencies for task in response.tasks):
        tree = Tree("[bold]Task Dependencies[/bold]")
        
        # Find root tasks (no dependencies)
        root_tasks = [t for t in response.tasks if not t.dependencies]
        
        for root in root_tasks:
            _add_task_to_tree(tree, root, response.tasks, task_map)
        
        console.print("\n")
        console.print(tree)
    
    # Calculate total time
    total_hours = sum(task.expected_hours for task in response.tasks)
    console.print(f"\n[bold]Total estimated time:[/bold] {total_hours:.1f} hours")


def _add_task_to_tree(parent_node, task, all_tasks, task_map, processed=None):
    """Recursively add tasks to tree based on dependencies."""
    if processed is None:
        processed = set()
    
    if task.id in processed:
        return
    
    processed.add(task.id)
    
    # Find task index for display
    task_idx = all_tasks.index(task) + 1
    time_str = f" ({task.expected_hours:.1f}h)" if task.time_estimate else ""
    node = parent_node.add(f"#{task_idx}: {task.title}{time_str}")
    
    # Find tasks that depend on this one
    dependent_tasks = [t for t in all_tasks if task.id in t.dependencies]
    
    for dep_task in dependent_tasks:
        _add_task_to_tree(node, dep_task, all_tasks, task_map, processed)


@app.command()
def version():
    """Show version information."""
    console.print("[bold]Task Manager AI Agent[/bold] v0.1.0")
    console.print("An intelligent AI agent for project planning")


if __name__ == "__main__":
    app()