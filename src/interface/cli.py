"""Command-line interface for task manager AI agent."""
import os
import asyncio
from typing import Optional
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.prompt import Prompt
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
    help="AI-powered task manager for breaking down complex projects",
    rich_markup_mode=None  # Disable Rich formatting to avoid help bug
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


@app.command()
def interactive():
    """Run in interactive mode with a persistent session."""
    asyncio.run(_interactive_mode())


async def _interactive_mode():
    """Interactive mode implementation."""
    # Clear screen and show header
    console.clear()
    _show_header()
    
    # Initialize session
    api_key = os.getenv("OPENAI_API_KEY")
    using_mock = not api_key
    
    if using_mock:
        llm_provider = MockLLMProvider()
        mode_text = "Mode: Mock (No API Key)"
        tip_text = "💡 Tip: Set OPENAI_API_KEY for AI-powered planning"
    else:
        llm_provider = OpenAIProvider(api_key=api_key)
        mode_text = "Mode: OpenAI GPT-4"
        tip_text = "💡 Tip: Type your project description for AI planning"
    
    task_repository = InMemoryTaskRepository()
    use_case = PlanProjectUseCase(llm_provider, task_repository)
    
    # Session history
    history = []
    
    while True:
        # Show mode and tip
        console.print(f"\n🔍 {mode_text}")
        console.print(f"{tip_text}")
        
        # Show available commands
        console.print("\n📋 Available Commands:")
        console.print("  - Type your project description to get a task breakdown")
        console.print("  - 'help' or 'h' - Show this help message")
        console.print("  - 'history' - Show previous projects")
        console.print("  - 'clear' - Clear the screen")
        console.print("  - 'quit' or 'exit' - Exit the program")
        
        # Get user input
        console.print()
        user_input = Prompt.ask("[red]⬤[/red] Enter project (or command) >")
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        
        elif user_input.lower() in ['help', 'h']:
            console.clear()
            _show_header()
            continue
        
        elif user_input.lower() == 'clear':
            console.clear()
            _show_header()
            continue
        
        elif user_input.lower() == 'history':
            if not history:
                console.print("\n[dim]No projects planned yet.[/dim]")
            else:
                console.print("\n[bold]Previous Projects:[/bold]")
                for i, project in enumerate(history, 1):
                    console.print(f"{i}. {project['description']} ({len(project['tasks'])} tasks)")
            continue
        
        elif user_input.strip():
            # Plan the project
            console.print()
            with console.status("[bold green]AI is analyzing your project..."):
                request = PlanProjectRequest(
                    project_id=uuid4(),
                    description=user_input,
                    context=None
                )
                try:
                    response = await use_case.execute(request)
                    
                    # Add to history
                    history.append({
                        'description': user_input,
                        'tasks': response.tasks
                    })
                    
                    # Display results
                    console.print(f"\n[bold green]✓[/bold green] {response.summary}\n")
                    _display_tasks_table(response.tasks)
                    
                    # Show total time
                    total_hours = sum(task.expected_hours for task in response.tasks)
                    console.print(f"\n[bold]Total estimated time:[/bold] {total_hours:.1f} hours")
                    
                except Exception as e:
                    console.print(f"\n[red]Error: {str(e)}[/red]")
        
        # Add spacing before next prompt
        console.print()


def _show_header():
    """Display the application header."""
    header_text = "🤖 Task Manager AI Agent 🤖\n\nBreak down complex projects into manageable tasks with AI!"
    
    panel = Panel(
        header_text,
        style="cyan",
        border_style="bright_blue",
        padding=(1, 2),
        title_align="center"
    )
    console.print(panel)


def _display_tasks_table(tasks):
    """Display tasks in a formatted table."""
    table = Table(show_lines=True)
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Task", style="white")
    table.add_column("Priority", style="yellow", width=8)
    table.add_column("Est. Hours", style="green", width=10)
    table.add_column("Dependencies", style="dim")
    
    task_map = {task.id: task for task in tasks}
    
    for i, task in enumerate(tasks, 1):
        # Format dependencies
        deps = []
        for dep_id in task.dependencies:
            dep_task = task_map.get(dep_id)
            if dep_task:
                dep_idx = tasks.index(dep_task) + 1
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


if __name__ == "__main__":
    import sys
    
    # If no arguments provided, run interactive mode
    if len(sys.argv) == 1:
        asyncio.run(_interactive_mode())
    else:
        app()