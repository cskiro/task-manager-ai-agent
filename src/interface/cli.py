"""Command-line interface for task manager AI agent."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.tree import Tree
import rich.box

from src.application.use_cases.plan_project import PlanProjectRequest, PlanProjectUseCase
from src.application.use_cases.streaming_runner import StreamingPlanRunner
from src.application.models import UpdateType, PlanningUpdate
from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.infrastructure.llm import MockLLMProvider, OpenAIProvider
from src.infrastructure.persistence.json_task_repository import JSONTaskRepository

# Load environment variables
load_dotenv()

# Initialize Typer app and Rich console
app = typer.Typer(
    name="taskman",
    help="AI-powered task manager for breaking down complex projects",
    rich_markup_mode=None,  # Disable Rich formatting to avoid Typer 0.12.5 bug
)
console = Console()

# Global state for current project (simple solution for MVP)
_current_tasks = []

# Example projects for quick demonstration
EXAMPLE_PROJECTS = {
    "1": {
        "name": "🛍️  E-commerce Platform",
        "description": "Build a modern e-commerce platform with React frontend, Node.js API, PostgreSQL database, product catalog, shopping cart, payment processing with Stripe, and AWS deployment",
        "keywords": ["ecommerce", "shop", "store", "marketplace"]
    },
    "2": {
        "name": "📱  Mobile Fitness App", 
        "description": "Create a fitness tracking mobile app with React Native, exercise database, workout plans, progress tracking, social features, and premium subscription model",
        "keywords": ["fitness", "health", "mobile", "workout"]
    },
    "3": {
        "name": "🤖  Discord Bot",
        "description": "Develop a Discord bot with Python, moderation features, custom commands, role management, music playback, and MongoDB for persistent data storage",
        "keywords": ["discord", "bot", "automation", "chat"]
    },
    "4": {
        "name": "🎮  Indie Game MVP",
        "description": "Create a 2D indie game prototype with Unity, player mechanics, level design system, enemy AI, inventory system, and Steam integration preparation",
        "keywords": ["game", "unity", "godot", "gaming"]
    }
}


class InMemoryTaskRepository:
    """Simple in-memory task storage for CLI."""

    def __init__(self):
        self.tasks = {}

    async def save(self, task):
        self.tasks[task.id] = task


@app.command()
def plan(
    description: str = typer.Argument(None, help="Project description"),
    context: str | None = typer.Option(None, "--context", "-c", help="Additional context"),
    mock: bool = typer.Option(False, "--mock", help="Use mock LLM for testing"),
    model: str = typer.Option("o3", "--model", "-m", help="OpenAI model to use"),
    example: str | None = typer.Option(None, "--example", "-e", help="Use an example project (1-4)"),
):
    """Plan a project by breaking it down into tasks using AI."""
    # Handle example projects
    if example and example in EXAMPLE_PROJECTS:
        project = EXAMPLE_PROJECTS[example]
        console.print(f"\n[bold cyan]Using example project:[/bold cyan] {project['name']}")
        description = project["description"]
    elif not description:
        console.print("[red]Error: Either provide a project description or use --example flag[/red]")
        raise typer.Exit(1)
    
    asyncio.run(_plan_project(description, context, mock, model))


async def _plan_project(description: str, context: str | None, mock: bool, model: str):
    """Async implementation of project planning."""
    global _current_tasks

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

        # Use environment variable for model if not specified
        env_model = os.getenv("OPENAI_MODEL", model)
        llm_provider = OpenAIProvider(api_key=api_key, model=env_model)
        console.print(f"[green]Using OpenAI {env_model}[/green]")

    task_repository = InMemoryTaskRepository()
    
    # Use streaming runner for real-time updates
    runner = StreamingPlanRunner(llm_provider, task_repository)
    
    # Execute planning with streaming updates
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    
    # Create progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
    
    # Initialize progress tracking
    main_task = progress.add_task("Planning project...", total=100)
    created_tasks = []
    current_status = ""
    
    # Update handler for streaming progress
    def handle_update(update: PlanningUpdate):
        nonlocal current_status, created_tasks
        
        # Update progress bar
        progress.update(main_task, completed=int(update.progress * 100))
        
        # Update status message based on update type
        if update.type == UpdateType.STARTED:
            progress.update(main_task, description="[cyan]🚀 Starting project analysis...")
        elif update.type == UpdateType.ANALYZING:
            progress.update(main_task, description="[blue]🔍 Analyzing requirements...")
        elif update.type == UpdateType.DECOMPOSING:
            progress.update(main_task, description="[yellow]📝 Breaking down into tasks...")
        elif update.type == UpdateType.TASK_CREATED:
            task = update.data.get("task") if update.data else None
            if task:
                created_tasks.append(task.title)
                progress.update(main_task, description=f"[green]✨ Created: {task.title[:40]}...")
        elif update.type == UpdateType.ESTIMATING:
            progress.update(main_task, description="[magenta]⏱️  Estimating effort...")
        elif update.type == UpdateType.OPTIMIZING:
            progress.update(main_task, description="[cyan]🔄 Optimizing dependencies...")
        elif update.type == UpdateType.COMPLETED:
            progress.update(main_task, description="[bold green]✅ Planning complete!")
        elif update.type == UpdateType.ERROR:
            progress.update(main_task, description=f"[bold red]❌ Error: {update.message}")
    
    # Execute with live display
    with Live(progress, console=console, refresh_per_second=10):
        request = PlanProjectRequest(project_id=uuid4(), description=description, context=context)
        response = await runner.execute(request, handle_update)

    # Store tasks globally for save command
    _current_tasks = response.tasks

    # Display results with enhanced formatting
    console.print(f"\n[bold green]✓[/bold green] {response.summary}\n")
    
    # Show project overview
    total_hours = sum(task.expected_hours for task in response.tasks)
    overview_panel = Panel(
        f"[bold]📊 Project Overview[/bold]\n\n"
        f"  • [cyan]{len(response.tasks)}[/cyan] tasks identified\n"
        f"  • [green]~{total_hours:.0f} hours[/green] total effort\n"
        f"  • [yellow]{sum(1 for t in response.tasks if t.priority.value >= 4)}[/yellow] high-priority tasks\n"
        f"  • [magenta]{sum(1 for t in response.tasks if not t.dependencies)}[/magenta] tasks can start immediately",
        border_style="blue",
        padding=(0, 1),
    )
    console.print(overview_panel)
    console.print()

    # Create enhanced task table
    table = Table(title="📋 Project Tasks", show_lines=True, border_style="blue")
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
            str(i), task.title, task.priority.emoji, time_est, ", ".join(deps) if deps else "-"
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

    # Show insights
    insights = []
    
    # Check for parallel opportunities
    parallel_tasks = sum(1 for t in response.tasks if not t.dependencies)
    if parallel_tasks > 1:
        insights.append(f"💡 {parallel_tasks} tasks can be done in parallel, saving time!")
    
    # Check for critical tasks
    critical_tasks = [t for t in response.tasks if t.priority.value >= 4]
    if critical_tasks:
        insights.append(f"⚡ Focus on '{critical_tasks[0].title}' first - it's critical!")
    
    if insights:
        console.print("\n[bold]Insights:[/bold]")
        for insight in insights:
            console.print(f"  {insight}")


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
    raise typer.Exit()


@app.command()
def save(file_path: Path = typer.Argument(..., help="Path to save the project file")):
    """Save the current project plan to a JSON file."""
    global _current_tasks

    if not _current_tasks:
        console.print("[red]Error: No project planned yet.[/red]")
        console.print("Use 'plan' command first to create a project plan.")
        raise typer.Exit(1)

    # Create repository and save tasks
    repo = JSONTaskRepository(file_path)

    asyncio.run(_save_tasks(repo, _current_tasks))

    console.print(f"[green]✓ Project saved to {file_path}[/green]")
    console.print(f"  {len(_current_tasks)} tasks saved")


@app.command()
def export(
    format: str = typer.Argument(..., help="Export format: markdown, json, csv"),
    file_path: Path | None = typer.Option(None, "--output", "-o", help="Output file path"),
    clipboard: bool = typer.Option(False, "--clipboard", "-c", help="Copy to clipboard"),
):
    """Export the current project plan in various formats."""
    global _current_tasks
    
    if not _current_tasks:
        console.print("[red]Error: No project planned yet.[/red]")
        console.print("Use 'plan' command first to create a project plan.")
        raise typer.Exit(1)
    
    # Validate format
    format = format.lower()
    if format not in ["markdown", "json", "csv"]:
        console.print(f"[red]Error: Unknown format '{format}'[/red]")
        console.print("Supported formats: markdown, json, csv")
        raise typer.Exit(1)
    
    # Generate export content
    if format == "markdown":
        content = _export_to_markdown(_current_tasks)
        default_ext = ".md"
    elif format == "json":
        content = _export_to_json(_current_tasks)
        default_ext = ".json"
    else:  # csv
        content = _export_to_csv(_current_tasks)
        default_ext = ".csv"
    
    # Handle output
    if file_path:
        file_path.write_text(content)
        console.print(f"[green]✅ Exported to {file_path}[/green]")
    
    if clipboard:
        try:
            import pyperclip
            pyperclip.copy(content)
            console.print("[green]📋 Copied to clipboard![/green]")
        except ImportError:
            console.print("[yellow]Warning: pyperclip not installed. Install with: pip install pyperclip[/yellow]")
    
    if not file_path and not clipboard:
        # Default to saving with auto-generated name
        from datetime import datetime
        filename = f"project_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}{default_ext}"
        Path(filename).write_text(content)
        console.print(f"[green]✅ Exported to {filename}[/green]")
    
    # Show export summary
    lines = content.count('\n') + 1
    size = len(content.encode('utf-8'))
    console.print(f"[dim]  Format: {format.upper()} | Tasks: {len(_current_tasks)} | Lines: {lines} | Size: {size} bytes[/dim]")


def _export_to_markdown(_current_tasks: list) -> str:
    """Export tasks to Markdown format."""
    from datetime import datetime
    
    lines = []
    lines.append("# Project Plan")
    lines.append(f"\n*Generated by Task Manager AI on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    
    # Overview
    total_hours = sum(task.expected_hours for task in _current_tasks)
    lines.append("## 📊 Overview\n")
    lines.append(f"- **Total Tasks:** {len(_current_tasks)}")
    lines.append(f"- **Total Effort:** ~{total_hours:.0f} hours")
    lines.append(f"- **High Priority:** {sum(1 for t in _current_tasks if t.priority.value >= 4)} tasks")
    lines.append("")
    
    # Task list
    lines.append("## 📋 Tasks\n")
    
    # Group by priority
    priority_groups = {}
    for task in _current_tasks:
        priority = task.priority.name
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(task)
    
    # Sort priorities
    priority_order = ["URGENT", "HIGH", "MEDIUM", "LOW"]
    
    for priority in priority_order:
        if priority in priority_groups:
            tasks = priority_groups[priority]
            lines.append(f"### {tasks[0].priority.emoji} {priority.title()} Priority\n")
            
            for i, task in enumerate(tasks, 1):
                lines.append(f"#### {i}. {task.title}")
                if task.description:
                    lines.append(f"\n{task.description}\n")
                
                lines.append(f"- **Estimated Time:** {task.expected_hours:.1f} hours")
                lines.append(f"- **Status:** {task.status.value}")
                
                if task.dependencies:
                    dep_titles = []
                    for dep_id in task.dependencies:
                        dep_task = next((t for t in _current_tasks if t.id == dep_id), None)
                        if dep_task:
                            dep_titles.append(dep_task.title)
                    if dep_titles:
                        lines.append(f"- **Dependencies:** {', '.join(dep_titles)}")
                
                lines.append("")
    
    # Execution plan
    lines.append("## 🎯 Execution Plan\n")
    lines.append("Tasks that can be started immediately:\n")
    
    immediate_tasks = [t for t in _current_tasks if not t.dependencies]
    for task in immediate_tasks:
        lines.append(f"- [ ] {task.title} ({task.expected_hours:.1f}h)")
    
    lines.append("\n---\n")
    lines.append("*Created with [Task Manager AI](https://github.com/yourusername/task-manager)*")
    
    return '\n'.join(lines)


def _export_to_json(_current_tasks: list) -> str:
    """Export tasks to JSON format."""
    import json
    from datetime import datetime
    
    data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generator": "Task Manager AI",
            "version": "1.0",
            "task_count": len(_current_tasks),
            "total_hours": sum(task.expected_hours for task in _current_tasks)
        },
        "tasks": []
    }
    
    for i, task in enumerate(_current_tasks):
        task_data = {
            "id": str(task.id),
            "index": i + 1,
            "title": task.title,
            "description": task.description,
            "priority": {
                "level": task.priority.value,
                "name": task.priority.name,
                "emoji": task.priority.emoji
            },
            "status": task.status.value,
            "estimated_hours": task.expected_hours,
            "dependencies": [str(dep) for dep in task.dependencies],
            "dependency_titles": []
        }
        
        # Add dependency titles for readability
        for dep_id in task.dependencies:
            dep_task = next((t for t in _current_tasks if t.id == dep_id), None)
            if dep_task:
                task_data["dependency_titles"].append(dep_task.title)
        
        data["tasks"].append(task_data)
    
    return json.dumps(data, indent=2)


def _export_to_csv(_current_tasks: list) -> str:
    """Export tasks to CSV format."""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "ID", "Title", "Description", "Priority", "Priority Level", 
        "Status", "Estimated Hours", "Dependencies", "Can Start"
    ])
    
    # Tasks
    for i, task in enumerate(_current_tasks, 1):
        # Get dependency titles
        dep_titles = []
        for dep_id in task.dependencies:
            dep_task = next((t for t in _current_tasks if t.id == dep_id), None)
            if dep_task:
                dep_titles.append(dep_task.title)
        
        writer.writerow([
            i,
            task.title,
            task.description,
            task.priority.name,
            task.priority.value,
            task.status.value,
            f"{task.expected_hours:.1f}",
            "; ".join(dep_titles) if dep_titles else "None",
            "Yes" if not task.dependencies else "No"
        ])
    
    return output.getvalue()


async def _save_tasks(repo: JSONTaskRepository, tasks):
    """Save tasks to repository."""
    for task in tasks:
        await repo.save(task)


@app.command()
def load(file_path: Path = typer.Argument(..., help="Path to the project file to load")):
    """Load a project plan from a JSON file."""
    global _current_tasks

    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    # Load tasks from repository
    repo = JSONTaskRepository(file_path)
    _current_tasks = asyncio.run(repo.get_all())

    if not _current_tasks:
        console.print("[yellow]Warning: No tasks found in file.[/yellow]")
        return

    # Display loaded project
    console.print(f"\n[green]✓ Project loaded from {file_path}[/green]")
    console.print(f"  {len(_current_tasks)} tasks loaded\n")

    _display_tasks_table(_current_tasks)

    # Calculate total time
    total_hours = sum(task.expected_hours for task in _current_tasks)
    console.print(f"\n[bold]Total estimated time:[/bold] {total_hours:.1f} hours")


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
        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break

        elif user_input.lower() in ["help", "h"]:
            console.clear()
            _show_header()
            continue

        elif user_input.lower() == "clear":
            console.clear()
            _show_header()
            continue

        elif user_input.lower() == "history":
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
                    project_id=uuid4(), description=user_input, context=None
                )
                try:
                    response = await use_case.execute(request)

                    # Add to history
                    history.append({"description": user_input, "tasks": response.tasks})

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
    header_text = (
        "🤖 Task Manager AI Agent 🤖\n\nBreak down complex projects into manageable tasks with AI!"
    )

    panel = Panel(
        header_text, style="cyan", border_style="bright_blue", padding=(1, 2), title_align="center"
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
            str(i), task.title, task.priority.emoji, time_est, ", ".join(deps) if deps else "-"
        )

    console.print(table)


# Create agent subcommand group
agent_app = typer.Typer(help="Agent-based planning commands")
app.add_typer(agent_app, name="agent")


@agent_app.command(name="plan")
def agent_plan(
    description: str = typer.Argument(..., help="Project description to plan"),
    mock: bool = typer.Option(False, "--mock", help="Use mock LLM for testing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show agent reasoning process"),
    estimate: bool = typer.Option(
        False, "--estimate", "-e", help="Calculate time and cost estimations"
    ),
    tools: bool = typer.Option(
        False, "--tools", "-t", help="Enable web search and calculator tools"
    ),
    no_safety: bool = typer.Option(
        False, "--no-safety", help="Disable safety filters (not recommended)"
    ),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from last checkpoint"),
    checkpoint: bool = typer.Option(True, "--checkpoint/--no-checkpoint", help="Enable auto-checkpointing"),
):
    """Use AI agent to plan a project with reasoning steps."""
    asyncio.run(_agent_plan_project(description, mock, verbose, estimate, tools, not no_safety, resume, checkpoint))


async def _agent_plan_project(
    description: str, mock: bool, verbose: bool, estimate: bool = False, enable_tools: bool = False,
    enable_safety: bool = True, resume: bool = False, enable_checkpoint: bool = True
):
    """Execute agent-based project planning."""
    global _current_tasks

    console.print(f"\n[bold blue]🤖 Agent Planning:[/bold blue] {description}")

    # Show safety status
    if not enable_safety:
        console.print("[bold red]⚠️  Safety filters disabled - Use with caution![/bold red]")

    # Initialize LLM provider
    if mock:
        console.print("[yellow]Using mock LLM provider[/yellow]")
        llm_provider = MockLLMProvider()
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY not found[/red]")
            raise typer.Exit(1)
        llm_provider = OpenAIProvider(api_key=api_key)

    # Create agent with CoT reasoning enabled
    task_decomposer = TaskDecomposer()
    
    # Check for resume
    if resume:
        # Try to find latest checkpoint
        from src.domain.services.checkpoint_service import CheckpointService
        checkpoint_service = CheckpointService()
        latest_checkpoint = await checkpoint_service.get_latest_checkpoint("TaskPlanner")
        
        if not latest_checkpoint:
            console.print("[yellow]No checkpoint found to resume from. Starting fresh.[/yellow]")
            resume = False
    
    agent = TaskPlanningAgent(
        name="TaskPlanner",
        llm_provider=llm_provider,
        task_decomposer=task_decomposer,
        enable_cot=True,  # Enable Chain of Thought reasoning
        enable_tools=enable_tools,  # Enable tools if requested
        enable_estimation=estimate,  # Enable estimation if requested
        enable_safety=enable_safety,  # Enable safety filters
        enable_auto_checkpoint=enable_checkpoint,  # Enable auto-checkpointing
    )
    
    # Resume from checkpoint if requested
    if resume and latest_checkpoint:
        console.print(f"[green]Resuming from checkpoint: {latest_checkpoint.name}[/green]")
        await agent.restore_checkpoint(latest_checkpoint)
        
        # Show resume summary
        resume_info = agent.get_resume_summary()
        console.print(f"\n[bold]Resume Summary:[/bold]")
        console.print(f"  Progress: {resume_info['progress_percentage']}%")
        console.print(f"  Last completed: {resume_info['last_completed']}")
        console.print(f"  Next task: {resume_info['next_task']}")
        console.print(f"  Estimated remaining: {resume_info['estimated_remaining_hours']}h\n")
        
        # Update description to continue from where we left off
        if agent.task_context and "original_request" in agent.task_context:
            description = agent.task_context["original_request"]

    # Register tools based on flags
    if enable_tools or estimate:
        from src.domain.tools.calculator import CalculatorTool
        calculator = CalculatorTool(name="calculator")
        agent.register_tool(calculator)
        
    if enable_tools:
        from src.domain.tools.web_search import WebSearchTool
        web_search = WebSearchTool(name="web_search")
        agent.register_tool(web_search)
        console.print("[green]✅ Tools enabled: calculator, web_search[/green]")

    # Run agent with graceful shutdown handling
    import signal
    
    async def save_checkpoint_on_shutdown(signum, frame):
        """Save checkpoint on graceful shutdown."""
        console.print("\n[yellow]Interrupted! Saving checkpoint...[/yellow]")
        await agent.save_checkpoint(milestone="interrupted")
        console.print("[green]Checkpoint saved. You can resume with --resume flag.[/green]")
        raise typer.Exit(0)
    
    # Set up signal handler for graceful shutdown
    original_handler = signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(save_checkpoint_on_shutdown(s, f)))
    
    try:
        with console.status("[bold green]Agent is thinking..."):
            result = await agent.run(description)
    finally:
        # Restore original handler
        signal.signal(signal.SIGINT, original_handler)

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    # Extract tasks and update global state
    _current_tasks = result.get("tasks", [])

    # Display results
    console.print("\n[bold green]✓ Agent completed planning[/bold green]")

    if verbose:
        # Show Chain of Thought reasoning steps
        reasoning_steps = result.get("reasoning_steps", [])
        if reasoning_steps:
            console.print("\n[bold cyan]Chain of Thought Reasoning:[/bold cyan]")
            for step in reasoning_steps:
                console.print(f"\n  [bold]Step {step['step']}:[/bold]")
                console.print(f"  💭 {step['thought']}")
                console.print(f"  ✓ {step['conclusion']}")

        # Show agent's memory
        console.print("\n[dim]Agent Memory:[/dim]")
        for i, memory in enumerate(agent.memory.short_term):
            if memory["content"].get("type") != "reasoning_step":  # Don't duplicate reasoning steps
                console.print(f"  Step {i+1}: {memory['content'].get('thought', 'N/A')}")

    # Display tasks
    if _current_tasks:
        _display_tasks_table(_current_tasks)

        # Show execution plan
        execution_plan = result.get("execution_plan", [])
        if execution_plan:
            console.print("\n[bold]Execution Plan:[/bold]")
            for step in execution_plan:
                console.print(f"\n  Step {step['step']}:")
                for task in step["tasks"]:
                    console.print(f"    - {task['title']} {task['priority']}")
                if step.get("can_parallel"):
                    console.print("    [dim](Can be done in parallel)[/dim]")

    # Display estimations if available
    estimations = result.get("estimations", {})
    if estimations:
        console.print("\n[bold]Project Estimations:[/bold]")
        console.print(f"  Total Hours: {estimations.get('total_hours', 0):.1f}h")
        console.print(f"  With 20% Buffer: {estimations.get('hours_with_buffer', 0):.1f}h")
        console.print(f"  Estimated Cost: {estimations.get('cost_calculation', 'N/A')}")
        if estimations.get("parallel_time_saved", 0) > 0:
            console.print(f"  Time Saved (Parallel): {estimations['parallel_time_saved']:.1f}h")

    console.print(f"\n[dim]Agent State: {agent.state.value}[/dim]")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Show welcome screen when no command is provided."""
    if ctx.invoked_subcommand is None:
        # No command provided, show welcome wizard
        asyncio.run(_welcome_wizard())


async def _welcome_wizard():
    """Interactive welcome wizard for first-time users."""
    console.clear()
    
    # Show impressive welcome banner
    from rich.align import Align
    from rich.panel import Panel
    
    banner = Panel(
        Align.center(
            "[bold cyan]🚀 Welcome to Task Manager AI[/bold cyan]\n"
            "[dim]Your Project Planning Ally[/dim]\n\n"
            "[italic]Turn your ideas into actionable plans in seconds![/italic]"
        ),
        border_style="bright_blue",
        padding=(1, 2),
        box=rich.box.DOUBLE_EDGE,
    )
    console.print(banner)
    console.print()
    
    # Show example projects
    console.print("[bold]First time? Let me show you something amazing![/bold]\n")
    console.print("Choose an example project to see the magic:")
    
    for key, project in EXAMPLE_PROJECTS.items():
        console.print(f"  {key}. {project['name']}")
    
    console.print("  5. ✍️  Enter your own project")
    console.print()
    
    # Get user choice
    choice = Prompt.ask("[bold cyan]→ Your choice (1-5)[/bold cyan]", default="1")
    
    if choice in EXAMPLE_PROJECTS:
        # Use example project
        project = EXAMPLE_PROJECTS[choice]
        console.print(f"\n[green]Great choice![/green] Let's plan: {project['name']}\n")
        
        # Check for API key and use mock if not available
        api_key = os.getenv("OPENAI_API_KEY")
        use_mock = not api_key
        
        if use_mock:
            console.print("\n[yellow]ℹ️  No API key found - using mock mode for demonstration[/yellow]")
            console.print("[dim]To use real AI, add your OpenAI API key to .env file[/dim]\n")
        
        # Show progress theater
        await _show_progress_theater("Analyzing your project")
        
        # Run the planning with error handling
        try:
            await _plan_project(project["description"], None, use_mock, os.getenv("OPENAI_MODEL", "o3"))
            # Show success celebration
            _show_success_celebration()
        except Exception as e:
            console.print(f"\n[red]❌ Error: {str(e)}[/red]")
            if "model" in str(e).lower() and "not exist" in str(e).lower():
                console.print("[yellow]ℹ️  This might be due to an invalid model name or insufficient API access.[/yellow]")
                console.print("[dim]Try using --model gpt-3.5-turbo, --model gpt-4, or check your API key permissions.[/dim]")
                console.print("[dim]Note: o3 is a newer model that may require special access.[/dim]")
            console.print("\n[dim]You can also use --mock flag to test without an API key.[/dim]")
        
    elif choice == "5":
        # Get custom project
        console.print("\n[bold]Tell me about your project:[/bold]")
        description = Prompt.ask("[cyan]→[/cyan]")
        
        if description.strip():
            # Check for API key and use mock if not available
            api_key = os.getenv("OPENAI_API_KEY")
            use_mock = not api_key
            
            if use_mock:
                console.print("\n[yellow]ℹ️  No API key found - using mock mode for demonstration[/yellow]")
                console.print("[dim]To use real AI, add your OpenAI API key to .env file[/dim]\n")
            
            await _show_progress_theater("Understanding your vision")
            try:
                await _plan_project(description, None, use_mock, os.getenv("OPENAI_MODEL", "o3"))
                _show_success_celebration()
            except Exception as e:
                console.print(f"\n[red]❌ Error: {str(e)}[/red]")
                if "model" in str(e).lower() and "not exist" in str(e).lower():
                    console.print("[yellow]ℹ️  This might be due to an invalid model name or insufficient API access.[/yellow]")
                    console.print("[dim]Try using --model gpt-3.5-turbo, --model gpt-4, or check your API key permissions.[/dim]")
                console.print("[dim]Note: o3 is a newer model that may require special access.[/dim]")
                console.print("\n[dim]You can also use --mock flag to test without an API key.[/dim]")
    else:
        console.print("[yellow]Invalid choice. Please run the command again.[/yellow]")


async def _show_progress_theater(base_message: str):
    """Show engaging progress animation while processing."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    import asyncio
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Simulate progress steps
        steps = [
            ("🧠 Understanding your project...", 25),
            ("📊 Analyzing scope and complexity...", 50),
            ("🔍 Identifying key components...", 75),
            ("🎯 Optimizing task dependencies...", 90),
            ("✨ Finalizing your plan...", 100),
        ]
        
        task = progress.add_task(f"[cyan]{base_message}...", total=100)
        
        for step_text, target_progress in steps:
            progress.update(task, description=f"[cyan]{step_text}")
            
            # Animate progress
            current = progress.tasks[0].completed
            for i in range(int(current), target_progress + 1):
                progress.update(task, advance=1)
                await asyncio.sleep(0.02)  # Small delay for animation
            
            await asyncio.sleep(0.3)  # Pause between steps


def _show_success_celebration():
    """Show success message with helpful next steps."""
    console.print("\n")
    
    # Create quick export prompt
    from rich.prompt import Confirm
    
    success_panel = Panel(
        "[bold green]🎉 Success! Your project plan has been created![/bold green]\n\n"
        "[bold]📤 Export Your Plan:[/bold]\n"
        "  • [cyan]'task-manager export markdown'[/cyan] - GitHub-ready documentation\n"
        "  • [cyan]'task-manager export json'[/cyan] - For integrations\n"
        "  • [cyan]'task-manager export csv'[/cyan] - For spreadsheets\n\n"
        "[bold]🚀 Next Steps:[/bold]\n"
        "  • Add [cyan]'--tools'[/cyan] for AI-powered web research\n"
        "  • Use [cyan]'--estimate'[/cyan] for time & cost calculations\n\n"
        "[dim]💝 Enjoying Task Manager? Star us on GitHub![/dim]",
        border_style="green",
        padding=(1, 2),
    )
    console.print(success_panel)
    
    # Offer quick export
    if _current_tasks and Confirm.ask("\n[bold cyan]Would you like to export your plan now?[/bold cyan]", default=True):
        console.print("\n[bold]Choose format:[/bold]")
        console.print("  1. 📝 Markdown (recommended)")
        console.print("  2. 📊 CSV (for Excel/Sheets)")
        console.print("  3. 🗂️  JSON (for developers)")
        
        format_choice = Prompt.ask("[cyan]→ Format (1-3)[/cyan]", default="1")
        format_map = {"1": "markdown", "2": "csv", "3": "json"}
        
        if format_choice in format_map:
            format = format_map[format_choice]
            from datetime import datetime
            filename = f"project_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if format == "markdown":
                content = _export_to_markdown(_current_tasks)
                filename += ".md"
            elif format == "csv":
                content = _export_to_csv(_current_tasks)
                filename += ".csv"
            else:
                content = _export_to_json(_current_tasks)
                filename += ".json"
            
            Path(filename).write_text(content)
            console.print(f"\n[green]✅ Exported to {filename}[/green]")
            
            # Try to copy to clipboard
            try:
                import pyperclip
                pyperclip.copy(content)
                console.print("[green]📋 Also copied to clipboard![/green]")
            except:
                pass


if __name__ == "__main__":
    app()
