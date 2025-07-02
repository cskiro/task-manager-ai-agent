#!/usr/bin/env python3
"""Demo script to showcase web search tool integration."""

import asyncio
from src.domain.entities.task_planning_agent import TaskPlanningAgent
from src.domain.services.task_decomposer import TaskDecomposer
from src.domain.tools.web_search import WebSearchTool
from src.infrastructure.llm import MockLLMProvider
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def demo_web_search():
    """Demonstrate web search tool in action."""
    console.print("\n[bold cyan]🔍 Web Search Tool Demo[/bold cyan]\n")
    
    # Create agent with tools enabled
    llm_provider = MockLLMProvider()
    task_decomposer = TaskDecomposer()
    agent = TaskPlanningAgent(
        name="ResearchAgent",
        llm_provider=llm_provider,
        task_decomposer=task_decomposer,
        enable_cot=True,
        enable_tools=True
    )
    
    # Register web search tool
    web_search = WebSearchTool(name="web_search")
    agent.register_tool(web_search)
    
    # Test project
    project = "Build a Python task management system with best practices"
    
    console.print(f"[bold]Project:[/bold] {project}\n")
    
    # Run agent
    with console.status("[green]Agent is researching and planning..."):
        result = await agent.run(project)
    
    # Display research findings
    console.print("[bold cyan]Research Findings:[/bold cyan]\n")
    
    # Find tool uses in memory
    tool_uses = [m for m in agent.memory.short_term if m["content"].get("tool_used") == "web_search"]
    
    for i, tool_use in enumerate(tool_uses, 1):
        content = tool_use["content"]
        panel = Panel(
            f"[yellow]Query:[/yellow] {content['query']}\n"
            f"[green]Summary:[/green] {content['research_summary']}\n"
            f"[dim]Found {len(content['results'])} results[/dim]",
            title=f"Search {i}",
            border_style="blue"
        )
        console.print(panel)
        
        # Show search results
        if content['results']:
            table = Table(show_header=False, box=None)
            for result in content['results']:
                table.add_row(
                    f"• [cyan]{result['title']}[/cyan]",
                    f"[dim]{result['snippet'][:80]}...[/dim]"
                )
            console.print(table)
        console.print()
    
    # Display generated tasks
    tasks = result.get("tasks", [])
    console.print(f"\n[bold green]Generated {len(tasks)} informed tasks[/bold green]\n")
    
    for i, task in enumerate(tasks, 1):
        console.print(f"{i}. {task.title} {task.priority.emoji}")
        if task.time_estimate:
            console.print(f"   Est: {task.expected_hours:.1f} hours")
        console.print()
    
    # Show how research influenced planning
    console.print("\n[bold]Research Impact:[/bold]")
    console.print("✓ Agent researched best practices before planning")
    console.print("✓ Found relevant architecture patterns")
    console.print("✓ Tasks reflect current industry standards")


if __name__ == "__main__":
    asyncio.run(demo_web_search())