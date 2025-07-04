"""Streaming plan runner for real-time updates during planning."""

import asyncio
import inspect
from typing import Awaitable, Callable, Protocol
from uuid import UUID

from src.application.use_cases.plan_project import (
    PlanProjectRequest,
    PlanProjectResponse,
)
from src.domain.entities import Task
from src.domain.services.task_decomposer import TaskDecomposer
from src.domain.value_objects import Priority, TaskStatus, TimeEstimate
from src.application.models import UpdateType, PlanningUpdate


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Get completion from LLM."""
        ...
    
    async def structured_complete(self, prompt: str, response_model, **kwargs):
        """Get structured response."""
        ...


class TaskRepository(Protocol):
    """Protocol for task persistence."""
    
    async def save(self, task: Task) -> None:
        """Save a task."""
        ...
    
    async def get_by_project(self, project_id: UUID) -> list[Task]:
        """Get tasks for a project."""
        ...


UpdateHandler = Callable[[PlanningUpdate], None] | Callable[[PlanningUpdate], Awaitable[None]]


class StreamingPlanRunner:
    """
    Plan runner that provides real-time updates during planning.
    
    This class orchestrates the project planning process while emitting
    updates at each stage, allowing UIs to show progress and intermediate
    results to users.
    
    Example:
        ```python
        runner = StreamingPlanRunner(llm, repo)
        
        async def show_progress(update: PlanningUpdate):
            print(f"[{update.progress:.0%}] {update.message}")
        
        response = await runner.execute(request, show_progress)
        ```
    """
    
    # Progress milestones for different phases
    PROGRESS_START = 0.0
    PROGRESS_ANALYZING = 0.2
    PROGRESS_DECOMPOSING = 0.4
    PROGRESS_TASKS_START = 0.4
    PROGRESS_TASKS_END = 0.7
    PROGRESS_ESTIMATING = 0.7
    PROGRESS_OPTIMIZING = 0.9
    PROGRESS_COMPLETE = 1.0
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        task_repository: TaskRepository,
        task_decomposer: TaskDecomposer | None = None,
        agent_name: str = "Task Planner"
    ):
        """
        Initialize streaming runner.
        
        Args:
            llm_provider: LLM for generating tasks
            task_repository: Repository for persisting tasks
            task_decomposer: Service for decomposing projects (optional)
            agent_name: Name of the agent for updates
        """
        self.llm = llm_provider
        self.repo = task_repository
        self.decomposer = task_decomposer or TaskDecomposer()
        self.agent_name = agent_name
    
    async def execute(
        self,
        request: PlanProjectRequest,
        on_update: UpdateHandler
    ) -> PlanProjectResponse:
        """
        Execute planning with streaming updates.
        
        Args:
            request: Project planning request
            on_update: Callback for progress updates
            
        Returns:
            Planning response with created tasks
            
        Raises:
            asyncio.CancelledError: If planning is cancelled
            RuntimeError: If LLM or other services fail
        """
        try:
            # Phase 1: Start planning
            await self._update_progress(
                on_update,
                UpdateType.STARTED,
                f"Starting project planning for: {self._truncate(request.description)}",
                self.PROGRESS_START
            )
            
            # Phase 2: Analyze project
            await self._update_progress(
                on_update,
                UpdateType.ANALYZING,
                "Analyzing project requirements and complexity...",
                self.PROGRESS_ANALYZING
            )
            await self._simulate_work()
            
            # Phase 3: Decompose into tasks
            await self._update_progress(
                on_update,
                UpdateType.DECOMPOSING,
                "Breaking down project into manageable tasks...",
                self.PROGRESS_DECOMPOSING
            )
            
            # Generate tasks using LLM
            tasks = await self._generate_tasks(request, on_update)
            
            # Phase 4: Estimate effort
            await self._update_progress(
                on_update,
                UpdateType.ESTIMATING,
                "Estimating time and effort for tasks...",
                self.PROGRESS_ESTIMATING
            )
            await self._simulate_work()
            
            # Phase 5: Optimize schedule
            await self._update_progress(
                on_update,
                UpdateType.OPTIMIZING,
                "Optimizing task dependencies and schedule...",
                self.PROGRESS_OPTIMIZING
            )
            await self._simulate_work()
            
            # Phase 6: Complete
            await self._update_progress(
                on_update,
                UpdateType.COMPLETED,
                f"Successfully planned project with {len(tasks)} tasks",
                self.PROGRESS_COMPLETE
            )
            
            return PlanProjectResponse(
                project_id=request.project_id,
                tasks=tasks,
                summary=f"Created {len(tasks)} tasks for project"
            )
            
        except asyncio.CancelledError:
            # Allow clean cancellation
            raise
            
        except Exception as e:
            # Report error through update system
            await self._update_progress(
                on_update,
                UpdateType.ERROR,
                f"Error during planning: {str(e)}",
                self._estimate_error_progress()
            )
            raise
    
    async def _generate_tasks(
        self,
        request: PlanProjectRequest,
        on_update: UpdateHandler
    ) -> list[Task]:
        """Generate tasks using LLM with progress updates."""
        prompt = self._build_decomposition_prompt(request)
        
        # Get structured response from LLM
        llm_response = await self.llm.structured_complete(
            prompt,
            response_model=dict  # Would use Pydantic model in production
        )
        
        # Parse and create tasks
        tasks = []
        task_data = llm_response.get("tasks", [])
        
        for i, task_info in enumerate(task_data):
            # Create task entity
            task = self._create_task_from_data(task_info)
            
            # Persist task
            await self.repo.save(task)
            tasks.append(task)
            
            # Calculate progress within task creation phase
            task_progress = self._calculate_task_progress(i, len(task_data))
            
            # Emit task creation update
            await self._update_progress(
                on_update,
                UpdateType.TASK_CREATED,
                f"Created task: {task.title}",
                task_progress,
                {"task": task}
            )
        
        return tasks
    
    def _create_task_from_data(self, task_info: dict) -> Task:
        """Create task entity from LLM response data."""
        return Task(
            title=task_info["title"],
            description=task_info.get("description", ""),
            priority=self._parse_priority(task_info.get("priority", 3)),
            status=TaskStatus.PENDING,
            dependencies=[],  # Would parse dependencies in full implementation
            time_estimate=self._parse_time_estimate(task_info.get("time_estimate"))
        )
    
    def _parse_priority(self, priority_value: int | str) -> Priority:
        """Safely parse priority value."""
        try:
            return Priority(int(priority_value))
        except (ValueError, TypeError):
            return Priority.MEDIUM
    
    def _parse_time_estimate(self, estimate_data: dict | None) -> TimeEstimate | None:
        """Parse time estimate from LLM response."""
        if not estimate_data:
            return None
        
        try:
            return TimeEstimate(
                optimistic_hours=float(estimate_data.get("optimistic", 1.0)),
                realistic_hours=float(estimate_data.get("realistic", 2.0)),
                pessimistic_hours=float(estimate_data.get("pessimistic", 3.0))
            )
        except (ValueError, TypeError):
            return None
    
    def _calculate_task_progress(self, current: int, total: int) -> float:
        """Calculate progress within task creation phase."""
        if total == 0:
            return self.PROGRESS_TASKS_START
        
        task_range = self.PROGRESS_TASKS_END - self.PROGRESS_TASKS_START
        return self.PROGRESS_TASKS_START + (task_range * (current + 1) / total)
    
    def _estimate_error_progress(self) -> float:
        """Estimate progress when error occurs."""
        # This would be more sophisticated in production
        return 0.0
    
    async def _update_progress(
        self,
        handler: UpdateHandler,
        update_type: UpdateType,
        message: str,
        progress: float,
        data: dict | None = None
    ) -> None:
        """Emit progress update to handler."""
        update = PlanningUpdate(
            type=update_type,
            message=message,
            progress=max(0.0, min(1.0, progress)),  # Clamp to 0-1
            data=data,
            agent_name=self.agent_name
        )
        
        # Support both sync and async handlers
        if inspect.iscoroutinefunction(handler):
            await handler(update)
        else:
            handler(update)
    
    def _build_decomposition_prompt(self, request: PlanProjectRequest) -> str:
        """Build prompt for task decomposition."""
        return f"""Break down this project into concrete tasks:

Project: {request.description}
Context: {request.context or 'General software project'}

Create 3-7 specific tasks with:
- Clear, actionable titles
- Brief descriptions  
- Priority levels (1-5)
- Time estimates (optimistic, realistic, pessimistic hours)
- Dependencies between tasks

Focus on deliverable milestones."""
    
    def _truncate(self, text: str, max_length: int = 50) -> str:
        """Truncate text for display."""
        if len(text) <= max_length:
            return text
        return f"{text[:max_length]}..."
    
    async def _simulate_work(self, duration: float = 0.1) -> None:
        """Simulate work being done (for realistic UX)."""
        await asyncio.sleep(duration)