"""Task Planning Agent implementation."""

from pathlib import Path
from typing import Any, Optional

from src.domain.entities.agent import AgentState, BaseAgent
from src.domain.entities.checkpoint import AgentCheckpoint
from src.domain.entities.task import Task
from src.domain.safety.basic_filter import BasicSafetyFilter
from src.domain.services.checkpoint_service import CheckpointService
from src.domain.services.task_decomposer import LLMProvider, TaskDecomposer


class TaskPlanningAgent(BaseAgent):
    """Agent specialized in project planning and task decomposition."""

    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        task_decomposer: TaskDecomposer,
        enable_cot: bool = False,
        enable_tools: bool = False,
        enable_estimation: bool = False,
        enable_safety: bool = True,
        max_tasks: int = 20,
        enable_auto_checkpoint: bool = False,
        checkpoint_dir: Optional[Path] = None,
        max_checkpoints: int = 3,
    ):
        super().__init__(name)
        self.llm_provider = llm_provider
        self.task_decomposer = task_decomposer
        self.enable_cot = enable_cot
        self.enable_tools = enable_tools
        self.enable_estimation = enable_estimation
        self.enable_safety = enable_safety
        self.project_context = {}
        self.current_tasks = []
        self.reasoning_steps = []
        self.enable_auto_checkpoint = enable_auto_checkpoint
        
        # Initialize checkpoint service
        self.checkpoint_service = CheckpointService(checkpoint_dir, max_checkpoints)

        # Initialize safety filter if enabled
        self.safety_filter = BasicSafetyFilter(max_tasks=max_tasks) if enable_safety else None

    async def think(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Analyze project requirements and decide on planning approach."""
        project_input = observation.get("input", "")
        iteration = observation.get("iteration", 0)

        # Validate input if safety is enabled
        if self.safety_filter and iteration == 0:
            validation_result = await self.safety_filter.validate_input(project_input)
            if not validation_result.valid:
                return {
                    "thought": f"Input validation failed: {validation_result.reason}",
                    "action": "reject_input",
                    "is_complete": True,
                    "error": f"Safety check failed: {validation_result.reason}",
                }

        if iteration == 0:
            # Generate Chain of Thought reasoning if enabled
            if self.enable_cot:
                await self._generate_cot_reasoning(project_input)

            # Use tools to research if enabled
            if self.enable_tools and "web_search" in self.tools:
                await self._research_project(project_input)

            # First iteration: decompose the project
            tasks = await self.task_decomposer.decompose(project_input, self.llm_provider)

            self.current_tasks = tasks
            
            # Auto-checkpoint after task decomposition if enabled
            if self.enable_auto_checkpoint:
                await self.save_checkpoint(milestone="task_decomposition_complete")

            return {
                "thought": f"Decomposed project into {len(tasks)} tasks",
                "action": "decompose_project",
                "tasks": tasks,
                "is_complete": False,
            }
        else:
            # Second iteration: create execution plan
            result = {
                "tasks": self.current_tasks,
                "execution_plan": self._create_execution_plan(),
                "total_tasks": len(self.current_tasks),
                "project_description": project_input,
            }

            # Validate output if safety is enabled
            if self.safety_filter:
                validation_result = await self.safety_filter.validate_output(result)
                if not validation_result.valid:
                    return {
                        "thought": f"Output validation failed: {validation_result.reason}",
                        "action": "reject_output",
                        "is_complete": True,
                        "error": f"Safety check failed: {validation_result.reason}",
                    }

            # Include reasoning steps if CoT is enabled
            if self.enable_cot and self.reasoning_steps:
                result["reasoning_steps"] = self.reasoning_steps

            # Include estimations if enabled and calculator available
            if self.enable_estimation and "calculator" in self.tools:
                result["estimations"] = await self._calculate_estimations()

            return {
                "thought": "Creating execution plan from decomposed tasks",
                "action": "finalize_plan",
                "is_complete": True,
                "result": result,
            }

    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute planning actions."""
        action_type = action.get("action")

        if action_type == "decompose_project":
            # Tasks are already decomposed in think phase
            return {"status": "tasks_decomposed", "task_count": len(action.get("tasks", []))}

        elif action_type == "create_plan":
            tasks = action.get("tasks", [])
            execution_plan = self._create_execution_plan_for_tasks(tasks)

            return {
                "status": "plan_created",
                "execution_plan": execution_plan,
                "total_tasks": len(tasks),
            }

        elif action_type in ["reject_input", "reject_output"]:
            # Safety rejection - return error
            return {"status": "rejected", "error": action.get("error", "Safety check failed")}

        else:
            return {"status": "completed"}

    def _create_execution_plan(self) -> list[dict[str, Any]]:
        """Create execution plan from current tasks."""
        return self._create_execution_plan_for_tasks(self.current_tasks)

    def _create_execution_plan_for_tasks(self, tasks: list[Task]) -> list[dict[str, Any]]:
        """Create ordered execution plan respecting dependencies."""
        completed = set()
        execution_plan = []

        while len(completed) < len(tasks):
            # Find tasks that can be executed
            available_tasks = []

            for task in tasks:
                if task.id in completed:
                    continue

                # Check if dependencies are satisfied
                if all(dep in completed for dep in task.dependencies):
                    available_tasks.append(task)

            if not available_tasks:
                # No more tasks can be executed
                break

            # Sort by priority and add to plan
            available_tasks.sort(key=lambda t: t.priority.value, reverse=True)

            step_tasks = []
            for task in available_tasks:
                step_tasks.append(
                    {
                        "task_id": str(task.id),
                        "title": task.title,
                        "description": task.description,
                        "priority": task.priority.emoji,
                        "estimated_hours": task.expected_hours,
                    }
                )
                completed.add(task.id)

            execution_plan.append(
                {
                    "step": len(execution_plan) + 1,
                    "tasks": step_tasks,
                    "can_parallel": len(step_tasks) > 1,
                }
            )

        return execution_plan

    async def _generate_cot_reasoning(self, project_description: str) -> None:
        """Generate Chain of Thought reasoning steps for project analysis."""
        self.reasoning_steps = []

        # Step 1: Understand project scope
        step1 = {
            "step": 1,
            "thought": f"First, I need to understand the project scope. The user wants to: {project_description}",
            "conclusion": "This appears to be a software development project that will require careful planning.",
        }
        self.reasoning_steps.append(step1)
        self._add_reasoning_to_memory(step1)

        # Step 2: Analyze architecture requirements
        step2 = {
            "step": 2,
            "thought": f"I should analyze the architecture requirements for this project. {'distributed' in project_description.lower() and 'This involves distributed systems architecture.' or 'This requires a solid software architecture.'}",
            "conclusion": "The architecture needs to be scalable and maintainable.",
        }
        self.reasoning_steps.append(step2)
        self._add_reasoning_to_memory(step2)

        # Step 3: Consider dependencies and scaling
        step3 = {
            "step": 3,
            "thought": "I need to think about task dependencies and how the system will scale. Some components must be built before others.",
            "conclusion": "Setup and infrastructure tasks should come first, with consideration for future scaling needs.",
        }
        self.reasoning_steps.append(step3)
        self._add_reasoning_to_memory(step3)

        # Step 4: Time estimation strategy
        step4 = {
            "step": 4,
            "thought": "I should consider time estimates for each task based on complexity and dependencies.",
            "conclusion": "I'll use PERT estimation (optimistic, realistic, pessimistic) for more accurate planning.",
        }
        self.reasoning_steps.append(step4)
        self._add_reasoning_to_memory(step4)

    def _add_reasoning_to_memory(self, reasoning_step: dict[str, Any]) -> None:
        """Add a reasoning step to agent memory."""
        self.memory.add_observation({"type": "reasoning_step", **reasoning_step})

    async def _research_project(self, project_description: str) -> None:
        """Use web search tool to research the project."""
        web_search = self.tools["web_search"]

        # Research best practices
        query = f"{project_description} best practices"
        result = await web_search.search(query, max_results=3)

        # Store research in memory
        self.memory.add_observation(
            {
                "tool_used": "web_search",
                "query": query,
                "results": result.data.get("results", []),
                "research_summary": "Found best practices for the project",
            }
        )

        # If specific technology mentioned, research it
        if "React" in project_description or "Python" in project_description:
            tech_query = f"{project_description} architecture patterns"
            tech_result = await web_search.search(tech_query, max_results=2)

            self.memory.add_observation(
                {
                    "tool_used": "web_search",
                    "query": tech_query,
                    "results": tech_result.data.get("results", []),
                    "research_summary": "Researched architecture patterns",
                }
            )

    async def _calculate_estimations(self) -> dict[str, Any]:
        """Calculate project estimations using calculator tool."""
        calculator = self.tools["calculator"]
        estimations = {}

        # Calculate total hours
        total_hours = sum(task.expected_hours for task in self.current_tasks)
        estimations["total_hours"] = total_hours

        # Calculate with buffer (20% buffer is common)
        buffer_calc = await calculator.calculate(f"{total_hours} * 1.2")
        if buffer_calc.success:
            estimations["hours_with_buffer"] = buffer_calc.data["result"]

        # Calculate cost at standard rate ($100/hour for example)
        cost_calc = await calculator.calculate(f"{total_hours} * 100")
        if cost_calc.success:
            estimations["total_cost"] = cost_calc.data["result"]
            estimations["cost_calculation"] = f"${cost_calc.data['result']:.2f} at $100/hour"

        # Calculate parallel execution time savings
        execution_plan = self._create_execution_plan()
        sequential_time = total_hours

        # Calculate parallel time (max time in each parallel step)
        parallel_time = 0
        for step in execution_plan:
            if step.get("can_parallel") and len(step["tasks"]) > 1:
                # Max time of parallel tasks
                step_times = [t["estimated_hours"] for t in step["tasks"]]
                parallel_time += max(step_times)
            else:
                # Sequential tasks
                parallel_time += sum(t["estimated_hours"] for t in step["tasks"])

        time_saved_calc = await calculator.calculate(f"{sequential_time} - {parallel_time}")
        if time_saved_calc.success:
            estimations["parallel_time_saved"] = time_saved_calc.data["result"]

        return estimations
    
    async def save_checkpoint(self, filepath: Optional[Path] = None, milestone: Optional[str] = None) -> Path:
        """Save current agent state to checkpoint.
        
        Args:
            filepath: Optional specific filepath for checkpoint
            milestone: Optional milestone description
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = self.checkpoint_service.create_checkpoint_from_agent(self)
        checkpoint.milestone = milestone
        
        # Add additional context
        checkpoint.reasoning_traces = self.reasoning_steps
        if self.current_tasks:
            checkpoint.task_context = {
                "current_tasks": [
                    {
                        "id": str(task.id),
                        "title": task.title,
                        "status": task.status.value,
                        "priority": task.priority.value,
                    }
                    for task in self.current_tasks
                ]
            }
        
        saved_path = await self.checkpoint_service.save_checkpoint(checkpoint, filepath)
        
        # Auto-checkpoint milestone if enabled
        if self.enable_auto_checkpoint and milestone:
            await self.checkpoint_service.save_checkpoint(checkpoint)
        
        return saved_path
    
    async def restore_checkpoint(self, filepath: Path) -> None:
        """Restore agent state from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = await self.checkpoint_service.load_checkpoint(filepath)
        
        # Restore basic properties
        self.name = checkpoint.agent_name
        self.state = AgentState(checkpoint.state)
        
        # Restore memory
        if "short_term" in checkpoint.memory:
            self.memory.short_term = checkpoint.memory["short_term"]
        
        # Restore task context
        if checkpoint.task_context:
            self.task_context = checkpoint.task_context
        
        # Restore reasoning traces
        self.reasoning_traces = checkpoint.reasoning_traces
        self.reasoning_steps = checkpoint.reasoning_traces  # For backward compatibility
    
    def create_checkpoint(self, milestone: str) -> None:
        """Create a checkpoint synchronously (for test compatibility).
        
        Args:
            milestone: Description of the checkpoint milestone
        """
        import asyncio
        
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.save_checkpoint(milestone=milestone))
        finally:
            loop.close()
