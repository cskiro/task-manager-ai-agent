"""Task Planning Agent implementation."""

from typing import Dict, Any, List
from uuid import UUID

from src.domain.entities.agent import BaseAgent
from src.domain.entities.task import Task
from src.domain.services.task_decomposer import TaskDecomposer, LLMProvider


class TaskPlanningAgent(BaseAgent):
    """Agent specialized in project planning and task decomposition."""
    
    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider,
        task_decomposer: TaskDecomposer,
        enable_cot: bool = False,
        enable_tools: bool = False
    ):
        super().__init__(name)
        self.llm_provider = llm_provider
        self.task_decomposer = task_decomposer
        self.enable_cot = enable_cot
        self.enable_tools = enable_tools
        self.project_context = {}
        self.current_tasks = []
        self.reasoning_steps = []
    
    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project requirements and decide on planning approach."""
        project_input = observation.get("input", "")
        iteration = observation.get("iteration", 0)
        
        if iteration == 0:
            # Generate Chain of Thought reasoning if enabled
            if self.enable_cot:
                await self._generate_cot_reasoning(project_input)
            
            # Use tools to research if enabled
            if self.enable_tools and "web_search" in self.tools:
                await self._research_project(project_input)
            
            # First iteration: decompose the project
            tasks = await self.task_decomposer.decompose(
                project_input,
                self.llm_provider
            )
            
            self.current_tasks = tasks
            
            return {
                "thought": f"Decomposed project into {len(tasks)} tasks",
                "action": "decompose_project",
                "tasks": tasks,
                "is_complete": False
            }
        else:
            # Second iteration: create execution plan
            result = {
                "tasks": self.current_tasks,
                "execution_plan": self._create_execution_plan(),
                "total_tasks": len(self.current_tasks),
                "project_description": project_input
            }
            
            # Include reasoning steps if CoT is enabled
            if self.enable_cot and self.reasoning_steps:
                result["reasoning_steps"] = self.reasoning_steps
            
            return {
                "thought": "Creating execution plan from decomposed tasks",
                "action": "finalize_plan",
                "is_complete": True,
                "result": result
            }
    
    async def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning actions."""
        action_type = action.get("action")
        
        if action_type == "decompose_project":
            # Tasks are already decomposed in think phase
            return {
                "status": "tasks_decomposed",
                "task_count": len(action.get("tasks", []))
            }
        
        elif action_type == "create_plan":
            tasks = action.get("tasks", [])
            execution_plan = self._create_execution_plan_for_tasks(tasks)
            
            return {
                "status": "plan_created",
                "execution_plan": execution_plan,
                "total_tasks": len(tasks)
            }
        
        else:
            return {"status": "completed"}
    
    def _create_execution_plan(self) -> List[Dict[str, Any]]:
        """Create execution plan from current tasks."""
        return self._create_execution_plan_for_tasks(self.current_tasks)
    
    def _create_execution_plan_for_tasks(self, tasks: List[Task]) -> List[Dict[str, Any]]:
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
                step_tasks.append({
                    "task_id": str(task.id),
                    "title": task.title,
                    "description": task.description,
                    "priority": task.priority.emoji,
                    "estimated_hours": task.expected_hours
                })
                completed.add(task.id)
            
            execution_plan.append({
                "step": len(execution_plan) + 1,
                "tasks": step_tasks,
                "can_parallel": len(step_tasks) > 1
            })
        
        return execution_plan
    
    async def _generate_cot_reasoning(self, project_description: str) -> None:
        """Generate Chain of Thought reasoning steps for project analysis."""
        self.reasoning_steps = []
        
        # Step 1: Understand project scope
        step1 = {
            "step": 1,
            "thought": f"First, I need to understand the project scope. The user wants to: {project_description}",
            "conclusion": "This appears to be a software development project that will require careful planning."
        }
        self.reasoning_steps.append(step1)
        self._add_reasoning_to_memory(step1)
        
        # Step 2: Analyze architecture requirements
        step2 = {
            "step": 2,
            "thought": f"I should analyze the architecture requirements for this project. {'distributed' in project_description.lower() and 'This involves distributed systems architecture.' or 'This requires a solid software architecture.'}",
            "conclusion": "The architecture needs to be scalable and maintainable."
        }
        self.reasoning_steps.append(step2)
        self._add_reasoning_to_memory(step2)
        
        # Step 3: Consider dependencies and scaling
        step3 = {
            "step": 3,
            "thought": "I need to think about task dependencies and how the system will scale. Some components must be built before others.",
            "conclusion": "Setup and infrastructure tasks should come first, with consideration for future scaling needs."
        }
        self.reasoning_steps.append(step3)
        self._add_reasoning_to_memory(step3)
        
        # Step 4: Time estimation strategy
        step4 = {
            "step": 4,
            "thought": "I should consider time estimates for each task based on complexity and dependencies.",
            "conclusion": "I'll use PERT estimation (optimistic, realistic, pessimistic) for more accurate planning."
        }
        self.reasoning_steps.append(step4)
        self._add_reasoning_to_memory(step4)
    
    def _add_reasoning_to_memory(self, reasoning_step: Dict[str, Any]) -> None:
        """Add a reasoning step to agent memory."""
        self.memory.add_observation({
            "type": "reasoning_step",
            **reasoning_step
        })
    
    async def _research_project(self, project_description: str) -> None:
        """Use web search tool to research the project."""
        web_search = self.tools["web_search"]
        
        # Research best practices
        query = f"{project_description} best practices"
        result = await web_search.search(query, max_results=3)
        
        # Store research in memory
        self.memory.add_observation({
            "tool_used": "web_search",
            "query": query,
            "results": result.data.get("results", []),
            "research_summary": "Found best practices for the project"
        })
        
        # If specific technology mentioned, research it
        if "React" in project_description or "Python" in project_description:
            tech_query = f"{project_description} architecture patterns"
            tech_result = await web_search.search(tech_query, max_results=2)
            
            self.memory.add_observation({
                "tool_used": "web_search", 
                "query": tech_query,
                "results": tech_result.data.get("results", []),
                "research_summary": "Researched architecture patterns"
            })