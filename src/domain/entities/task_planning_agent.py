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
        task_decomposer: TaskDecomposer
    ):
        super().__init__(name)
        self.llm_provider = llm_provider
        self.task_decomposer = task_decomposer
        self.project_context = {}
        self.current_tasks = []
    
    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project requirements and decide on planning approach."""
        project_input = observation.get("input", "")
        iteration = observation.get("iteration", 0)
        
        if iteration == 0:
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
            return {
                "thought": "Creating execution plan from decomposed tasks",
                "action": "finalize_plan",
                "is_complete": True,
                "result": {
                    "tasks": self.current_tasks,
                    "execution_plan": self._create_execution_plan(),
                    "total_tasks": len(self.current_tasks),
                    "project_description": project_input
                }
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