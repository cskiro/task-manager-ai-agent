"""Unit tests for Task entity."""
import pytest
from uuid import uuid4

from src.domain.entities import Task
from src.domain.value_objects import Priority, TaskStatus, TimeEstimate


class TestTask:
    """Test cases for Task entity."""
    
    def test_task_creation(self):
        """Test creating a task with default values."""
        task = Task(title="Implement feature")
        
        assert task.title == "Implement feature"
        assert task.description == ""
        assert task.priority == Priority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.dependencies == []
        assert task.time_estimate is None
    
    def test_task_creation_requires_title(self):
        """Test that task creation fails without title."""
        with pytest.raises(ValueError, match="Task title cannot be empty"):
            Task(title="")
    
    def test_task_with_time_estimate(self):
        """Test creating task with time estimate."""
        estimate = TimeEstimate.from_hours(
            optimistic=2.0,
            realistic=4.0,
            pessimistic=8.0
        )
        task = Task(
            title="Complex feature",
            time_estimate=estimate
        )
        
        assert task.expected_hours == pytest.approx(4.33, rel=0.01)
    
    def test_task_lifecycle(self):
        """Test task state transitions."""
        task = Task(title="Test task")
        
        # Start task
        task.start()
        assert task.status == TaskStatus.IN_PROGRESS
        
        # Complete task
        task.complete()
        assert task.status == TaskStatus.COMPLETED
        assert task.is_completed
    
    def test_cannot_start_non_pending_task(self):
        """Test that only pending tasks can be started."""
        task = Task(title="Test task")
        task.start()
        
        with pytest.raises(ValueError, match="Cannot start task in in_progress status"):
            task.start()
    
    def test_task_dependencies(self):
        """Test task dependency management."""
        task1_id = uuid4()
        task2_id = uuid4()
        
        task = Task(title="Dependent task")
        task.add_dependency(task1_id)
        task.add_dependency(task2_id)
        
        assert len(task.dependencies) == 2
        assert not task.can_start(set())
        assert not task.can_start({task1_id})
        assert task.can_start({task1_id, task2_id})
    
    def test_task_cannot_depend_on_itself(self):
        """Test that task cannot have itself as dependency."""
        task = Task(title="Test task")
        
        with pytest.raises(ValueError, match="Task cannot depend on itself"):
            task.add_dependency(task.id)