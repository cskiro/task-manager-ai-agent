"""Tests for JSON task repository."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from src.domain.entities.task import Task
from src.domain.value_objects import Priority, TaskStatus, TimeEstimate
from src.infrastructure.persistence.json_task_repository import JSONTaskRepository


@pytest.mark.asyncio
class TestJSONTaskRepository:
    """Test cases for JSON task repository."""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        temp_path.unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return Task(
            title="Implement user authentication",
            description="Add JWT-based auth system",
            priority=Priority.HIGH,
            status=TaskStatus.PENDING,
            time_estimate=TimeEstimate.from_hours(4.0, 8.0, 12.0)
        )
    
    async def test_save_single_task(self, temp_file, sample_task):
        """Test saving a single task to JSON file."""
        repo = JSONTaskRepository(temp_file)
        
        await repo.save(sample_task)
        
        # Verify file was created and contains task data
        assert temp_file.exists()
        
        data = json.loads(temp_file.read_text())
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["id"] == str(sample_task.id)
        assert data["tasks"][0]["title"] == sample_task.title
        assert data["tasks"][0]["priority"] == sample_task.priority.value
    
    async def test_get_by_id(self, temp_file, sample_task):
        """Test retrieving a task by ID."""
        repo = JSONTaskRepository(temp_file)
        
        await repo.save(sample_task)
        retrieved = await repo.get_by_id(sample_task.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_task.id
        assert retrieved.title == sample_task.title
        assert retrieved.priority == sample_task.priority
    
    async def test_get_all(self, temp_file):
        """Test retrieving all tasks."""
        repo = JSONTaskRepository(temp_file)
        
        # Create multiple tasks
        tasks = [
            Task(title=f"Task {i}", priority=Priority.MEDIUM)
            for i in range(3)
        ]
        
        for task in tasks:
            await repo.save(task)
        
        retrieved = await repo.get_all()
        assert len(retrieved) == 3
        assert all(t.title in [task.title for task in tasks] for t in retrieved)
    
    async def test_update_task(self, temp_file, sample_task):
        """Test updating an existing task."""
        repo = JSONTaskRepository(temp_file)
        
        await repo.save(sample_task)
        
        # Update task
        sample_task.status = TaskStatus.IN_PROGRESS
        sample_task.title = "Updated title"
        
        await repo.update(sample_task)
        
        retrieved = await repo.get_by_id(sample_task.id)
        assert retrieved.status == TaskStatus.IN_PROGRESS
        assert retrieved.title == "Updated title"
    
    async def test_delete_task(self, temp_file, sample_task):
        """Test deleting a task."""
        repo = JSONTaskRepository(temp_file)
        
        await repo.save(sample_task)
        await repo.delete(sample_task.id)
        
        retrieved = await repo.get_by_id(sample_task.id)
        assert retrieved is None
    
    async def test_persistence_across_instances(self, temp_file, sample_task):
        """Test that data persists when creating new repository instances."""
        repo1 = JSONTaskRepository(temp_file)
        await repo1.save(sample_task)
        
        # Create new instance
        repo2 = JSONTaskRepository(temp_file)
        retrieved = await repo2.get_by_id(sample_task.id)
        
        assert retrieved is not None
        assert retrieved.title == sample_task.title
    
    async def test_empty_repository(self, temp_file):
        """Test operations on empty repository."""
        repo = JSONTaskRepository(temp_file)
        
        tasks = await repo.get_all()
        assert tasks == []
        
        task = await repo.get_by_id(uuid4())
        assert task is None
    
    async def test_task_with_dependencies(self, temp_file):
        """Test saving and loading tasks with dependencies."""
        repo = JSONTaskRepository(temp_file)
        
        # Create tasks with dependencies
        task1 = Task(title="Setup database")
        task2 = Task(title="Create API", dependencies=[task1.id])
        
        await repo.save(task1)
        await repo.save(task2)
        
        retrieved = await repo.get_by_id(task2.id)
        assert len(retrieved.dependencies) == 1
        assert retrieved.dependencies[0] == task1.id