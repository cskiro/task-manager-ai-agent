# Domain Layer

This layer contains the core business logic and entities of the Task Manager AI Agent.

## Structure

- **entities/**: Core domain entities (Task, Project, Agent)
- **value_objects/**: Immutable value objects (Priority, TaskStatus, TimeEstimate)
- **services/**: Domain services for complex business logic

## Key Concepts

### Task Entity
The central entity representing a unit of work with:
- Lifecycle management (pending → in_progress → completed)
- Dependency tracking
- Time estimation using PERT method
- Priority levels

### Value Objects
- **Priority**: 5-level priority system with emoji representation
- **TaskStatus**: State machine for task lifecycle
- **TimeEstimate**: Three-point estimation (optimistic, realistic, pessimistic)

## Usage Example

```python
from src.domain.entities import Task
from src.domain.value_objects import Priority, TimeEstimate

# Create a task
task = Task(
    title="Implement user authentication",
    description="Add JWT-based auth system",
    priority=Priority.HIGH,
    time_estimate=TimeEstimate.from_hours(
        optimistic=4.0,
        realistic=8.0,
        pessimistic=12.0
    )
)

# Start the task
task.start()

# Complete the task
task.complete()
```

## Design Principles

1. **Rich Domain Model**: Entities contain business logic, not just data
2. **Immutable Value Objects**: Ensure data integrity
3. **No External Dependencies**: Pure Python, no framework dependencies
4. **Comprehensive Validation**: Business rules enforced at domain level