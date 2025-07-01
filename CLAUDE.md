## Task Manager AI Agent Project Guidelines

### Technology Stack & Architecture

This project demonstrates modern Python development with clean architecture principles:

**Core Technologies:**
- **Python 3.12** with type hints and dataclasses for robust development
- **Poetry** for dependency management and virtual environments
- **Pytest** with async support for comprehensive testing
- **Pydantic** for data validation and settings management
- **Rich** for beautiful CLI interfaces
- **OpenAI/Anthropic SDKs** for LLM integration (pluggable)

**Key Architectural Patterns:**
- **Clean Architecture** with domain-driven design
- **Hexagonal Architecture** with ports and adapters
- **Dependency Injection** for testability and flexibility
- **Repository Pattern** for data persistence abstraction
- **Domain Services** for business logic encapsulation
- **Value Objects** for type safety and immutability

### Quick Start Guide

Get productive with Claude in 5 minutes:

1. **Start with Context**
   ```
   I'm working on a **modern Python AI agent application** that uses clean architecture principles. This app helps users break down complex projects into manageable tasks using AI-powered planning and dependency analysis.

   Current challenge: [Describe your specific problem]

   **Architecture Context:**
   - Domain Layer: Entities (Task, Project, Agent), Value Objects (Priority, TaskStatus, TimeEstimate), Services
   - Application Layer: Use Cases, Ports (interfaces for LLM, storage, memory)
   - Infrastructure Layer: LLM providers (OpenAI, Anthropic, Mock), persistence, memory systems
   - Interface Layer: CLI, API endpoints (future)

   **Phase Status:**
   - ✅ Phase 1: Domain Layer (Complete - 27 tests passing)
   - 🚧 Phase 2: Application Layer (In Progress)
   - 📋 Phase 3: Infrastructure Layer (Planned)
   - 📋 Phase 4: Interface Layer (Planned)
   ```

2. **Be Specific About Requirements**
   ```
   Create a Python component that:
   - Follows our clean architecture patterns (domain/application/infrastructure)
   - Uses dependency injection with protocols/interfaces
   - Includes comprehensive type hints and docstrings
   - Has unit tests with pytest and realistic test scenarios
   - Follows our error handling conventions
   ```

3. **Iterate and Refine**
   - Start with a working solution that fits the architecture
   - Ask for improvements: "Can you optimize this for testability?"
   - Request alternatives: "Show me another approach using composition"

4. **Save Successful Patterns**
   - Document prompts that work well for clean architecture
   - Build a library of domain modeling patterns
   - Share architectural decisions with the team

---

## Core Capabilities

### Code Generation and Review

Claude excels at both generating new code and reviewing existing code within our clean architecture.

#### Code Generation Best Practices

1. **Provide Architectural Context**
   ```
   Using our clean architecture (Domain/Application/Infrastructure layers),
   create a new use case that handles project planning with dependency injection.
   Follow our established patterns:
   - Domain entities with business logic
   - Application use cases that orchestrate
   - Ports (protocols) for external dependencies
   - Immutable value objects with validation
   ```

2. **Include Domain Context**
   ```
   In our task management domain, create a new entity that:
   - Extends our base entity patterns
   - Uses value objects for type safety
   - Implements domain events if needed
   - Includes comprehensive validation
   - Follows our naming conventions
   ```

3. **Request Specific Patterns**
   ```
   Implement this using our established patterns:
   - Repository pattern with protocols
   - Domain service for complex business logic
   - Use case with dependency injection
   - Value objects for type safety
   ```

### Project-Specific Development Guidelines

#### 1. Domain Modeling Pattern

When working with domain entities, follow the established pattern:

```python
@dataclass
class Task:
    """
    Task entity with rich domain behavior.
    
    Represents a unit of work with lifecycle management,
    dependencies, and time estimation.
    """
    
    id: UUID = field(default_factory=uuid4)
    title: str
    description: str = ""
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[UUID] = field(default_factory=list)
    time_estimate: Optional[TimeEstimate] = None
    
    def can_start(self, completed_tasks: set[UUID]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def start(self) -> None:
        """Mark task as started with validation."""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot start task in {self.status.value} status")
        self.status = TaskStatus.IN_PROGRESS
```

**Prompt for Domain Modeling:**
```
I need to add a new domain entity to the Task Manager AI Agent.

Current domain architecture:
- Entities: Task, Project, Agent (rich domain models with behavior)
- Value Objects: Priority, TaskStatus, TimeEstimate (immutable, validated)
- Services: TaskDecomposer, DependencyAnalyzer, TaskScheduler
- Repository interfaces in application layer

Please implement [entity name] following these patterns:
1. Use dataclasses with type hints
2. Include domain behavior methods
3. Use value objects for type safety
4. Add comprehensive validation
5. Include docstrings and examples
6. Write unit tests with realistic scenarios
```

#### 2. Application Layer Architecture

Follow the hexagonal architecture with use cases and ports:

```python
class PlanProjectUseCase:
    """
    Use case for AI-powered project planning.
    
    Orchestrates domain services and external dependencies
    through dependency injection.
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        task_repository: TaskRepository,
        decomposer: TaskDecomposer
    ):
        self.llm = llm_provider
        self.repo = task_repository
        self.decomposer = decomposer
    
    async def execute(
        self,
        request: PlanProjectRequest
    ) -> PlanProjectResponse:
        """Execute the project planning use case."""
        # 1. Decompose project using domain service + LLM
        tasks = await self.decomposer.decompose(
            request.description,
            self.llm
        )
        
        # 2. Persist tasks through repository
        for task in tasks:
            await self.repo.save(task)
        
        # 3. Return response
        return PlanProjectResponse(
            project_id=request.project_id,
            tasks=tasks,
            summary=f"Created {len(tasks)} tasks"
        )
```

**Prompt for Application Layer:**
```
Create a new use case for the Task Manager AI Agent following our patterns:

Current application architecture:
- Use cases orchestrate domain logic with external dependencies
- Ports (protocols) define interfaces for infrastructure
- Dependency injection for testability
- Request/Response objects for clear contracts
- Error handling with domain exceptions

Use case requirements: [describe the business operation]

Please implement:
1. Use case class with dependency injection
2. Port interfaces for external dependencies
3. Request/Response objects with Pydantic validation
4. Comprehensive error handling
5. Unit tests with mocked dependencies
6. Integration points with existing domain services
```

#### 3. Testing Strategy

Maintain high test coverage with realistic domain scenarios:

```python
class TestTaskLifecycle:
    """Test task entity behavior with realistic scenarios."""
    
    def test_task_creation_with_dependencies(self):
        """Test creating a task with complex dependencies."""
        task = Task(
            title="Implement user authentication",
            description="Add JWT-based authentication system",
            priority=Priority.HIGH,
            time_estimate=TimeEstimate.from_hours(
                optimistic=4.0,
                realistic=8.0,
                pessimistic=12.0
            )
        )
        
        assert task.status == TaskStatus.PENDING
        assert not task.can_start(set())  # No dependencies completed
    
    def test_project_dependency_analysis(self):
        """Test complex project with circular dependency detection."""
        project = Project("E-commerce Platform")
        
        # Create tasks with dependencies
        auth_task = project.add_task("Authentication", dependencies=[])
        db_task = project.add_task("Database Setup", dependencies=[])
        api_task = project.add_task("API Layer", dependencies=[auth_task.id, db_task.id])
        
        analyzer = DependencyAnalyzer()
        cycles = analyzer.find_cycles(project)
        
        assert len(cycles) == 0
        assert analyzer.get_critical_path(project)[0] == api_task
```

**Prompt for Testing:**
```
Write comprehensive tests for this Task Manager AI Agent component:

Current testing patterns:
- Domain-focused test scenarios (realistic business cases)
- Property-based testing for value objects
- Integration tests for use cases with mocked ports
- Performance tests for large datasets (1000+ tasks)
- Error scenario coverage (invalid states, network failures)

Component to test: [component name and description]

Please include:
1. Unit tests for individual methods
2. Integration tests for workflows
3. Edge cases and error conditions
4. Performance tests if applicable
5. Property-based tests for value objects
6. Mock strategies for external dependencies
```

#### 4. Infrastructure Integration

Implement clean interfaces for external dependencies:

```python
class LLMProvider(Protocol):
    """Protocol for LLM service providers."""
    
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """Get completion from LLM."""
        ...
    
    async def structured_complete(
        self,
        prompt: str,
        response_model: type[BaseModel],
        **kwargs
    ) -> BaseModel:
        """Get structured response from LLM."""
        ...

class OpenAIProvider:
    """OpenAI implementation of LLM provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
```

**Prompt for Infrastructure:**
```
Implement infrastructure adapter for the Task Manager AI Agent:

Current infrastructure patterns:
- Protocol-based interfaces (ports) in application layer
- Concrete implementations in infrastructure layer
- Dependency injection container for wiring
- Configuration-driven provider selection
- Error handling with retry logic and circuit breakers

Infrastructure component: [describe the external service]

Please implement:
1. Protocol interface in application/ports/
2. Concrete implementation in infrastructure/
3. Configuration management with Pydantic
4. Error handling and retry logic
5. Unit tests for the adapter
6. Integration tests with real/mock services
7. Documentation for configuration options
```

#### 5. Development Workflow

Use these established patterns for development:

```bash
# Development commands
poetry install                    # Install dependencies
poetry run pytest               # Run all tests
poetry run pytest tests/unit/   # Run unit tests only
poetry run pytest --cov=src     # Run with coverage
poetry run python scripts/demo_domain.py  # Demo domain layer

# Testing specific layers
poetry run pytest tests/unit/domain/      # Domain tests
poetry run pytest tests/integration/      # Integration tests
poetry run pytest -k "test_task"         # Run specific tests

# Code quality
poetry run mypy src/             # Type checking
poetry run black src/ tests/    # Code formatting
poetry run ruff check src/      # Linting
```

**Prompt for Development:**
```
I need to [develop/debug/refactor] a component in the Task Manager AI Agent:

Current development setup:
- Poetry for dependency management
- Pytest with coverage reporting
- MyPy for static type checking
- Black for code formatting
- Ruff for linting
- Scripts in scripts/ for domain demos and testing

Development task: [describe what you need to do]

Project context:
- Clean architecture with domain/application/infrastructure layers
- 27 domain tests currently passing
- Working on Phase 2: Application layer implementation
- Python 3.12 with comprehensive type hints

Please provide:
1. Implementation approach
2. Testing strategy
3. Integration points with existing code
4. Commands to verify the changes
```

### Common Prompt Templates

#### Feature Implementation Template
```
Implement a new feature for the Task Manager AI Agent:

**Architecture Context:**
- Clean Architecture: Domain → Application → Infrastructure → Interface
- Current Status: Phase 1 (Domain) complete, Phase 2 (Application) in progress
- Domain entities: Task, Project, Agent with rich behavior
- Value objects: Priority, TaskStatus, TimeEstimate for type safety
- Services: TaskDecomposer, DependencyAnalyzer, TaskScheduler

**Feature:** [Name and description]

**Requirements:**
- Follow clean architecture boundaries
- Use dependency injection with protocols
- Implement comprehensive type hints
- Add thorough test coverage (unit + integration)
- Include proper error handling
- Consider performance for large datasets
- Document configuration options

**Integration Points:**
- Domain: [relevant entities and services]
- Application: [use cases and ports]
- Infrastructure: [external services]
- Interface: [CLI/API endpoints]
```

#### Bug Fix Template
```
Fix this issue in the Task Manager AI Agent:

**Architecture Context:**
Clean architecture with [relevant layer] where issue occurs
Related components: [list relevant files and classes]

**Issue:** [Detailed description]
**Expected:** [What should happen]
**Actual:** [What's happening]
**Reproduction:** [Steps to reproduce]

**Investigation Guidelines:**
1. Check type hints and MyPy errors first
2. Verify domain invariants are maintained
3. Ensure proper dependency injection
4. Test with realistic data scenarios
5. Check error propagation through layers
6. Validate test coverage for the code path

**Logs/Errors:** [Relevant error messages]
**Tests:** [Failing test cases]
```

#### Architecture Review Template
```
Review this Task Manager AI Agent implementation:

**Review Focus:**
- [ ] Clean architecture boundaries respected
- [ ] Domain logic in appropriate layer
- [ ] Proper use of dependency injection
- [ ] Type safety and comprehensive hints
- [ ] Error handling and propagation
- [ ] Test coverage and realistic scenarios
- [ ] Performance considerations
- [ ] Documentation and docstrings
- [ ] Configuration management
- [ ] Integration with existing patterns

**Code:** [Paste implementation]

**Architecture Layer:** [Domain/Application/Infrastructure/Interface]
**Specific Questions:** [Any particular concerns]
```

#### Performance Optimization Template
```
Optimize this component for the Task Manager AI Agent:

**Current Performance Patterns:**
- Async/await for I/O operations
- Dataclasses for efficient memory usage
- Generator patterns for large datasets
- Caching strategies for expensive operations
- Batch processing for repository operations

**Component:** [Description]
**Performance Requirements:** [Specific needs]
**Current Bottlenecks:** [Known issues]

**Optimization Focus:**
- [ ] Memory usage optimization
- [ ] I/O operation efficiency
- [ ] Algorithm complexity improvement
- [ ] Caching strategy implementation
- [ ] Batch processing patterns
```

---

## Domain Model Reference

### Core Entities
- **Task**: Work unit with lifecycle, dependencies, time estimation
- **Project**: Container for tasks with dependency management
- **Agent**: Domain model for AI agents with capabilities

### Value Objects
- **Priority**: Enum (LOW=1 to URGENT=5) with emoji display
- **TaskStatus**: State machine (PENDING → IN_PROGRESS → COMPLETED)
- **TimeEstimate**: PERT-based estimation (optimistic/realistic/pessimistic)

### Domain Services
- **TaskDecomposer**: Breaks down project descriptions into tasks
- **DependencyAnalyzer**: Finds cycles, bottlenecks, critical paths
- **TaskScheduler**: Critical path method, parallel task detection

### Test Coverage
- ✅ 27 domain tests passing
- ✅ Value objects: Priority, TaskStatus, TimeEstimate
- ✅ Entities: Task lifecycle, dependencies, validation
- ✅ Services: Decomposition, analysis, scheduling

---

*This guide is a living document. Update it as the architecture evolves through phases. Last updated: June 2025*