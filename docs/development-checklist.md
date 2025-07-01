# 📋 Task Manager AI Agent - Development Checklist

Based on analysis of project documentation and lessons learned from previous attempts (avoiding over-engineering), this checklist provides a focused, sequential approach that starts with GitHub configuration and follows TDD principles.

## 🔧 Phase 1: Repository Setup (Day 1)

- [ ] **Configure GitHub Repository** 
  - [ ] Enable branch protection rules for `main`
  - [ ] Require 2 PR approvers and passing status checks
  - [ ] Disable force pushes and deletions

- [ ] **Set Up GitHub Actions CI/CD**
  - [ ] Create `.github/workflows/ci.yml` for automated testing
  - [ ] Configure to run on all PRs to main/develop branches
  - [ ] Include pytest, type checking (mypy), and linting (ruff)

- [ ] **Configure Repository Secrets**
  - [ ] Add `OPENAI_API_KEY` for integration tests
  - [ ] Add `ANTHROPIC_API_KEY` as backup provider
  - [ ] Set up Codecov token for coverage reports

## 🏗️ Phase 2: Development Environment (Day 1-2)

- [ ] **Initialize Poetry Project**
  ```bash
  poetry init
  poetry add pydantic pytest python-dotenv openai anthropic
  poetry add --dev pytest-asyncio pytest-cov mypy ruff black
  ```

- [ ] **Create Clean Architecture Structure**
  ```
  src/
  ├── domain/         # ✅ Already complete (Phase 1)
  ├── application/    # 🚧 Current focus (Phase 2)
  │   ├── ports/
  │   └── use_cases/
  └── tests/
      ├── unit/
      └── integration/
  ```

## 🧪 Phase 3: TDD Application Layer (Week 1)

- [ ] **Write First Failing Test** (RED)
  ```python
  # tests/unit/application/test_plan_project_use_case.py
  async def test_plan_project_creates_tasks():
      # Arrange: Mock LLM provider
      # Act: Call use case
      # Assert: Returns valid project plan
  ```

- [ ] **Implement Minimal Use Case** (GREEN)
  - [ ] Create `PlanProjectUseCase` with just enough code to pass
  - [ ] No fancy features, just basic project → tasks conversion

- [ ] **Define LLM Port Interface** (REFACTOR)
  ```python
  # application/ports/llm_provider.py
  class LLMProvider(Protocol):
      async def complete(self, prompt: str) -> str: ...
  ```

- [ ] **Create Mock Provider for Testing**
  - [ ] Predictable responses for unit tests
  - [ ] No real API calls during testing

- [ ] **Integration Test with Real AI**
  - [ ] One test that actually calls OpenAI
  - [ ] Validates the "magic" actually works

## 🚀 Phase 4: MVP Validation (Week 1-2)

- [ ] **Build Minimal Demo Script**
  ```python
  # scripts/demo_planning.py
  # 50 lines max - just prove AI planning works
  ```

- [ ] **Get Early User Feedback**
  - [ ] Deploy to 3 trusted users
  - [ ] Gather feedback before adding ANY complexity
  - [ ] Validate the core value proposition

## 🎯 Key Success Criteria

- ✅ GitHub CI/CD runs on every PR
- ✅ All tests pass (domain + new application tests)
- ✅ Demo script successfully plans a real project
- ✅ 3 users confirm the AI planning is valuable

## ⚠️ Anti-Patterns to Avoid

- ❌ Don't build a web UI yet
- ❌ Don't add multiple LLM providers yet  
- ❌ Don't implement complex scheduling algorithms
- ❌ Don't create elaborate configuration systems
- ❌ Keep it under 500 lines of code total

## 📊 Progress Tracking

After each task completion, run:
```bash
poetry run pytest
poetry run mypy src/
poetry run python scripts/demo_planning.py "Build a blog platform"
```

## 📝 Progress Notes

### Day 1 (Date: ______)
- [ ] Completed Phase 1 setup
- Notes: 

### Day 2 (Date: ______)
- [ ] Completed Phase 2 environment setup
- Notes:

### Week 1 Review (Date: ______)
- [ ] Phase 3 TDD implementation complete
- [ ] First demo working
- Key learnings:

### MVP Review (Date: ______)
- [ ] User feedback collected from 3 testers
- [ ] Core value proposition validated
- Next steps:

---

This checklist follows the **"Start with the magic, not the machinery"** principle from the optimal roadmap while respecting the clean architecture foundation already built. The key is to prove the AI planning delivers value before adding any infrastructure complexity.

**Remember**: Ship working software in 2 days, not 2 weeks!