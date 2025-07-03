# 🤖 Task Manager AI Agent

[![CI](https://github.com/YOUR_USERNAME/task-manager-ai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/task-manager-ai-agent/actions/workflows/ci.yml)

An intelligent AI agent that helps break down complex projects into manageable tasks using modern LLMs.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/task-manager-ai-agent.git
cd task-manager-ai-agent

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run the AI agent with mock provider
poetry run python -m src.interface.cli agent plan "Build a web application" --mock
```

## 🏗️ Architecture

This project follows clean architecture principles:
- **Domain Layer**: Core business logic and entities
- **Application Layer**: Use cases and business rules
- **Infrastructure Layer**: External services (LLMs, storage)
- **Interface Layer**: CLI, API endpoints

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src

# Run only unit tests
poetry run pytest tests/unit/

# Run only integration tests
poetry run pytest tests/integration/ -m integration
```

## 📖 Documentation

See the [docs](./docs) directory for detailed documentation:
- [Development Checklist](./docs/development-checklist.md)
- [Architecture Guide](./docs/ai-agent-task-manager-project-instructions.md)
- [Development Roadmap](./docs/optimal-ai-agent-development-roadmap.md)

## 📝 License

MIT License - see [LICENSE](./LICENSE) file for details.