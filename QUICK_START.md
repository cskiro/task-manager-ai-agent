# 🚀 Task Manager AI - Quick Start Guide

## Installation (2 minutes)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/task-manager-ai-agent.git
   cd task-manager-ai-agent
   ```

2. **Install with Poetry:**
   ```bash
   # Install Poetry if you don't have it
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install
   ```

3. **Set up your API key (optional but recommended):**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # Get one at: https://platform.openai.com/api-keys
   ```
   
   **Note:** The app works without an API key! It will automatically use mock mode for testing.

## Your First Project Plan (30 seconds)

### Option 1: Interactive Welcome (Recommended!)
```bash
poetry run task-manager
```
This launches our beautiful welcome wizard with example projects.

### Option 2: Direct Planning
```bash
# With mock data (no API key needed)
poetry run task-manager plan "Build a blog platform" --mock

# With real AI
poetry run task-manager plan "Build a blog platform"
```

### Option 3: Use Example Projects
```bash
poetry run task-manager plan --example 1  # E-commerce
poetry run task-manager plan --example 2  # Mobile app
poetry run task-manager plan --example 3  # Discord bot
poetry run task-manager plan --example 4  # Game MVP
```

## Export Your Plan

After planning, export in your preferred format:

```bash
# Interactive export (easiest!)
# You'll be prompted after planning completes

# Manual export
poetry run task-manager export markdown  # For documentation
poetry run task-manager export csv       # For spreadsheets
poetry run task-manager export json      # For developers
```

## Pro Tips

1. **Enable Poetry Shell** for shorter commands:
   ```bash
   poetry shell
   task-manager plan "Your project"  # No 'poetry run' needed!
   ```

2. **Use Advanced Features:**
   ```bash
   # Web research for better plans
   task-manager agent plan "Build SaaS" --tools
   
   # Time and cost estimation
   task-manager agent plan "Mobile app" --estimate
   ```

3. **Save and Resume:**
   ```bash
   # Save your work
   task-manager save myproject.json
   
   # Resume later
   task-manager load myproject.json
   ```

## Troubleshooting

- **"task-manager: command not found"** → Use `poetry run task-manager`
- **"No API key"** → Use `--mock` flag or set OPENAI_API_KEY in .env
- **Need help?** → Run `task-manager --help`

---

Ready to be amazed? Run `poetry run task-manager` now! 🎉