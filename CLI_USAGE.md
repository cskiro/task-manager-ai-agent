# Task Manager AI Agent - CLI Usage

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/task-manager-ai-agent.git
cd task-manager-ai-agent

# Install dependencies
poetry install

# Copy environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Basic Usage

### Plan a Project

```bash
# Using OpenAI (requires API key in .env)
poetry run python -m src plan "Build a blog platform with user authentication"

# With additional context
poetry run python -m src plan "Create an e-commerce site" \
  --context "Using Next.js, Stripe, and PostgreSQL"

# Using mock provider (for testing without API key)
poetry run python -m src plan "Build a mobile app" --mock

# Using a specific model
poetry run python -m src plan "Create a SaaS platform" \
  --model gpt-3.5-turbo
```

### CLI Options

- `DESCRIPTION`: Project description (required)
- `--context, -c`: Additional context for better planning
- `--mock`: Use mock LLM provider (no API key needed)
- `--model, -m`: OpenAI model to use (default: gpt-4-turbo-preview)

## Output Format

The CLI provides rich, formatted output including:

1. **Task Table**: Shows all tasks with IDs, priorities, time estimates, and dependencies
2. **Dependency Tree**: Visual representation of task relationships
3. **Total Time**: Sum of all task estimates

## Example Output

```
Planning project: Build a blog platform

✓ Created 6 tasks for project

┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ ID ┃ Task                     ┃ Priority ┃ Est. Hours ┃ Dependencies ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 1  │ Set up project structure │ 🟡       │ 3.0h       │ -            │
│ 2  │ Design database schema   │ 🟡       │ 6.0h       │ #1           │
│ 3  │ Implement API endpoints  │ 🟡       │ 12.0h      │ #2           │
└────┴──────────────────────────┴──────────┴────────────┴──────────────┘

Total estimated time: 21.0 hours
```

## Tips

1. **Be specific**: More detailed descriptions yield better task breakdowns
2. **Use context**: Provide technology stack and requirements for accurate planning
3. **Start with mock**: Test your prompts with `--mock` before using API credits
4. **Model selection**: Use `gpt-3.5-turbo` for faster/cheaper results, `gpt-4` for best quality

## Troubleshooting

- **"OPENAI_API_KEY not found"**: Create `.env` file with your API key
- **Rate limits**: Add delays between multiple runs or upgrade your OpenAI plan
- **Poor results**: Try adding more context or using a better model

## Future Features

- [ ] Export to JSON/Markdown/CSV
- [ ] Interactive mode for refining plans
- [ ] Integration with project management tools
- [ ] Time tracking and progress updates