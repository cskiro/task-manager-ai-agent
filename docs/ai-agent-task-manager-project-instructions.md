# Comprehensive Guide to Building Modern AI Agents (2025)

## 1. Foundation & Prerequisites

### Essential Concepts and Terminology

**AI Agent**: An autonomous system that perceives its environment, makes decisions, and takes actions to achieve specific goals. Modern agents leverage LLMs for reasoning and decision-making.

**Key Concepts**:
- **Reasoning Loop**: The cyclic process of observation → thinking → action → observation
- **Tool Use**: Agents' ability to invoke external functions/APIs to extend capabilities
- **Memory Systems**: Mechanisms for storing and retrieving information across interactions
- **Prompt Chaining**: Sequential prompting to break complex tasks into manageable steps
- **Context Management**: Efficiently managing token limits and relevant information

### Required Technical Skills and Tools

**Technical Prerequisites**:
- Python 3.10+ proficiency
- Understanding of async/await patterns
- Basic knowledge of REST APIs
- Familiarity with LLM APIs (OpenAI, Anthropic, etc.)
- Version control (Git)

**Development Environment**:
```bash
# Recommended Python environment setup
python -m venv agent_env
source agent_env/bin/activate  # On Windows: agent_env\Scripts\activate

# Core dependencies
pip install langchain>=0.2.0
pip install openai>=1.30.0
pip install pydantic>=2.0
pip install python-dotenv
pip install httpx
pip install tenacity  # For retry logic
```

### Modern Frameworks and Libraries (2025)

**Primary Frameworks**:
1. **LangChain** (v0.2+): Comprehensive framework for LLM applications
2. **AutoGen** (Microsoft): Multi-agent conversation framework
3. **CrewAI**: Role-based agent orchestration
4. **Haystack**: Production-ready NLP framework
5. **Semantic Kernel** (Microsoft): Enterprise-focused agent framework

**Supporting Libraries**:
- **LlamaIndex**: Advanced RAG and memory systems
- **Instructor**: Structured output parsing
- **Guidance**: Constrained generation
- **DSPy**: Declarative self-improving agents

## 2. Agent Architecture Template

### Core Components of a Modern AI Agent

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentMemory:
    """Structured memory storage"""
    short_term: List[Dict[str, Any]]
    long_term: Dict[str, Any]
    episodic: List[Dict[str, Any]]
    
    def add_observation(self, observation: Dict[str, Any]):
        """Add new observation to short-term memory"""
        self.short_term.append({
            "timestamp": datetime.now().isoformat(),
            "content": observation
        })
        # Maintain memory size limits
        if len(self.short_term) > 10:
            self.short_term.pop(0)

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, model: str = "gpt-4"):
        self.name = name
        self.model = model
        self.state = AgentState.IDLE
        self.memory = AgentMemory(
            short_term=[],
            long_term={},
            episodic=[]
        )
        self.tools = {}
        self.max_iterations = 10
        
    @abstractmethod
    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation and decide next action"""
        pass
    
    @abstractmethod
    async def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decided action"""
        pass
    
    def register_tool(self, name: str, func: callable):
        """Register a tool for the agent to use"""
        self.tools[name] = func
    
    async def run(self, initial_input: str) -> Any:
        """Main execution loop"""
        self.state = AgentState.THINKING
        observation = {"input": initial_input, "iteration": 0}
        
        for i in range(self.max_iterations):
            try:
                # Think phase
                thought = await self.think(observation)
                self.memory.add_observation(thought)
                
                # Check if task is complete
                if thought.get("is_complete", False):
                    self.state = AgentState.COMPLETED
                    return thought.get("result")
                
                # Act phase
                self.state = AgentState.ACTING
                action_result = await self.act(thought)
                
                # Update observation for next iteration
                observation = {
                    "previous_thought": thought,
                    "action_result": action_result,
                    "iteration": i + 1
                }
                
            except Exception as e:
                self.state = AgentState.ERROR
                return {"error": str(e)}
        
        return {"error": "Max iterations reached"}
```

### Decision-Making Mechanisms

```python
from langchain.schema import BasePromptTemplate
from langchain.prompts import PromptTemplate

class ReActAgent(BaseAgent):
    """Reasoning and Acting agent implementation"""
    
    def __init__(self, name: str, model: str = "gpt-4"):
        super().__init__(name, model)
        self.reasoning_prompt = PromptTemplate(
            input_variables=["observation", "tools", "memory"],
            template="""You are {agent_name}, an AI agent that can reason and act.

Current observation: {observation}

Available tools: {tools}

Recent memory: {memory}

Based on the observation, reason step by step about what to do next.
Format your response as:
Thought: [Your reasoning about the current situation]
Action: [tool_name]
Action Input: [input for the tool]

If the task is complete, respond with:
Thought: [Final reasoning]
Answer: [Final answer]
"""
        )
    
    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Implement ReAct reasoning"""
        # Format recent memory
        recent_memory = self.memory.short_term[-3:] if self.memory.short_term else []
        
        # Create prompt
        prompt = self.reasoning_prompt.format(
            agent_name=self.name,
            observation=observation,
            tools=list(self.tools.keys()),
            memory=recent_memory
        )
        
        # Call LLM (placeholder - implement with actual LLM call)
        response = await self._call_llm(prompt)
        
        # Parse response
        return self._parse_reasoning(response)
    
    def _parse_reasoning(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        lines = response.strip().split('\n')
        result = {}
        
        for line in lines:
            if line.startswith("Thought:"):
                result["thought"] = line[8:].strip()
            elif line.startswith("Action:"):
                result["action"] = line[7:].strip()
            elif line.startswith("Action Input:"):
                result["action_input"] = line[13:].strip()
            elif line.startswith("Answer:"):
                result["answer"] = line[7:].strip()
                result["is_complete"] = True
        
        return result
```

### Memory Systems Implementation

```python
import json
from typing import List, Optional
import numpy as np
from datetime import datetime, timedelta

class VectorMemory:
    """Long-term memory with vector similarity search"""
    
    def __init__(self, embedding_model="text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.memories = []
        self.embeddings = []
    
    async def store(self, content: str, metadata: Dict[str, Any] = None):
        """Store memory with embedding"""
        embedding = await self._get_embedding(content)
        self.memories.append({
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        })
        self.embeddings.append(embedding)
    
    async def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve k most similar memories"""
        if not self.memories:
            return []
        
        query_embedding = await self._get_embedding(query)
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            memory = self.memories[idx].copy()
            memory["similarity"] = float(similarities[idx])
            memory["access_count"] += 1
            results.append(memory)
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: List[np.ndarray]) -> np.ndarray:
        """Calculate cosine similarity"""
        b_array = np.array(b)
        dot_product = np.dot(b_array, a)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b_array, axis=1)
        return dot_product / (norm_a * norm_b)

class EpisodicMemory:
    """Memory for storing complete task episodes"""
    
    def __init__(self, max_episodes: int = 100):
        self.max_episodes = max_episodes
        self.episodes = []
    
    def store_episode(self, episode: Dict[str, Any]):
        """Store a complete task episode"""
        self.episodes.append({
            "timestamp": datetime.now().isoformat(),
            "task": episode.get("task"),
            "steps": episode.get("steps", []),
            "outcome": episode.get("outcome"),
            "duration": episode.get("duration"),
            "success": episode.get("success", False)
        })
        
        # Maintain size limit
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
    
    def get_similar_episodes(self, task: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve episodes similar to current task"""
        # Simple keyword matching (can be enhanced with embeddings)
        task_words = set(task.lower().split())
        scored_episodes = []
        
        for episode in self.episodes:
            episode_words = set(episode["task"].lower().split())
            similarity = len(task_words & episode_words) / len(task_words | episode_words)
            scored_episodes.append((similarity, episode))
        
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep[1] for ep in scored_episodes[:limit]]
```

## 3. Step-by-Step Implementation Guide

### Environment Setup and Dependencies

```python
# requirements.txt
langchain>=0.2.0
openai>=1.30.0
anthropic>=0.25.0
pydantic>=2.0
python-dotenv
httpx
tenacity
numpy
tiktoken
chromadb  # For vector storage
redis  # For distributed memory
```

### Basic Agent Scaffold Code

```python
import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class SimpleAgent:
    """A simple but effective agent implementation"""
    
    def __init__(
        self,
        name: str,
        description: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7
    ):
        self.name = name
        self.description = description
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tools = []
        self.memory = []
        
    def add_tool(self, func: callable, name: str, description: str):
        """Add a tool to the agent's toolkit"""
        tool = Tool(
            name=name,
            func=func,
            description=description
        )
        self.tools.append(tool)
        logger.info(f"Added tool: {name}")
    
    async def think_and_act(self, task: str) -> str:
        """Main reasoning and action loop"""
        system_prompt = f"""You are {self.name}, {self.description}
        
You have access to the following tools:
{self._format_tools()}

To use a tool, respond with:
TOOL: tool_name
INPUT: tool_input

When you have a final answer, respond with:
FINAL ANSWER: your_answer

Think step by step and explain your reasoning."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        max_steps = 5
        for step in range(max_steps):
            # Get LLM response
            response = await self.llm.agenerate([messages])
            content = response.generations[0][0].text
            
            # Add to conversation history
            messages.append({"role": "assistant", "content": content})
            
            # Parse response
            if "FINAL ANSWER:" in content:
                return content.split("FINAL ANSWER:")[1].strip()
            elif "TOOL:" in content and "INPUT:" in content:
                # Extract tool call
                tool_name = content.split("TOOL:")[1].split("INPUT:")[0].strip()
                tool_input = content.split("INPUT:")[1].strip()
                
                # Execute tool
                tool_result = await self._execute_tool(tool_name, tool_input)
                
                # Add result to conversation
                messages.append({
                    "role": "user", 
                    "content": f"Tool '{tool_name}' returned: {tool_result}"
                })
            else:
                # Continue reasoning
                messages.append({
                    "role": "user",
                    "content": "Please continue with your task. Use a tool or provide a final answer."
                })
        
        return "Maximum steps reached without completing the task."
    
    def _format_tools(self) -> str:
        """Format tools for prompt"""
        return "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
    
    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = await tool.arun(tool_input)
                    return str(result)
                except Exception as e:
                    return f"Error executing tool: {str(e)}"
        return f"Tool '{tool_name}' not found"
```

### Adding Reasoning Capabilities

```python
from enum import Enum
from typing import Optional, List, Dict, Any
import json

class ReasoningStrategy(Enum):
    """Different reasoning strategies for agents"""
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHT = "tot"
    REFLEXION = "reflexion"
    SELF_CONSISTENCY = "self_consistency"

class AdvancedReasoningAgent(SimpleAgent):
    """Agent with advanced reasoning capabilities"""
    
    def __init__(
        self,
        name: str,
        description: str,
        reasoning_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        **kwargs
    ):
        super().__init__(name, description, **kwargs)
        self.reasoning_strategy = reasoning_strategy
        self.reasoning_traces = []
    
    async def reason_with_cot(self, task: str) -> Dict[str, Any]:
        """Chain of Thought reasoning"""
        cot_prompt = f"""Task: {task}

Let's think about this step by step:
1) First, I need to understand what's being asked
2) Then, I'll identify what information or tools I need
3) Next, I'll plan my approach
4) Finally, I'll execute and verify the result

My reasoning:"""
        
        response = await self.llm.agenerate([[
            {"role": "system", "content": "You are a logical reasoning assistant."},
            {"role": "user", "content": cot_prompt}
        ]])
        
        reasoning = response.generations[0][0].text
        self.reasoning_traces.append({
            "strategy": "chain_of_thought",
            "task": task,
            "reasoning": reasoning
        })
        
        return {"reasoning": reasoning, "next_action": self._extract_next_action(reasoning)}
    
    async def reason_with_tot(self, task: str, branches: int = 3) -> Dict[str, Any]:
        """Tree of Thought reasoning - explore multiple reasoning paths"""
        tot_prompt = f"""Task: {task}

Generate {branches} different approaches to solve this task.
For each approach, think through the pros and cons.

Format as:
Approach 1: [description]
Pros: [list pros]
Cons: [list cons]
Viability: [score 1-10]"""
        
        response = await self.llm.agenerate([[
            {"role": "system", "content": "You are a strategic planning assistant."},
            {"role": "user", "content": tot_prompt}
        ]])
        
        approaches = response.generations[0][0].text
        
        # Evaluate best approach
        eval_prompt = f"""Given these approaches:
{approaches}

Which approach is most likely to succeed and why?"""
        
        eval_response = await self.llm.agenerate([[
            {"role": "user", "content": eval_prompt}
        ]])
        
        return {
            "approaches": approaches,
            "selected_approach": eval_response.generations[0][0].text
        }
    
    async def reason_with_reflexion(self, task: str, attempt: str, outcome: str) -> Dict[str, Any]:
        """Reflexion - learn from previous attempts"""
        reflexion_prompt = f"""Task: {task}
Previous attempt: {attempt}
Outcome: {outcome}

Reflect on what went wrong and how to improve:
1) What specific mistakes were made?
2) What assumptions were incorrect?
3) What should be done differently?
4) What's the improved approach?"""
        
        response = await self.llm.agenerate([[
            {"role": "system", "content": "You are a self-improving AI assistant."},
            {"role": "user", "content": reflexion_prompt}
        ]])
        
        reflection = response.generations[0][0].text
        return {
            "reflection": reflection,
            "improved_approach": self._extract_improved_approach(reflection)
        }
    
    def _extract_next_action(self, reasoning: str) -> Optional[str]:
        """Extract next action from reasoning text"""
        # Simple extraction logic - can be enhanced
        if "need to" in reasoning.lower():
            return reasoning.split("need to")[1].split(".")[0].strip()
        return None
```

### Implementing Tool Usage

```python
import httpx
import json
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential

class ToolRegistry:
    """Centralized tool management"""
    
    def __init__(self):
        self.tools = {}
        self._setup_default_tools()
    
    def _setup_default_tools(self):
        """Register default tools"""
        self.register("web_search", self._web_search, 
                     "Search the web for information")
        self.register("calculator", self._calculator,
                     "Perform mathematical calculations")
        self.register("file_reader", self._file_reader,
                     "Read content from files")
        self.register("api_caller", self._api_caller,
                     "Make HTTP API calls")
    
    def register(self, name: str, func: callable, description: str):
        """Register a new tool"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "usage_count": 0
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _web_search(self, query: str) -> Dict[str, Any]:
        """Web search implementation"""
        # Placeholder - implement with actual search API
        return {
            "results": [
                {"title": "Example Result", "snippet": f"Information about {query}"}
            ]
        }
    
    async def _calculator(self, expression: str) -> float:
        """Safe math evaluation"""
        try:
            # Safe evaluation of math expressions
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            return eval(expression, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _file_reader(self, filepath: str) -> str:
        """Read file content"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    async def _api_caller(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP API calls"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method=config.get("method", "GET"),
                    url=config["url"],
                    headers=config.get("headers", {}),
                    json=config.get("body"),
                    timeout=30.0
                )
                return {
                    "status_code": response.status_code,
                    "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                }
            except Exception as e:
                return {"error": str(e)}

class ToolCallingAgent(AdvancedReasoningAgent):
    """Agent with sophisticated tool calling capabilities"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description, **kwargs)
        self.tool_registry = ToolRegistry()
        self.tool_history = []
    
    async def call_tool(self, tool_name: str, tool_input: Any) -> Any:
        """Execute a tool with proper error handling"""
        if tool_name not in self.tool_registry.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        tool_info = self.tool_registry.tools[tool_name]
        tool_info["usage_count"] += 1
        
        try:
            result = await tool_info["function"](tool_input)
            
            # Log tool usage
            self.tool_history.append({
                "tool": tool_name,
                "input": tool_input,
                "output": result,
                "timestamp": datetime.now().isoformat()
            })
            
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {str(e)}")
            return {"error": str(e)}
    
    def get_tool_description(self) -> str:
        """Get formatted tool descriptions for prompts"""
        descriptions = []
        for name, info in self.tool_registry.tools.items():
            descriptions.append(f"{name}: {info['description']}")
        return "\n".join(descriptions)
```

### Error Handling and Robustness

```python
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Any
import signal

class RobustAgent(ToolCallingAgent):
    """Agent with comprehensive error handling"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description, **kwargs)
        self.timeout = 300  # 5 minutes default timeout
        self.retry_attempts = 3
        self.error_recovery_strategies = {
            "timeout": self._handle_timeout,
            "api_error": self._handle_api_error,
            "parsing_error": self._handle_parsing_error,
            "tool_error": self._handle_tool_error
        }
    
    @asynccontextmanager
    async def error_handling(self):
        """Context manager for comprehensive error handling"""
        try:
            yield
        except asyncio.TimeoutError:
            await self.error_recovery_strategies["timeout"]()
        except httpx.HTTPError as e:
            await self.error_recovery_strategies["api_error"](e)
        except json.JSONDecodeError as e:
            await self.error_recovery_strategies["parsing_error"](e)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    async def execute_with_timeout(self, coro, timeout: Optional[float] = None):
        """Execute coroutine with timeout"""
        timeout = timeout or self.timeout
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {timeout}s")
            return {"error": "Operation timed out", "timeout": timeout}
    
    async def _handle_timeout(self):
        """Handle timeout errors"""
        logger.warning("Handling timeout - attempting graceful recovery")
        # Save current state
        self._save_checkpoint()
        # Return partial results if available
        return {"partial_result": self.memory.short_term[-1] if self.memory.short_term else None}
    
    async def _handle_api_error(self, error: httpx.HTTPError):
        """Handle API errors with exponential backoff"""
        logger.error(f"API error: {str(error)}")
        for attempt in range(self.retry_attempts):
            wait_time = 2 ** attempt
            logger.info(f"Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
            # Retry logic here
    
    async def _handle_parsing_error(self, error: json.JSONDecodeError):
        """Handle parsing errors"""
        logger.error(f"Parsing error: {str(error)}")
        # Attempt to fix common JSON issues
        # Return structured error for agent to handle
        return {"error": "parsing_failed", "raw_content": str(error)}
    
    async def _handle_tool_error(self, tool_name: str, error: Exception):
        """Handle tool-specific errors"""
        logger.error(f"Tool {tool_name} error: {str(error)}")
        # Try alternative tools if available
        alternatives = self._find_alternative_tools(tool_name)
        if alternatives:
            logger.info(f"Trying alternative tool: {alternatives[0]}")
            # Attempt with alternative
    
    def _save_checkpoint(self):
        """Save agent state for recovery"""
        checkpoint = {
            "state": self.state,
            "memory": self.memory,
            "tool_history": self.tool_history,
            "timestamp": datetime.now().isoformat()
        }
        # Save to file or database
        with open(f"checkpoints/{self.name}_{datetime.now().timestamp()}.json", 'w') as f:
            json.dump(checkpoint, f)
```

## 4. Best Practices & Patterns

### Prompt Engineering for Agents

```python
class PromptTemplates:
    """Collection of effective prompt templates for agents"""
    
    SYSTEM_PROMPT = """You are {agent_name}, an AI agent specialized in {specialization}.

Core Capabilities:
{capabilities}

Operating Principles:
1. Think step-by-step before taking any action
2. Use tools when they would provide more accurate or up-to-date information
3. Verify important information when possible
4. Be explicit about uncertainty
5. Learn from previous interactions stored in memory

Current Context:
- Date/Time: {datetime}
- User: {user_info}
- Session: {session_id}
"""
    
    REASONING_PROMPT = """Given the current situation:
{context}

Available Actions:
{available_actions}

Previous Steps:
{history}

Analyze the situation and determine the best next action. Consider:
1. What is the goal?
2. What information do we have?
3. What information is missing?
4. Which action would make the most progress?
5. What could go wrong?

Reasoning:"""
    
    TOOL_SELECTION_PROMPT = """Task: {task}

Available tools:
{tools}

Previous tool calls in this session:
{tool_history}

Select the most appropriate tool for this task. Consider:
- Tool capabilities and limitations
- Input requirements
- Expected output format
- Performance characteristics

If multiple tools could work, prefer:
1. More specific/specialized tools
2. Tools with better error handling
3. Tools you haven't tried yet for this type of task

Selected tool and rationale:"""
    
    SELF_REFLECTION_PROMPT = """Task: {original_task}
Actions taken: {actions}
Result: {result}

Reflect on this interaction:
1. Was the task completed successfully?
2. What went well?
3. What could be improved?
4. What would you do differently next time?
5. What patterns should be remembered for similar future tasks?

Reflection:"""

def format_prompt(template: str, **kwargs) -> str:
    """Safely format prompt templates"""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.warning(f"Missing prompt variable: {e}")
        # Return template with missing vars marked
        import re
        missing_vars = re.findall(r'\{(\w+)\}', template)
        for var in missing_vars:
            if var not in kwargs:
                kwargs[var] = f"[MISSING: {var}]"
        return template.format(**kwargs)
```

### State Management

```python
from enum import Enum
from typing import Dict, Any, Optional, List
import pickle
import aioredis

class StateManager:
    """Distributed state management for agents"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or "redis://localhost"
        self.local_state = {}
        self.state_history = []
        self.redis_client = None
    
    async def initialize(self):
        """Initialize connection to state store"""
        if self.redis_url:
            self.redis_client = await aioredis.create_redis_pool(self.redis_url)
    
    async def save_state(self, agent_id: str, state: Dict[str, Any]):
        """Save agent state"""
        state_with_metadata = {
            "agent_id": agent_id,
            "state": state,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Save to history
        self.state_history.append(state_with_metadata)
        
        # Save locally
        self.local_state[agent_id] = state_with_metadata
        
        # Save to Redis if available
        if self.redis_client:
            await self.redis_client.setex(
                f"agent_state:{agent_id}",
                3600,  # 1 hour TTL
                pickle.dumps(state_with_metadata)
            )
    
    async def load_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent state"""
        # Try Redis first
        if self.redis_client:
            state_data = await self.redis_client.get(f"agent_state:{agent_id}")
            if state_data:
                return pickle.loads(state_data)
        
        # Fall back to local state
        return self.local_state.get(agent_id)
    
    async def get_state_history(
        self, 
        agent_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get state history for debugging"""
        history = [
            s for s in self.state_history 
            if s["agent_id"] == agent_id
        ]
        return history[-limit:]
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()

class StatefulAgent(RobustAgent):
    """Agent with persistent state management"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description, **kwargs)
        self.state_manager = StateManager()
        self.state_checkpoint_interval = 5  # Save state every 5 steps
        self.steps_since_checkpoint = 0
    
    async def initialize(self):
        """Initialize agent with saved state if available"""
        await self.state_manager.initialize()
        saved_state = await self.state_manager.load_state(self.name)
        
        if saved_state:
            logger.info(f"Restored state for {self.name}")
            self._restore_from_state(saved_state["state"])
    
    def _restore_from_state(self, state: Dict[str, Any]):
        """Restore agent from saved state"""
        self.memory = state.get("memory", self.memory)
        self.tool_history = state.get("tool_history", [])
        self.reasoning_traces = state.get("reasoning_traces", [])
    
    async def checkpoint(self):
        """Save current state"""
        state = {
            "memory": self.memory,
            "tool_history": self.tool_history,
            "reasoning_traces": self.reasoning_traces,
            "current_task": getattr(self, "current_task", None)
        }
        await self.state_manager.save_state(self.name, state)
        self.steps_since_checkpoint = 0
    
    async def step(self, *args, **kwargs):
        """Execute one step with automatic checkpointing"""
        result = await super().step(*args, **kwargs)
        
        self.steps_since_checkpoint += 1
        if self.steps_since_checkpoint >= self.state_checkpoint_interval:
            await self.checkpoint()
        
        return result
```

### Safety and Alignment Considerations

```python
from typing import List, Dict, Any, Set
import re

class SafetyFilter:
    """Safety filtering for agent actions"""
    
    def __init__(self):
        self.blocked_patterns = [
            r"rm\s+-rf\s+/",  # Dangerous file operations
            r"sudo\s+",       # Privilege escalation
            r"curl\s+.*\|\s*sh",  # Remote code execution
        ]
        self.sensitive_info_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit card
        ]
        self.safe_domains = {
            "wikipedia.org",
            "stackoverflow.com",
            "github.com",
            "docs.python.org"
        }
    
    def is_action_safe(self, action: str) -> tuple[bool, str]:
        """Check if an action is safe to execute"""
        # Check for dangerous patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, action, re.IGNORECASE):
                return False, f"Blocked dangerous pattern: {pattern}"
        
        # Check for sensitive information
        for pattern in self.sensitive_info_patterns:
            if re.search(pattern, action):
                return False, "Contains potentially sensitive information"
        
        return True, "Safe"
    
    def sanitize_output(self, text: str) -> str:
        """Remove sensitive information from output"""
        # Redact sensitive patterns
        for pattern in self.sensitive_info_patterns:
            text = re.sub(pattern, "[REDACTED]", text)
        return text
    
    def is_url_safe(self, url: str) -> bool:
        """Check if URL is safe to access"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        # Check domain whitelist
        domain = parsed.netloc.lower()
        return any(domain.endswith(safe) for safe in self.safe_domains)

class AlignedAgent(StatefulAgent):
    """Agent with built-in safety and alignment features"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description, **kwargs)
        self.safety_filter = SafetyFilter()
        self.alignment_principles = [
            "Prioritize user safety and privacy",
            "Refuse requests that could cause harm",
            "Be transparent about capabilities and limitations",
            "Respect intellectual property and licensing",
            "Maintain professional boundaries"
        ]
        self.refused_requests = []
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with safety checks"""
        # Check action safety
        action_str = json.dumps(action)
        is_safe, reason = self.safety_filter.is_action_safe(action_str)
        
        if not is_safe:
            logger.warning(f"Refused unsafe action: {reason}")
            self.refused_requests.append({
                "action": action,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })
            return {
                "error": "Action refused for safety reasons",
                "reason": reason
            }
        
        # Execute with monitoring
        result = await super().execute_action(action)
        
        # Sanitize output
        if isinstance(result, dict) and "output" in result:
            result["output"] = self.safety_filter.sanitize_output(
                str(result["output"])
            )
        
        return result
    
    def add_alignment_check(self, principle: str):
        """Add custom alignment principle"""
        self.alignment_principles.append(principle)
    
    async def get_safety_report(self) -> Dict[str, Any]:
        """Generate safety metrics report"""
        return {
            "total_requests": len(self.tool_history),
            "refused_requests": len(self.refused_requests),
            "safety_rate": 1 - (len(self.refused_requests) / max(len(self.tool_history), 1)),
            "recent_refusals": self.refused_requests[-5:],
            "active_principles": self.alignment_principles
        }
```

### Performance Optimization

```python
import asyncio
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor
import psutil

class PerformanceMonitor:
    """Monitor and optimize agent performance"""
    
    def __init__(self):
        self.metrics = {
            "llm_calls": 0,
            "tool_calls": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.timing_data = []
    
    def record_timing(self, operation: str, duration: float):
        """Record operation timing"""
        self.timing_data.append({
            "operation": operation,
            "duration": duration,
            "timestamp": time.time()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        if not self.timing_data:
            return {"message": "No performance data available"}
        
        # Calculate aggregates
        operation_times = {}
        for entry in self.timing_data:
            op = entry["operation"]
            if op not in operation_times:
                operation_times[op] = []
            operation_times[op].append(entry["duration"])
        
        summary = {
            "metrics": self.metrics,
            "operation_stats": {}
        }
        
        for op, times in operation_times.items():
            summary["operation_stats"][op] = {
                "count": len(times),
                "avg_duration": sum(times) / len(times),
                "min_duration": min(times),
                "max_duration": max(times)
            }
        
        # System resources
        summary["system_resources"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
        
        return summary

class OptimizedAgent(AlignedAgent):
    """Performance-optimized agent implementation"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, description, **kwargs)
        self.performance_monitor = PerformanceMonitor()
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.batch_size = 5
        self.pending_operations = []
    
    @lru_cache(maxsize=128)
    async def _cached_llm_call(self, prompt_hash: str) -> str:
        """Cached LLM calls for repeated queries"""
        self.performance_monitor.metrics["cache_hits"] += 1
        # Actual LLM call implementation
        return await self._make_llm_call(prompt_hash)
    
    async def batch_process(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Process multiple operations in batch"""
        start_time = time.time()
        
        # Group operations by type
        grouped = {}
        for op in operations:
            op_type = op["type"]
            if op_type not in grouped:
                grouped[op_type] = []
            grouped[op_type].append(op)
        
        # Process each group concurrently
        tasks = []
        for op_type, ops in grouped.items():
            if op_type == "llm_call":
                tasks.append(self._batch_llm_calls(ops))
            elif op_type == "tool_call":
                tasks.append(self._batch_tool_calls(ops))
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for result_group in results:
            all_results.extend(result_group)
        
        duration = time.time() - start_time
        self.performance_monitor.record_timing("batch_process", duration)
        
        return all_results
    
    async def _batch_llm_calls(self, calls: List[Dict[str, Any]]) -> List[Any]:
        """Batch multiple LLM calls"""
        # Create batched prompt
        batched_prompt = "\n---\n".join([
            f"Query {i+1}: {call['prompt']}"
            for i, call in enumerate(calls)
        ])
        
        # Single LLM call for all queries
        response = await self.llm.agenerate([[
            {"role": "system", "content": "Answer each query separately."},
            {"role": "user", "content": batched_prompt}
        ]])
        
        # Parse batched response
        # Implementation depends on LLM response format
        return self._parse_batched_response(response.generations[0][0].text)
    
    def add_to_batch(self, operation: Dict[str, Any]):
        """Add operation to pending batch"""
        self.pending_operations.append(operation)
        
        if len(self.pending_operations) >= self.batch_size:
            # Process batch in background
            asyncio.create_task(self._process_pending_batch())
    
    async def _process_pending_batch(self):
        """Process pending batched operations"""
        if not self.pending_operations:
            return
        
        operations = self.pending_operations[:self.batch_size]
        self.pending_operations = self.pending_operations[self.batch_size:]
        
        await self.batch_process(operations)
    
    async def optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt for token efficiency"""
        # Remove redundant whitespace
        prompt = " ".join(prompt.split())
        
        # Use compression techniques for long prompts
        if len(prompt) > 1000:
            # Implement prompt compression
            # This is a placeholder - actual implementation would use
            # techniques like removing examples, summarizing context, etc.
            pass
        
        return prompt
```

### Testing and Evaluation Methods

```python
import unittest
from typing import List, Dict, Any, Callable
import statistics

class AgentEvaluator:
    """Comprehensive agent evaluation framework"""
    
    def __init__(self):
        self.test_cases = []
        self.metrics = {
            "accuracy": [],
            "latency": [],
            "cost": [],
            "safety_violations": 0,
            "failure_rate": 0
        }
    
    def add_test_case(
        self, 
        name: str, 
        input_data: Any, 
        expected_output: Any,
        evaluator: Callable[[Any, Any], float]
    ):
        """Add a test case for evaluation"""
        self.test_cases.append({
            "name": name,
            "input": input_data,
            "expected": expected_output,
            "evaluator": evaluator
        })
    
    async def evaluate_agent(
        self, 
        agent: BaseAgent, 
        test_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run evaluation suite on agent"""
        results = []
        
        for test_case in self.test_cases:
            if test_subset and test_case["name"] not in test_subset:
                continue
            
            start_time = time.time()
            
            try:
                # Run agent
                output = await agent.run(test_case["input"])
                
                # Evaluate result
                score = test_case["evaluator"](
                    output, 
                    test_case["expected"]
                )
                
                duration = time.time() - start_time
                
                results.append({
                    "test": test_case["name"],
                    "success": True,
                    "score": score,
                    "duration": duration
                })
                
                self.metrics["accuracy"].append(score)
                self.metrics["latency"].append(duration)
                
            except Exception as e:
                results.append({
                    "test": test_case["name"],
                    "success": False,
                    "error": str(e)
                })
                self.metrics["failure_rate"] += 1
        
        return self._compile_results(results)
    
    def _compile_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile evaluation results"""
        successful_results = [r for r in results if r.get("success", False)]
        
        return {
            "summary": {
                "total_tests": len(results),
                "passed": len(successful_results),
                "failed": len(results) - len(successful_results),
                "average_score": statistics.mean(self.metrics["accuracy"]) if self.metrics["accuracy"] else 0,
                "average_latency": statistics.mean(self.metrics["latency"]) if self.metrics["latency"] else 0,
                "failure_rate": self.metrics["failure_rate"] / len(results) if results else 0
            },
            "detailed_results": results,
            "metrics": self.metrics
        }

# Example evaluators
def exact_match_evaluator(output: Any, expected: Any) -> float:
    """Check for exact match"""
    return 1.0 if output == expected else 0.0

def semantic_similarity_evaluator(output: str, expected: str) -> float:
    """Evaluate semantic similarity between outputs"""
    # Simplified - use embedding similarity in practice
    output_words = set(output.lower().split())
    expected_words = set(expected.lower().split())
    
    if not expected_words:
        return 0.0
    
    intersection = output_words & expected_words
    union = output_words | expected_words
    
    return len(intersection) / len(union) if union else 0.0

def task_completion_evaluator(output: Dict[str, Any], expected: Dict[str, Any]) -> float:
    """Evaluate if task was completed successfully"""
    if "error" in output:
        return 0.0
    
    # Check required fields
    required_fields = expected.get("required_fields", [])
    for field in required_fields:
        if field not in output:
            return 0.0
    
    # Partial credit for partial completion
    if "subtasks" in expected:
        completed = sum(1 for task in expected["subtasks"] if task in output.get("completed_tasks", []))
        return completed / len(expected["subtasks"])
    
    return 1.0
```

## 5. Simple Example Agents

### Task-Planning Agent

```python
# task_planning_agent.py
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

@dataclass
class Task:
    """Represents a single task"""
    id: str
    description: str
    dependencies: List[str] = None
    priority: int = 1
    estimated_hours: float = 1.0
    status: str = "pending"
    assigned_to: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class TaskPlanningAgent(BaseAgent):
    """Agent specialized in breaking down projects into tasks and creating execution plans"""
    
    def __init__(self, name: str = "TaskPlanner"):
        super().__init__(name, "Task planning and project management")
        self.tasks = {}
        self.project_context = {}
        
    async def think(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project requirements and create task breakdown"""
        project_description = observation.get("input", "")
        
        # Create prompt for task breakdown
        prompt = f"""Break down this project into specific, actionable tasks:

Project: {project_description}

For each task, provide:
1. Task description (clear and specific)
2. Dependencies (other tasks that must be completed first)
3. Estimated time in hours
4. Priority (1-5, where 5 is highest)

Format each task as:
TASK: [description]
DEPENDS_ON: [comma-separated task numbers or "none"]
HOURS: [number]
PRIORITY: [1-5]

After listing all tasks, provide:
CRITICAL_PATH: [comma-separated task numbers in order]
TOTAL_TIME: [sum of hours for critical path]"""

        # Simulate LLM response (in practice, call actual LLM)
        llm_response = await self._simulate_llm_call(prompt)
        
        # Parse response into tasks
        tasks = self._parse_task_response(llm_response)
        
        return {
            "tasks": tasks,
            "thought": "Analyzed project and created task breakdown",
            "is_complete": len(tasks) > 0
        }
    
    async def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task planning actions"""
        tasks = action.get("tasks", [])
        
        # Create task objects
        for task_data in tasks:
            task = Task(
                id=task_data["id"],
                description=task_data["description"],
                dependencies=task_data.get("dependencies", []),
                priority=task_data.get("priority", 1),
                estimated_hours=task_data.get("hours", 1.0)
            )
            self.tasks[task.id] = task
        
        # Generate execution plan
        execution_plan = self._create_execution_plan()
        
        # Create Gantt chart data
        gantt_data = self._create_gantt_chart_data()
        
        return {
            "execution_plan": execution_plan,
            "gantt_data": gantt_data,
            "total_tasks": len(self.tasks),
            "critical_path": self._find_critical_path()
        }
    
    def _parse_task_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured task data"""
        tasks = []
        current_task = {}
        task_counter = 1
        
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("TASK:"):
                if current_task:
                    current_task["id"] = f"task_{task_counter}"
                    tasks.append(current_task)
                    task_counter += 1
                current_task = {"description": line[5:].strip()}
            elif line.startswith("DEPENDS_ON:"):
                deps = line[11:].strip()
                if deps.lower() != "none":
                    current_task["dependencies"] = [
                        f"task_{d.strip()}" for d in deps.split(",")
                    ]
            elif line.startswith("HOURS:"):
                current_task["hours"] = float(line[6:].strip())
            elif line.startswith("PRIORITY:"):
                current_task["priority"] = int(line[9:].strip())
        
        # Add last task
        if current_task:
            current_task["id"] = f"task_{task_counter}"
            tasks.append(current_task)
        
        return tasks
    
    def _create_execution_plan(self) -> List[Dict[str, Any]]:
        """Create ordered execution plan respecting dependencies"""
        completed = set()
        execution_plan = []
        
        while len(completed) < len(self.tasks):
            # Find tasks that can be executed
            available_tasks = []
            
            for task_id, task in self.tasks.items():
                if task_id in completed:
                    continue
                
                # Check if all dependencies are completed
                if all(dep in completed for dep in task.dependencies):
                    available_tasks.append(task)
            
            if not available_tasks:
                # Circular dependency detected
                break
            
            # Sort by priority and add to plan
            available_tasks.sort(key=lambda t: t.priority, reverse=True)
            
            for task in available_tasks:
                execution_plan.append({
                    "step": len(execution_plan) + 1,
                    "task_id": task.id,
                    "description": task.description,
                    "estimated_hours": task.estimated_hours,
                    "can_parallel": len(available_tasks) > 1
                })
                completed.add(task.id)
        
        return execution_plan
    
    def _find_critical_path(self) -> List[str]:
        """Find the critical path through the project"""
        # Simplified critical path calculation
        # In practice, use proper CPM algorithm
        
        # Topological sort
        visited = set()
        path = []
        
        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            task = self.tasks.get(task_id)
            if task:
                for dep in task.dependencies:
                    visit(dep)
                path.append(task_id)
        
        for task_id in self.tasks:
            visit(task_id)
        
        return path
    
    def _create_gantt_chart_data(self) -> List[Dict[str, Any]]:
        """Create data structure for Gantt chart visualization"""
        gantt_data = []
        task_end_times = {}
        
        execution_plan = self._create_execution_plan()
        current_time = 0
        
        for step in execution_plan:
            task_id = step["task_id"]
            task = self.tasks[task_id]
            
            # Calculate start time based on dependencies
            start_time = 0
            for dep in task.dependencies:
                if dep in task_end_times:
                    start_time = max(start_time, task_end_times[dep])
            
            end_time = start_time + task.estimated_hours
            task_end_times[task_id] = end_time
            
            gantt_data.append({
                "task_id": task_id,
                "task_name": task.description,
                "start": start_time,
                "duration": task.estimated_hours,
                "end": end_time,
                "dependencies": task.dependencies
            })
        
        return gantt_data
    
    async def _simulate_llm_call(self, prompt: str) -> str:
        """Simulate LLM response for testing"""
        # In practice, replace with actual LLM call
        return """TASK: Set up project repository and development environment
DEPENDS_ON: none
HOURS: 2
PRIORITY: 5

TASK: Design database schema
DEPENDS_ON: none  
HOURS: 4
PRIORITY: 4

TASK: Implement user authentication
DEPENDS_ON: 1, 2
HOURS: 8
PRIORITY: 5

TASK: Create API endpoints
DEPENDS_ON: 2
HOURS: 12
PRIORITY: 4

TASK: Build frontend UI
DEPENDS_ON: 1
HOURS: 16
PRIORITY: 3

TASK: Integrate frontend with API
DEPENDS_ON: 4, 5
HOURS: 8
PRIORITY: 4

TASK: Write tests
DEPENDS_ON: 3, 4
HOURS: 10
PRIORITY: 3

TASK: Deploy to production
DEPENDS_ON: 6, 7
HOURS: 4
PRIORITY: 5

CRITICAL_PATH: 1,2,3,4,6,7,8
TOTAL_TIME: 50"""

# Example usage
async def demo_task_planner():
    agent = TaskPlanningAgent()
    
    result = await agent.run(
        "Build a web application with user authentication, "
        "RESTful API, and React frontend"
    )
    
    print("Execution Plan:")
    for step in result["execution_plan"]:
        print(f"\nStep {step['step']}: {step['description']}")
        print(f"  Estimated time: {step['estimated_hours']} hours")
        print(f"  Can parallelize: {step['can_parallel']}")
    
    print(f"\nTotal tasks: {result['total_tasks']}")
    print(f"Critical path: {' -> '.join(result['critical_path'])}")
```

## 6. Common Pitfalls & Solutions

### Typical Mistakes Beginners Make

```python
# common_mistakes.py

class CommonMistakes:
    """Examples of common agent implementation mistakes and solutions"""
    
    # Mistake 1: Not handling rate limits
    @staticmethod
    async def bad_api_calls():
        """DON'T: Spam API without rate limiting"""
        results = []
        for i in range(100):
            # This will likely hit rate limits
            result = await call_api(f"query_{i}")
            results.append(result)
        return results
    
    @staticmethod
    async def good_api_calls():
        """DO: Implement rate limiting"""
        from asyncio import Semaphore
        
        rate_limit = Semaphore(5)  # Max 5 concurrent calls
        results = []
        
        async def rate_limited_call(query):
            async with rate_limit:
                await asyncio.sleep(0.1)  # 100ms between calls
                return await call_api(query)
        
        tasks = [rate_limited_call(f"query_{i}") for i in range(100)]
        results = await asyncio.gather(*tasks)
        return results
    
    # Mistake 2: Poor error handling
    @staticmethod
    async def bad_error_handling(agent):
        """DON'T: Let errors crash the agent"""
        result = await agent.call_tool("some_tool", "input")
        # Assumes success - will crash on error
        return result["data"]
    
    @staticmethod
    async def good_error_handling(agent):
        """DO: Handle errors gracefully"""
        try:
            result = await agent.call_tool("some_tool", "input")
            if "error" in result:
                logger.warning(f"Tool error: {result['error']}")
                # Fallback strategy
                return await agent.use_alternative_approach()
            return result.get("data", {})
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"error": str(e), "fallback": True}
    
    # Mistake 3: Memory leaks
    @staticmethod
    def bad_memory_management():
        """DON'T: Keep unlimited history"""
        class LeakyAgent:
            def __init__(self):
                self.history = []  # Grows forever!
            
            def add_interaction(self, data):
                self.history.append(data)
    
    @staticmethod
    def good_memory_management():
        """DO: Implement memory limits"""
        class EfficientAgent:
            def __init__(self, max_history=1000):
                self.history = []
                self.max_history = max_history
            
            def add_interaction(self, data):
                self.history.append(data)
                # Keep only recent history
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
    
    # Mistake 4: Blocking operations
    @staticmethod
    async def bad_blocking_code():
        """DON'T: Use blocking operations in async code"""
        import time
        time.sleep(5)  # Blocks entire event loop!
        return "Done"
    
    @staticmethod
    async def good_async_code():
        """DO: Use async operations"""
        await asyncio.sleep(5)  # Non-blocking
        return "Done"
    
    # Mistake 5: Poor prompt engineering
    @staticmethod
    def bad_prompt():
        """DON'T: Vague, unstructured prompts"""
        return "Do the thing with the data"
    
    @staticmethod
    def good_prompt():
        """DO: Clear, structured prompts"""
        return """Analyze the provided sales data and:
1. Calculate total revenue by region
2. Identify top 3 performing products
3. Suggest 2 actionable improvements

Format your response as:
ANALYSIS:
- [Your findings]

RECOMMENDATIONS:
1. [First recommendation]
2. [Second recommendation]"""
```

### Debugging Strategies

```python
# debugging_strategies.py

class AgentDebugger:
    """Comprehensive debugging toolkit for agents"""
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.debug_log = []
        self.breakpoints = set()
        self.trace_enabled = False
    
    async def trace_execution(self, enabled: bool = True):
        """Enable execution tracing"""
        self.trace_enabled = enabled
        
        if enabled:
            # Wrap agent methods with tracing
            original_think = self.agent.think
            
            async def traced_think(*args, **kwargs):
                self._log_trace("think", "start", args, kwargs)
                try:
                    result = await original_think(*args, **kwargs)
                    self._log_trace("think", "success", result=result)
                    return result
                except Exception as e:
                    self._log_trace("think", "error", error=str(e))
                    raise
            
            self.agent.think = traced_think
    
    def _log_trace(self, method: str, status: str, *args, **kwargs):
        """Log trace information"""
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "status": status,
            "args": args,
            "kwargs": kwargs
        }
        self.debug_log.append(trace_entry)
        
        if self.trace_enabled:
            print(f"[TRACE] {method} - {status}: {kwargs}")
    
    async def step_through(self, task: str):
        """Interactive step-through debugging"""
        print(f"Starting step-through debug for task: {task}")
        
        # Set initial observation
        observation = {"input": task, "iteration": 0}
        
        while True:
            # Show current state
            print(f"\n--- Iteration {observation['iteration']} ---")
            print(f"Current observation: {observation}")
            
            # Wait for user input
            command = input("\nDebug command (n=next, s=state, t=tools, q=quit): ")
            
            if command == 'q':
                break
            elif command == 's':
                self._show_agent_state()
            elif command == 't':
                self._show_available_tools()
            elif command == 'n':
                # Execute next step
                try:
                    thought = await self.agent.think(observation)
                    print(f"Agent thought: {thought}")
                    
                    if thought.get("is_complete"):
                        print("Task completed!")
                        break
                    
                    action_result = await self.agent.act(thought)
                    print(f"Action result: {action_result}")
                    
                    observation = {
                        "previous_thought": thought,
                        "action_result": action_result,
                        "iteration": observation["iteration"] + 1
                    }
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
    
    def _show_agent_state(self):
        """Display current agent state"""
        print("\n=== Agent State ===")
        print(f"Name: {self.agent.name}")
        print(f"State: {self.agent.state}")
        print(f"Memory (short-term): {len(self.agent.memory.short_term)} items")
        print(f"Tools: {list(self.agent.tools.keys())}")
    
    def _show_available_tools(self):
        """Display available tools"""
        print("\n=== Available Tools ===")
        for name, tool in self.agent.tools.items():
            print(f"- {name}: {tool}")
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze agent performance from debug logs"""
        if not self.debug_log:
            return {"error": "No debug data available"}
        
        # Calculate metrics
        method_times = {}
        error_count = 0
        
        for i in range(len(self.debug_log) - 1):
            entry = self.debug_log[i]
            next_entry = self.debug_log[i + 1]
            
            if entry["method"] == next_entry["method"] and entry["status"] == "start":
                duration = (
                    datetime.fromisoformat(next_entry["timestamp"]) - 
                    datetime.fromisoformat(entry["timestamp"])
                ).total_seconds()
                
                method = entry["method"]
                if method not in method_times:
                    method_times[method] = []
                method_times[method].append(duration)
            
            if entry["status"] == "error":
                error_count += 1
        
        # Compile analysis
        analysis = {
            "total_operations": len(self.debug_log),
            "error_count": error_count,
            "error_rate": error_count / len(self.debug_log),
            "method_performance": {}
        }
        
        for method, times in method_times.items():
            analysis["method_performance"][method] = {
                "avg_time": statistics.mean(times),
                "min_time": min(times),
                "max_time": max(times),
                "total_calls": len(times)
            }
        
        return analysis
    
    def export_debug_log(self, filepath: str):
        """Export debug log for analysis"""
        with open(filepath, 'w') as f:
            json.dump(self.debug_log, f, indent=2)
        print(f"Debug log exported to {filepath}")

# Usage example
async def debug_agent_example():
    agent = SimpleAgent("DebugTest", "Testing agent")
    debugger = AgentDebugger(agent)
    
    # Enable tracing
    await debugger.trace_execution(True)
    
    # Step through execution
    await debugger.step_through("Calculate the sum of prime numbers below 100")
    
    # Analyze performance
    analysis = debugger.analyze_performance()
    print(f"\nPerformance Analysis: {json.dumps(analysis, indent=2)}")
```

### Scaling Considerations

```python
# scaling_strategies.py

class ScalableAgentSystem:
    """Strategies for scaling agent systems"""
    
    def __init__(self):
        self.agents = {}
        self.load_balancer = LoadBalancer()
        self.message_queue = MessageQueue()
        self.distributed_memory = DistributedMemory()
    
    async def horizontal_scaling(self, agent_class, num_instances: int):
        """Scale horizontally with multiple agent instances"""
        for i in range(num_instances):
            agent = agent_class(f"Agent_{i}")
            self.agents[f"agent_{i}"] = agent
            
            # Register with load balancer
            self.load_balancer.register_agent(agent)
        
        print(f"Scaled to {num_instances} agent instances")
    
    async def vertical_scaling(self, agent: BaseAgent):
        """Scale vertically by enhancing single agent capabilities"""
        # Increase resources
        agent.max_parallel_tools = 10
        agent.memory_capacity *= 2
        agent.cache_size *= 2
        
        # Add more sophisticated reasoning
        agent.reasoning_strategies.extend([
            "monte_carlo_tree_search",
            "beam_search",
            "adversarial_reasoning"
        ])
        
        print("Enhanced agent capabilities for vertical scaling")
    
    async def distributed_processing(self, task: str):
        """Distribute task across multiple agents"""
        # Decompose task
        subtasks = await self._decompose_task(task)
        
        # Distribute subtasks
        results = []
        for subtask in subtasks:
            agent = self.load_balancer.get_next_agent()
            
            # Async task execution
            future = asyncio.create_task(
                self._process_subtask(agent, subtask)
            )
            results.append(future)
        
        # Gather results
        subtask_results = await asyncio.gather(*results)
        
        # Aggregate results
        return await self._aggregate_results(subtask_results)
    
    async def _decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """Decompose complex task into subtasks"""
        # Simplified decomposition
        return [
            {"type": "research", "query": task},
            {"type": "analysis", "data": task},
            {"type": "synthesis", "context": task}
        ]
    
    async def _process_subtask(self, agent: BaseAgent, subtask: Dict[str, Any]):
        """Process individual subtask"""
        return await agent.run(subtask["query"])
    
    async def _aggregate_results(self, results: List[Any]) -> Dict[str, Any]:
        """Aggregate results from distributed processing"""
        return {
            "aggregated_results": results,
            "summary": "Combined analysis from all agents",
            "confidence": 0.85
        }

class LoadBalancer:
    """Simple round-robin load balancer for agents"""
    
    def __init__(self):
        self.agents = []
        self.current_index = 0
    
    def register_agent(self, agent: BaseAgent):
        """Register agent with load balancer"""
        self.agents.append(agent)
    
    def get_next_agent(self) -> BaseAgent:
        """Get next agent in round-robin fashion"""
        if not self.agents:
            raise ValueError("No agents registered")
        
        agent = self.agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.agents)
        return agent

class MessageQueue:
    """Message queue for agent communication"""
    
    def __init__(self):
        self.queues = {}
    
    async def send_message(self, agent_id: str, message: Dict[str, Any]):
        """Send message to agent"""
        if agent_id not in self.queues:
            self.queues[agent_id] = asyncio.Queue()
        
        await self.queues[agent_id].put(message)
    
    async def receive_message(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Receive message for agent"""
        if agent_id not in self.queues:
            return None
        
        try:
            return await asyncio.wait_for(
                self.queues[agent_id].get(),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            return None

class DistributedMemory:
    """Distributed memory system for agent clusters"""
    
    def __init__(self, backend: str = "redis"):
        self.backend = backend
        self.local_cache = {}
        self.sync_interval = 5.0  # seconds
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in distributed memory"""
        # Update local cache
        self.local_cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl
        }
        
        # Sync to distributed backend
        await self._sync_to_backend(key, value, ttl)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed memory"""
        # Check local cache first
        if key in self.local_cache:
            entry = self.local_cache[key]
            if time.time() - entry["timestamp"] < entry["ttl"]:
                return entry["value"]
        
        # Fetch from backend
        return await self._fetch_from_backend(key)
    
    async def _sync_to_backend(self, key: str, value: Any, ttl: int):
        """Sync to distributed backend"""
        # Implementation depends on backend
        pass
    
    async def _fetch_from_backend(self, key: str) -> Optional[Any]:
        """Fetch from distributed backend"""
        # Implementation depends on backend
        pass
```

## 7. Resources & Next Steps

### Recommended Readings and Courses

```python
# learning_resources.py

ESSENTIAL_PAPERS = [
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "authors": "Yao et al.",
        "year": 2023,
        "url": "https://arxiv.org/abs/2210.03629",
        "why_important": "Foundational paper on agent reasoning patterns"
    },
    {
        "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "authors": "Shinn et al.",
        "year": 2023,
        "url": "https://arxiv.org/abs/2303.11366",
        "why_important": "Self-improvement through reflection"
    },
    {
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "authors": "Yao et al.",
        "year": 2023,
        "url": "https://arxiv.org/abs/2305.10601",
        "why_important": "Advanced reasoning strategies"
    },
    {
        "title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
        "authors": "Wu et al.",
        "year": 2023,
        "url": "https://arxiv.org/abs/2308.08155",
        "why_important": "Multi-agent systems architecture"
    }
]

ONLINE_COURSES = [
    {
        "name": "Building AI Agents with LangChain",
        "platform": "DeepLearning.AI",
        "duration": "4 weeks",
        "level": "Intermediate",
        "topics": ["LangChain basics", "Tool use", "Memory systems", "Production deployment"]
    },
    {
        "name": "Multi-Agent Systems",
        "platform": "Coursera",
        "instructor": "University of Edinburgh",
        "topics": ["Agent communication", "Coordination", "Distributed problem solving"]
    },
    {
        "name": "Practical AI Agents",
        "platform": "Fast.ai",
        "topics": ["From scratch implementation", "Real-world applications", "Performance optimization"]
    }
]

BOOKS = [
    {
        "title": "Artificial Intelligence: A Modern Approach",
        "authors": "Stuart Russell and Peter Norvig",
        "edition": "4th",
        "chapters_relevant": [2, 3, 7, 11, 25],
        "why": "Comprehensive foundation in AI and agent architectures"
    },
    {
        "title": "Agents and Multi-Agent Systems",
        "authors": "Michael Wooldridge",
        "focus": "Theoretical foundations and practical implementations"
    }
]
```

### Open-Source Projects to Study

```python
# recommended_projects.py

STUDY_PROJECTS = [
    {
        "name": "AutoGPT",
        "url": "https://github.com/Significant-Gravitas/AutoGPT",
        "description": "Autonomous AI agent with goal-seeking behavior",
        "key_learnings": [
            "Task decomposition",
            "Memory management",
            "Tool integration"
        ]
    },
    {
        "name": "BabyAGI",
        "url": "https://github.com/yoheinakajima/babyagi",
        "description": "Simple but powerful task-driven agent",
        "key_learnings": [
            "Task prioritization",
            "Execution loops",
            "Vector memory"
        ]
    },
    {
        "name": "LangChain",
        "url": "https://github.com/langchain-ai/langchain",
        "description": "Comprehensive framework for LLM applications",
        "key_learnings": [
            "Agent architectures",
            "Tool abstractions",
            "Production patterns"
        ]
    },
    {
        "name": "CrewAI",
        "url": "https://github.com/joaomdmoura/crewAI",
        "description": "Framework for orchestrating role-playing AI agents",
        "key_learnings": [
            "Role-based agents",
            "Agent collaboration",
            "Task delegation"
        ]
    },
    {
        "name": "OpenAI Swarm",
        "url": "https://github.com/openai/swarm",
        "description": "Lightweight multi-agent orchestration",
        "key_learnings": [
            "Minimal agent design",
            "Handoff patterns",
            "Stateless agents"
        ]
    }
]

def get_project_starter(project_name: str) -> str:
    """Get starter code for studying a project"""
    starters = {
        "AutoGPT": '''
# Study AutoGPT's agent loop
git clone https://github.com/Significant-Gravitas/AutoGPT
cd AutoGPT

# Key files to study:
# - autogpt/agents/agent.py - Main agent class
# - autogpt/memory/ - Memory implementations
# - autogpt/workspace/ - Tool integrations
''',
        "LangChain": '''
# Install and explore LangChain
pip install langchain langchain-experimental

# Study agent implementations:
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool

# Examine source:
# - langchain/agents/react/base.py
# - langchain/agents/openai_functions_agent/base.py
'''
    }
    return starters.get(project_name, "Project not found")
```

### Communities and Forums

```python
# community_resources.py

COMMUNITIES = {
    "Discord Servers": [
        {
            "name": "LangChain Discord",
            "url": "discord.gg/langchain",
            "active_members": "50k+",
            "best_for": "LangChain help, agent architectures"
        },
        {
            "name": "AI Agents Collective",
            "focus": "Agent development, research papers, collaboration"
        }
    ],
    
    "Reddit Communities": [
        "r/LocalLLaMA - Self-hosted agents",
        "r/singularity - AGI and advanced agents",
        "r/MachineLearning - Research discussions"
    ],
    
    "Slack Workspaces": [
        "AutoGPT Slack - AutoGPT development",
        "AI Engineers - Production AI systems"
    ],
    
    "Forums": [
        {
            "name": "LessWrong",
            "focus": "AI alignment and agent safety",
            "url": "lesswrong.com"
        },
        {
            "name": "Hugging Face Forums",
            "focus": "Model integration and agent tools"
        }
    ],
    
    "Conferences": [
        "AAMAS - Autonomous Agents and Multi-Agent Systems",
        "NeurIPS - Neural Information Processing Systems",
        "ICML - International Conference on Machine Learning"
    ]
}

# Quick start guide for community engagement
GETTING_STARTED = """
1. Join the LangChain Discord for immediate help
2. Follow #ai-agents on Twitter for latest developments
3. Subscribe to r/LocalLLaMA for open-source agent discussions
4. Attend virtual meetups on AI agent development
5. Contribute to open-source agent projects
"""
```

### Next Steps Roadmap

```python
# learning_roadmap.py

def create_learning_roadmap(current_level: str) -> Dict[str, Any]:
    """Create personalized learning roadmap"""
    
    roadmaps = {
        "beginner": {
            "month_1": [
                "Complete LangChain quickstart",
                "Build simple ReAct agent",
                "Understand tool calling",
                "Create basic memory system"
            ],
            "month_2": [
                "Implement task planning agent",
                "Add error handling",
                "Study AutoGPT architecture",
                "Build web research agent"
            ],
            "month_3": [
                "Create multi-agent system",
                "Implement distributed memory",
                "Deploy agent to production",
                "Contribute to open source"
            ]
        },
        
        "intermediate": {
            "month_1": [
                "Master advanced reasoning (ToT, CoT)",
                "Build custom agent frameworks",
                "Implement sophisticated memory",
                "Study SOTA papers"
            ],
            "month_2": [
                "Create domain-specific agents",
                "Optimize for production scale",
                "Implement safety measures",
                "Build evaluation frameworks"
            ],
            "month_3": [
                "Design novel agent architectures",
                "Publish agent research",
                "Lead open source project",
                "Speak at conferences"
            ]
        },
        
        "advanced": {
            "focus_areas": [
                "Novel agent architectures",
                "Multi-modal agents",
                "Neurosymbolic approaches",
                "Agent alignment research",
                "Distributed agent systems"
            ],
            "research_directions": [
                "Self-improving agents",
                "Adversarial robustness",
                "Interpretable reasoning",
                "Efficient architectures"
            ]
        }
    }
    
    return roadmaps.get(current_level, roadmaps["beginner"])

# Final recommendations
FINAL_TIPS = """
🚀 Key Success Factors for Agent Development:

1. **Start Simple**: Master basic ReAct before complex architectures
2. **Focus on Robustness**: Error handling > features
3. **Test Extensively**: Unit tests, integration tests, evaluation suites
4. **Monitor Production**: Logging, metrics, observability
5. **Stay Updated**: Field evolves rapidly - read papers weekly
6. **Contribute Back**: Share learnings with community
7. **Think Safety**: Always consider alignment and misuse
8. **Iterate Quickly**: Launch MVP, gather feedback, improve

Remember: The best agent is one that reliably solves real problems!
"""

print(FINAL_TIPS)
```

This comprehensive guide provides everything needed to start building effective AI agents using modern best practices. The examples are functional and can be adapted for specific use cases. Remember to always consider safety, test thoroughly, and iterate based on real-world performance.