# Columbus Agent

CUA-compliant computer-using agent with DSPy integration, Mem0 memory, and Qdrant vector storage.

## Features

- 🖥️ **CUA ComputerAgent**: Follows official CUA framework patterns
- 🧠 **DSPy Integration**: Intelligent routing and planning 
- 💾 **Mem0 + Qdrant**: Persistent vector memory system
- 🦙 **Ollama Models**: Local execution with no external dependencies
- 📊 **HUD OSWorld**: Verified benchmark integration

## Quick Start

### Prerequisites

1. **Install uv** (Python package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install Ollama** (for local models):
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama and pull models
ollama serve
ollama pull llama3.2:3b
ollama pull llama3.2:8b
```

3. **Install CUA SDK** (optional - for full computer use capabilities):
```bash
# Install CUA SDK for computer use
pip install "cua-agent[all]"
```

4. **Install Qdrant** (optional - for memory):
```bash
# Docker (recommended)
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### CUA SDK Compatibility

Columbus is designed to work with the official CUA SDK for computer use capabilities. However, it includes fallback functionality:

- **With CUA SDK**: Full computer use capabilities with composed local models (omniparser + ollama)
- **Without CUA SDK**: Columbus runs in fallback mode with basic functionality
- **Compatibility Issues**: If CUA SDK has syntax errors (e.g., Python version compatibility), Columbus automatically falls back to safe mode

The agent will display appropriate warnings and continue to function with available capabilities.

### Installation

```bash
# Clone and install with uv
git clone <repository-url>
cd columbus
uv pip install -e .

# Copy example config
cp .env.example .env
# Edit .env with your API keys (optional)
```

### Usage

**Interactive Mode (like Claude Code):**
```bash
uv run columbus

# Interactive prompts:
# > Take a screenshot of my desktop
# > /help
# > /benchmark  
# > /exit
```

**Direct Commands:**
```bash
# Run HUD OSWorld benchmark
uv run columbus benchmark

# Show help
uv run columbus help-cmd
```

### Benchmarking

```bash
# Run benchmarks (automatically falls back to simulation if HUD unavailable)
uv run columbus benchmark

# Interactive mode with benchmarks
uv run columbus
# Then use: /benchmark

# For HUD OSWorld benchmarks (requires HUD API key):
# 1. Get API key from https://hud.ai
# 2. pip install hud-ai
# 3. export HUD_API_KEY='your_key'
HUD_API_KEY='your_key' uv run columbus benchmark

# Run with different VM configurations
COMPUTER_OS_TYPE=linux COMPUTER_PROVIDER=local uv run columbus benchmark

# View real-time execution
# Visit http://localhost:8080 while benchmark is running
# VNC: localhost:5900 for direct VM access (if VM enabled)
```

### Tool Management

```bash
# Show help and available commands
uv run columbus help-cmd

# Interactive mode for tool usage
uv run columbus

# View configuration in dashboard
# Visit http://localhost:8080 for real-time monitoring
```

### Memory Management

Memory is managed automatically through the Mem0 + Qdrant integration:

```bash
# Enable memory (requires Qdrant running)
ENABLE_MEMORY=true uv run columbus

# Memory operations are handled through the interactive interface
uv run columbus
# Agent automatically stores and recalls relevant memories

# Monitor memory usage in dashboard
# Visit http://localhost:8080 for memory statistics
```

## Architecture

### CUA ComputerAgent Structure

Columbus follows the official CUA ComputerAgent patterns:

```python
from columbus.cua_agent.agent import ComputerAgent

# CUA-compliant initialization
agent = ComputerAgent(
    model="ollama_chat/llama3.2:8b",
    tools=[computer],
    instructions="You are a computer-using agent...",
    callbacks=[]
)

# CUA-style step prediction
result = agent.predict_step("take a screenshot")
```

### Core Components

1. **ComputerAgent** (`columbus.cua_agent`): CUA-compliant agent with computer interface
2. **Router** (`columbus.router`): DSPy-powered task classification  
3. **Planner** (`columbus.planning`): DSPy-based plan generation
4. **Tools** (`columbus.cua_agent.tools`): CUA computer interface (screenshot, click, type)
5. **Memory** (`columbus.memory`): Mem0 + Qdrant integration

### Configuration

Minimal configuration via `.env`:

```bash
# API Keys (optional)
OPENAI_API_KEY=your_key
MEM0_API_KEY=your_key
QDRANT_API_KEY=your_key

# Service URLs
OLLAMA_BASE_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333

# Models
COMPUTER_USE_MODEL=ollama_chat/llama3.2:8b
ROUTER_MODEL=ollama_chat/llama3.2:3b
```

## Development

```bash
# Install for development
uv pip install -e .

# Run tests
uv run pytest

# Format and lint
uv run ruff format .
uv run ruff check .
```

### Project Structure

```
columbus/
├── config/          # Configuration management
├── router/          # Model routing and task classification  
├── planning/        # DSPy planning system
├── cua_agent/       # Main agent implementation
│   ├── tools/       # Tool implementations
│   ├── callbacks.py # Event system
│   ├── hitl.py      # Human-in-the-loop
│   └── agent.py     # Main agent class
├── memory/          # mem0 and Qdrant integration
├── eval/            # Benchmarking and evaluation
└── cli.py           # Command-line interface
```

### Adding New Tools

1. Create tool class inheriting from `Tool`:

```python
from columbus.cua_agent.tools.base import Tool, ToolResult, ToolCapability

class MyTool(Tool):
    def __init__(self):
        super().__init__("mytool", "My custom tool")
    
    def _initialize_capabilities(self):
        self._capabilities = [
            ToolCapability(
                name="action",
                description="Perform custom action",
                parameters={"type": "object", "properties": {...}},
                risk_level=0.3
            )
        ]
    
    async def execute(self, action: str, **kwargs) -> ToolResult:
        # Implementation here
        pass
```

2. Register in `ToolRegistry`:

```python
from columbus.cua_agent.tools import ToolRegistry
from mytool import MyTool

registry = ToolRegistry()
registry.register_tool(MyTool())
```

### Custom DSPy Programs

Extend the planning system with custom DSPy programs:

```python
import dspy
from columbus.planning.signatures import PlanSignature

class CustomPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(PlanSignature)
        self.validate = dspy.ChainOfThought(ValidationSignature)
    
    def forward(self, goal, context):
        plan = self.generate(goal=goal, context=context)
        validation = self.validate(plan=plan.plan)
        return dspy.Prediction(plan=plan.plan, validation=validation.result)
```

### Memory Integration

Store custom memories:

```python
from columbus.memory import MemoryStore

memory = MemoryStore(config)

# Store memory
memory_id = memory.remember(
    user_id="user123",
    text="User prefers dark mode interfaces",
    tags=["preference", "ui"],
    metadata={"confidence": 0.9}
)

# Recall memories
memories = memory.recall(
    user_id="user123", 
    query="user interface preferences",
    limit=5
)
```

## Safety and Human-in-the-Loop

The agent includes comprehensive safety features:

### Risk Assessment
- Tools and actions are assigned risk scores (0.0-1.0)
- High-risk operations require confirmation
- Configurable risk thresholds

### Confirmation Gates
- Step-level confirmations for dangerous operations
- Tool-level confirmations for sensitive tools
- Plan modification requests
- Error recovery decisions

### Dry Run Mode
- Preview actions without execution
- Test plans safely
- Debug tool interactions

## Benchmarking

Columbus supports both simulated and real OSWorld-Verified benchmarks for standardized evaluation.

### Running Benchmarks

```bash
# Run both HUD OSWorld-Verified and simulated benchmarks
uv run columbus benchmark

# This will:
# 1. Try to run HUD OSWorld-Verified tests (requires API keys)
# 2. Fall back to simulated benchmark if HUD is unavailable
```

### OSWorld-Verified Integration

The agent supports HUD OSWorld-Verified benchmarking when properly configured:

**Requirements:**
- CUA SDK: `pip install "cua-agent[all]"`
- CUA API Key: Get from [trycua.com](https://trycua.com)
- HUD API Key: Get from [hud.ai](https://hud.ai)

**Environment Setup:**
```bash
export CUA_API_KEY="your-cua-api-key"
export HUD_API_KEY="your-hud-api-key"
```

### Benchmark Results

The benchmark tests the following capabilities:
- Screenshot capture and analysis
- Coordinate-based clicking
- Text input and typing
- Screen content analysis  
- Multi-step task planning
- Computer use intent detection

**Example Results:**
```
📊 Columbus Benchmark Results
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Task             ┃ Success ┃ CUA Detection ┃ Steps ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ Simulated Task 1 │   ✅    │      ✅       │   3   │  1.0  │
│ Simulated Task 2 │   ✅    │      ✅       │   4   │  1.0  │
│ Simulated Task 3 │   ✅    │      ✅       │   4   │  1.0  │
│ Simulated Task 4 │   ✅    │      ✅       │   5   │  1.0  │
│ Simulated Task 5 │   ❌    │      ❌       │   5   │  0.0  │
└──────────────────┴─────────┴───────────────┴───────┴───────┘

Success Rate: 80.0% (4/5)
CUA Detection Accuracy: 80.0% (4/5)  
Average Score: 0.80
Framework: Columbus + DSPy + Mem0 + Qdrant
```

### Custom Benchmarking

```python
from columbus.cua_agent.agent import ComputerAgent
from columbus.config.settings import Config

# Create agent
config = Config()
agent = ComputerAgent.from_config(cfg=config)

runner = BenchmarkRunner(agent)
hud = HUDBenchmark(runner)

# Run benchmarks
results = await hud.run_full_suite()
print(f"Success rate: {results['metrics']['success_rate']:.1%}")
```

### Custom Benchmarks

Create custom benchmark tasks:

```python
from columbus.eval.runner import BenchmarkTask

task = BenchmarkTask(
    id="custom_001",
    name="Custom Task",
    description="Test custom functionality", 
    goal="Perform custom action",
    category="custom",
    difficulty="medium",
    timeout=300
)

results = await runner.run_task(task)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- **DSPy**: Stanford's framework for programming language models
- **mem0**: Personalized memory layer for AI agents  
- **Qdrant**: Vector similarity search engine
- **CUA Framework**: Computer use agents framework
- **HUD/OS-World**: Benchmarking framework for agent evaluation
