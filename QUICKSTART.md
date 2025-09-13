# Columbus Quick Start Guide

## 🚀 Setup & Installation

### 1. Install Prerequisites

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Ollama (for local AI models)
# macOS:
brew install ollama
# Linux:
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Setup Columbus

```bash
# Clone and install
git clone <your-repo-url>
cd columbus
uv pip install -e .

# Run setup script (starts Qdrant, creates .env)
./setup.sh

# Pull required AI models
ollama serve &  # Start in background
ollama pull llama3.2:3b   # Fast routing model
ollama pull llama3.2:8b   # Computer use model
```

### 3. Basic Usage

```bash
# Interactive mode (recommended for first use)
uv run columbus

# Direct benchmark
uv run columbus benchmark

# Help
uv run columbus help-cmd
```

## 🖥️ VM & Localhost Access

Columbus automatically configures VM access:

- **Dashboard**: http://localhost:8080 (real-time monitoring)
- **VNC**: localhost:5900 (direct VM desktop access)
- **SSH**: localhost:2222 (command line access to VM)
- **HTTP**: localhost:8081 (VM web services)

## 📊 Benchmarking Options

### Option 1: Simulated Benchmarks (Default)
No additional setup required - works out of the box.

### Option 2: HUD OSWorld Benchmarks
For official benchmarks:
```bash
# Get API key from https://hud.ai
pip install hud-ai
export HUD_API_KEY='your_key_here'
uv run columbus benchmark
```

## 🔧 Configuration

Edit `.env` file for customization:
```bash
# VM Configuration
COMPUTER_OS_TYPE=linux          # linux, windows, macOS
COMPUTER_PROVIDER=local         # local, cloud
VM_EXPOSE_PORTS=true
VM_VNC_PORT=5900

# Memory (requires Qdrant)
ENABLE_MEMORY=true

# Models
COMPUTER_USE_MODEL=ollama_chat/llama3.2:8b
ROUTER_MODEL=ollama_chat/llama3.2:3b
```

## 🎮 Interactive Commands

In interactive mode (`uv run columbus`):
- `/help` - Show commands
- `/benchmark` - Run benchmarks
- `/exit` - Exit
- Any other text - Send to agent

## 🐛 Troubleshooting

### Port Already in Use
- Dashboard: Check if another Columbus instance is running
- VNC: Change `VM_VNC_PORT` in `.env`

### Model Not Found
```bash
ollama list  # Check installed models
ollama pull llama3.2:8b  # Pull missing models
```

### Memory Issues
```bash
docker ps  # Check if Qdrant is running
./setup.sh  # Restart services
```

## 📈 Monitoring

- **Real-time Dashboard**: http://localhost:8080
- **VM Desktop**: VNC client → localhost:5900
- **Logs**: Console output shows detailed execution

Enjoy using Columbus! 🧭
