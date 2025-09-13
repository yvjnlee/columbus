#!/bin/bash

echo "🚀 Setting up Columbus Services..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "📦 Starting Qdrant vector database..."
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant >/dev/null 2>&1 || {
    echo "🔄 Qdrant container already exists, starting it..."
    docker start qdrant >/dev/null 2>&1
}

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to be ready..."
until curl -s http://localhost:6333/collections >/dev/null 2>&1; do
    sleep 1
done

echo "✅ Qdrant is running at http://localhost:6333"

# Setup VM environment for CUA Computer
echo "🖥️  Setting up VM environment for computer use..."
echo "Creating .env file with VM configuration..."

# Create .env file with VM settings
cat > .env << EOF
# Columbus Configuration
ENABLE_MEMORY=true

# VM Configuration
COMPUTER_OS_TYPE=linux
COMPUTER_PROVIDER=local
VM_EXPOSE_PORTS=true
VM_VNC_PORT=5900

# Optional: Set to 'cloud' and add CUA_API_KEY for cloud VMs
# COMPUTER_PROVIDER=cloud
# CUA_API_KEY=your_key_here

# Optional: HUD benchmarking (get key from https://hud.ai)
# HUD_API_KEY=your_hud_key_here
EOF

echo ""
echo "🎯 Ready to start Columbus:"
echo "   uv run columbus           # Interactive mode"
echo "   uv run columbus benchmark # Run benchmarks"
echo ""
echo "📊 Dashboard will be available at:"
echo "   http://localhost:8080     # Real-time monitoring"
echo ""
echo "🖥️  VM will be accessible at:"
echo "   VNC: localhost:5900       # Direct VM access"
echo "   SSH: localhost:2222       # SSH into VM"
echo "   HTTP: localhost:8081      # VM web services"
echo ""
echo "To stop services later:"
echo "   docker stop qdrant"