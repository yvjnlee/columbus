"""
CUA-style localhost dashboard for Columbus agent
"""

import asyncio
import json
import threading
import time
import warnings
from typing import Dict, Any, List
from pathlib import Path

# Suppress websockets deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="uvicorn.protocols.websockets"
)

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


class CUADashboard:
    """CUA-style dashboard for monitoring agent activity"""

    def __init__(self, port: int = 8080):
        self.port = port
        self.app = FastAPI(title="Columbus CUA Dashboard")
        self.connections: List[WebSocket] = []
        self.agent_state = {
            "status": "idle",
            "current_task": None,
            "steps": [],
            "screenshot": None,
            "trajectory": [],
        }

        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connections.append(websocket)
            try:
                # Send current state
                await websocket.send_text(json.dumps(self.agent_state))

                # Keep connection alive
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connections.remove(websocket)

        @self.app.get("/")
        async def dashboard():
            return HTMLResponse(self._get_dashboard_html())

        @self.app.get("/api/state")
        async def get_state():
            return self.agent_state

    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Columbus CUA Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .status { background: #333; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
                .screenshot { background: #222; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
                .trajectory { background: #333; padding: 15px; border-radius: 8px; }
                .step { background: #444; margin: 10px 0; padding: 10px; border-radius: 4px; }
                .computer-use { border-left: 4px solid #00ff00; }
                .reasoning { border-left: 4px solid #0080ff; }
                .error { border-left: 4px solid #ff0000; }
                .success { border-left: 4px solid #00ff00; }
                #screenshot-img { max-width: 100%; border-radius: 4px; }
                .timestamp { color: #888; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🖥️ Columbus CUA Dashboard</h1>
                    <p>Real-time Computer Use Agent Monitoring</p>
                </div>
                
                <div class="status">
                    <h3>Agent Status</h3>
                    <div id="status">Status: <span id="agent-status">Idle</span></div>
                    <div id="current-task">Task: <span id="task-name">None</span></div>
                </div>
                
                <div class="screenshot">
                    <h3>Latest Screenshot</h3>
                    <div id="screenshot-container">
                        <p>No screenshot available</p>
                    </div>
                </div>
                
                <div class="trajectory">
                    <h3>Execution Trajectory</h3>
                    <div id="trajectory-container">
                        <p>No steps executed yet</p>
                    </div>
                </div>
            </div>
            
            <script>
                const ws = new WebSocket('ws://localhost:8080/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                function updateDashboard(state) {
                    document.getElementById('agent-status').textContent = state.status;
                    document.getElementById('task-name').textContent = state.current_task || 'None';
                    
                    // Update screenshot
                    const screenshotContainer = document.getElementById('screenshot-container');
                    if (state.screenshot) {
                        screenshotContainer.innerHTML = `<img id="screenshot-img" src="data:image/png;base64,${state.screenshot}" alt="Screenshot">`;
                    }
                    
                    // Update trajectory
                    const trajectoryContainer = document.getElementById('trajectory-container');
                    if (state.steps && state.steps.length > 0) {
                        trajectoryContainer.innerHTML = state.steps.map(step => {
                            const stepClass = step.requires_computer_use ? 'computer-use' : 
                                            step.success === false ? 'error' :
                                            step.success === true ? 'success' : 'reasoning';
                            
                            return `
                                <div class="step ${stepClass}">
                                    <div><strong>Step ${step.step}:</strong> ${step.thought || step.message || 'Unknown'}</div>
                                    ${step.tool ? `<div><strong>Tool:</strong> ${step.tool}.${step.action || 'unknown'}</div>` : ''}
                                    ${step.success !== undefined ? `<div><strong>Status:</strong> ${step.success ? 'Success' : 'Failed'}</div>` : ''}
                                    <div class="timestamp">${new Date().toLocaleTimeString()}</div>
                                </div>
                            `;
                        }).join('');
                    }
                }
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
                
                ws.onclose = function() {
                    document.getElementById('agent-status').textContent = 'Disconnected';
                };
            </script>
        </body>
        </html>
        """

    async def update_state(self, key: str, value: Any):
        """Update agent state and notify all connected clients"""
        self.agent_state[key] = value

        # Broadcast to all connected clients
        if self.connections:
            message = json.dumps(self.agent_state)
            for connection in self.connections.copy():
                try:
                    await connection.send_text(message)
                except:
                    self.connections.remove(connection)

    def start_server(self):
        """Start the dashboard server in background thread"""

        def run_server():
            try:
                uvicorn.run(
                    self.app, host="localhost", port=self.port, log_level="error"
                )
            except OSError as e:
                if "Address already in use" in str(e):
                    print(
                        f"⚠️  Port {self.port} already in use, dashboard may already be running"
                    )
                else:
                    print(f"⚠️  Dashboard server error: {e}")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        print(f"🖥️  CUA Dashboard available at: http://localhost:{self.port}")
        return server_thread


# Global dashboard instance
dashboard = None


def get_dashboard() -> CUADashboard:
    """Get or create dashboard instance"""
    global dashboard
    if dashboard is None:
        dashboard = CUADashboard()
        dashboard.start_server()
    return dashboard
