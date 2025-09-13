"""
CUA-compliant Columbus Agent following ComputerAgent patterns
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field

from columbus.config.settings import Config
from columbus.router.router import Router, RouterInput
from columbus.planning.planning import Planner, PlanStep
from columbus.cua_agent.tools.base import ToolRegistry, ToolResult, ComputerInterface

try:
    from cua.computer import Computer

    CUA_COMPUTER_AVAILABLE = True
except ImportError:
    CUA_COMPUTER_AVAILABLE = False
from columbus.memory.memory import MemoryStore


class CuaAgentSettings(BaseModel):
    model_fast: Optional[str] = None
    model_reasoning: Optional[str] = None
    risk_threshold: float = 0.6
    confirm_tools: List[str] = Field(default_factory=list)
    dry_run: bool = False
    hitl_enabled: bool = True
    max_steps: int = 20
    timeout_s: int = 300
    tool_allowlist: Optional[List[str]] = None


class ComputerAgent:
    """CUA-compliant ComputerAgent following the official patterns"""

    def __init__(
        self,
        # Core CUA parameters (matching the official patterns)
        model: Optional[str] = None,
        computer: Optional[Any] = None,
        loop: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        instructions: Optional[str] = None,
        max_trajectory_budget: Optional[float] = None,
        # Columbus-specific extensions
        cfg: Optional[Config] = None,
        settings: Optional[CuaAgentSettings] = None,
    ):
        # CUA ComputerAgent standard parameters
        print(f"Initializing ComputerAgent with computer: {computer}")
        print(f"Initializing ComputerAgent with loop: {loop}")
        print(f"Initializing ComputerAgent with tools: {tools}")
        print(f"Initializing ComputerAgent with instructions: {instructions}")
        print(
            f"Initializing ComputerAgent with max_trajectory_budget: {max_trajectory_budget}"
        )
        print(f"Initializing ComputerAgent with cfg: {cfg}")
        print(f"Initializing ComputerAgent with settings: {settings}")
        self.model = model
        self.computer = computer
        self.loop = loop or "default"
        self.tools = tools or []
        self.instructions = instructions
        self.max_trajectory_budget = max_trajectory_budget

        # Columbus framework integration
        self.cfg = cfg or Config()
        self.settings = settings or CuaAgentSettings()

        # Initialize Columbus components
        self.router = Router(self.cfg)
        self.planner = Planner(model=model or self.cfg.computer_use_model, cfg=self.cfg)
        self.tool_registry = ToolRegistry()  # CUA-compliant tool registry

        # Only initialize memory if enabled
        if getattr(self.cfg, "enable_memory", False):
            self.memory = MemoryStore(self.cfg)
        else:
            self.memory = None

        # Initialize computer interface if not provided
        if not self.computer:
            if CUA_COMPUTER_AVAILABLE:
                # Use CUA Computer SDK for VM spawning
                self.computer = self._create_cua_computer()
            else:
                # Fallback to local computer interface
                self.computer = ComputerInterface()

        # Add computer to tools if not already present
        if self.computer not in self.tools:
            self.tools.append(self.computer)

    @classmethod
    def from_config(
        cls,
        cfg: Optional[Config] = None,
        settings: Optional[CuaAgentSettings] = None,
    ) -> "ComputerAgent":
        cfg = cfg or Config()
        s = settings or CuaAgentSettings(
            model_fast=cfg.router_model,
            model_reasoning=cfg.computer_use_model,
        )

        # CUA-style initialization with proper parameters
        model = cfg.computer_use_model
        instructions = """You are a computer-using agent powered by the Columbus framework.
You can interact with desktop applications, take screenshots, click, and type.
Always prioritize user safety and ask for confirmation on destructive actions."""

        return cls(
            model=model,
            computer=None,  # Will be auto-created
            loop="columbus",  # Our custom loop
            tools=[],
            instructions=instructions,
            max_trajectory_budget=5.0,
            cfg=cfg,
            settings=s,
        )

    async def run(
        self, goal: str, user_id: str = "default", context: Optional[str] = None
    ) -> Dict[str, Any]:
        # Enhance router input with context and CUA detection
        router_input = RouterInput(
            user_input=goal,
            history=[],
            context={"user_id": user_id, "context": context},
            requires_computer_use=self._detect_computer_use_intent(goal),
        )

        decision = self.router.route(router_input)

        # Get available tools for planning
        available_tools = (
            self.tools.get_available_tools()
            if hasattr(self.tools, "get_available_tools")
            else []
        )

        # Create comprehensive plan with computer use awareness
        steps: List[PlanStep] = self.planner.plan(goal, context, available_tools)

        # Retrieve relevant memories to inform execution (if memory is enabled)
        if self.memory:
            relevant_memories = self.memory.recall(goal, user_id, limit=3)
            memory_context = [mem.text for mem in relevant_memories]
        else:
            relevant_memories = []
            memory_context = []

        transcript: List[Dict[str, Any]] = []

        # Add initial context to transcript
        transcript.append(
            {
                "step": 0,
                "type": "initialization",
                "decision": {
                    "task_type": decision.task.value,
                    "model": decision.model,
                    "computer_use_model": decision.computer_use_model,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                },
                "memory_context": memory_context,
                "total_steps": len(steps),
            }
        )

        for i, step in enumerate(steps, start=1):
            if i > self.settings.max_steps:
                break

            entry: Dict[str, Any] = {
                "step": i,
                "thought": step.thought,
                "requires_computer_use": step.requires_computer_use,
                "expected_outcome": step.expected_outcome,
            }

            if step.tool and step.action:
                # Check tool allowlist
                if (
                    self.settings.tool_allowlist
                    and step.tool not in self.settings.tool_allowlist
                ):
                    entry["skipped"] = True
                    entry["reason"] = "Tool not allowed"
                    transcript.append(entry)
                    continue

                # For computer use actions, ensure we use the appropriate model
                active_model = (
                    decision.computer_use_model
                    if step.requires_computer_use
                    else decision.model
                )
                entry["active_model"] = active_model

                # Check if user confirmation is needed for this tool
                needs_confirmation = step.tool in self.settings.confirm_tools or (
                    step.requires_computer_use and self.settings.hitl_enabled
                )

                if needs_confirmation and not self.settings.dry_run:
                    entry["requires_confirmation"] = True
                    entry["tool"] = step.tool
                    entry["action"] = step.action
                    entry["args"] = step.args
                    entry["pending"] = True
                    transcript.append(entry)
                    continue

                if self.settings.dry_run:
                    entry["dry_run"] = True
                    entry["tool"] = step.tool
                    entry["action"] = step.action
                    entry["args"] = step.args
                else:
                    try:
                        # Execute tool with appropriate model context
                        execution_context = {
                            "model": active_model,
                            "requires_computer_use": step.requires_computer_use,
                            "user_id": user_id,
                        }

                        result: ToolResult = await self.tool_registry.execute(
                            step.tool,
                            step.action,
                            execution_context=execution_context,
                            **(step.args or {}),
                        )

                        entry["tool"] = step.tool
                        entry["action"] = step.action
                        entry["args"] = step.args
                        entry["result"] = result.model_dump()
                        entry["success"] = (
                            result.success if hasattr(result, "success") else True
                        )

                        # Store successful tool executions in memory (if enabled)
                        if entry.get("success", True) and self.memory:
                            memory_text = f"Successfully executed {step.tool}.{step.action}: {step.thought}"
                            self.memory.remember(
                                text=memory_text,
                                user_id=user_id,
                                tags=["execution", step.tool],
                                metadata={"step": i, "goal": goal},
                            )

                    except Exception as e:
                        entry["error"] = str(e)
                        entry["success"] = False
                        print(f"Error executing step {i}: {e}")

            transcript.append(entry)

        # Store the goal and outcome in memory (if enabled)
        if self.memory:
            self.memory.remember(
                text=f"Goal: {goal}. Completed {len([e for e in transcript if e.get('success', True)])} steps successfully.",
                user_id=user_id,
                tags=["goal", "completion"],
                metadata={"total_steps": len(steps), "transcript_id": id(transcript)},
            )

        return {
            "model": decision.model,
            "computer_use_model": decision.computer_use_model,
            "task_type": decision.task.value,
            "transcript": transcript,
            "summary": {
                "total_steps": len(steps),
                "executed_steps": len([e for e in transcript if e.get("step", 0) > 0]),
                "successful_steps": len(
                    [
                        e
                        for e in transcript
                        if e.get("success", True) and e.get("step", 0) > 0
                    ]
                ),
                "computer_use_steps": len(
                    [e for e in transcript if e.get("requires_computer_use", False)]
                ),
                "memory_context_used": len(memory_context),
            },
        }

    def _detect_computer_use_intent(self, goal: str) -> bool:
        """Detect if the goal likely requires computer use actions"""
        computer_use_indicators = [
            "click",
            "type",
            "screenshot",
            "capture",
            "open",
            "close",
            "drag",
            "select",
            "copy",
            "paste",
            "scroll",
            "mouse",
            "keyboard",
            "window",
            "application",
            "browser",
            "file",
            "folder",
            "desktop",
        ]

        goal_lower = goal.lower()
        return any(indicator in goal_lower for indicator in computer_use_indicators)

    def predict_step(self, goal: str, context: Optional[str] = None) -> Dict[str, Any]:
        """CUA-style predict_step method for single step prediction"""
        # This follows CUA ComputerAgent patterns for step-by-step execution
        router_input = RouterInput(
            user_input=goal,
            history=[],
            context={"context": context},
            requires_computer_use=self._detect_computer_use_intent(goal),
        )

        decision = self.router.route(router_input)
        steps = self.planner.plan(goal, context)

        if steps:
            first_step = steps[0]
            return {
                "step": first_step.model_dump()
                if hasattr(first_step, "model_dump")
                else str(first_step),
                "decision": decision.model_dump()
                if hasattr(decision, "model_dump")
                else str(decision),
                "requires_computer_use": first_step.requires_computer_use
                if hasattr(first_step, "requires_computer_use")
                else False,
            }

        return {
            "step": {
                "thought": "No actionable steps determined",
                "tool": None,
                "action": None,
            },
            "decision": decision.model_dump()
            if hasattr(decision, "model_dump")
            else str(decision),
            "requires_computer_use": False,
        }

    def _create_cua_computer(self):
        """Create CUA Computer instance with VM spawning"""
        if not CUA_COMPUTER_AVAILABLE:
            return ComputerInterface()

        try:
            from cua.computer import Computer

            # Configure CUA Computer for VM spawning
            computer_config = {
                "os_type": getattr(
                    self.cfg, "computer_os_type", "linux"
                ),  # Options: linux, windows, macOS
                "provider_type": getattr(
                    self.cfg, "computer_provider", "local"
                ),  # Options: cloud, local
                "name": f"columbus-{self.cfg.__hash__() % 10000}",  # Unique container name
            }

            # Add localhost port exposure for VM access
            if getattr(self.cfg, "vm_expose_ports", True):
                computer_config["port_mappings"] = {
                    "vnc": getattr(self.cfg, "vm_vnc_port", 5900),
                    "ssh": 2222,
                    "http": 8081,  # Avoid conflict with dashboard
                }

            # Add API key if using cloud provider
            if computer_config["provider_type"] == "cloud":
                computer_config["api_key"] = getattr(self.cfg, "cua_api_key", None)

            print(
                f"🖥️  Spawning CUA Computer: {computer_config['os_type']} ({computer_config['provider_type']})"
            )
            if computer_config.get("port_mappings"):
                print(f"🌐 VM will be accessible on localhost:")
                for service, port in computer_config["port_mappings"].items():
                    print(f"   {service.upper()}: localhost:{port}")

            # This will spawn the VM/container
            return Computer(**computer_config)

        except Exception as e:
            print(
                f"Warning: Failed to create CUA Computer, falling back to local interface: {e}"
            )
            return ComputerInterface()


# For backward compatibility
CUAAgent = ComputerAgent
