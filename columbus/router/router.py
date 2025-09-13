"""
Enhanced router layer with CUA-compliant Ollama integration
"""

import dspy
import litellm
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from columbus.config.settings import Config


class TaskType(str, Enum):
    CHAT = "chat"
    PLANNING = "planning"
    COMPUTER_USE = "computer_use"
    TOOL_EXECUTION = "tool_execution"
    MEMORY_QUERY = "memory_query"
    REASONING = "reasoning"


class RouterInput(BaseModel):
    user_input: str
    history: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    requires_computer_use: bool = False


class RouteDecision(BaseModel):
    task: TaskType
    model: str
    computer_use_model: Optional[str] = None
    reason: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_tools: List[str] = Field(default_factory=list)


class RouterSignature(dspy.Signature):
    """Analyze user input and determine the appropriate task type and CUA requirements."""
    
    user_input = dspy.InputField(desc="The user's request or query")
    history = dspy.InputField(desc="Previous conversation history")
    context = dspy.InputField(desc="Additional context information")
    
    task_type = dspy.OutputField(desc="Task type: chat, planning, computer_use, tool_execution, memory_query, or reasoning")
    requires_computer_use = dspy.OutputField(desc="True if the task requires computer interaction (clicking, typing, screenshots)")
    confidence = dspy.OutputField(desc="Confidence score between 0 and 1")
    reasoning = dspy.OutputField(desc="Explanation for the routing decision")
    suggested_tools = dspy.OutputField(desc="Comma-separated list of suggested tools if applicable")


class Router:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # Configure Ollama models for different purposes
        self.chat_model = getattr(cfg, 'router_model', "ollama_chat/llama3.2:3b")
        self.computer_use_model = getattr(cfg, 'computer_use_model', "ollama_chat/llama3.2:8b")
        self.reasoning_model = getattr(cfg, 'reasoning_model', "ollama_chat/llama3.2:8b")
        
        # Set Ollama base URL if configured
        ollama_base = getattr(cfg, 'ollama_base_url', 'http://localhost:11434')
        litellm.set_verbose = False
        
        try:
            # Configure DSPy with Ollama via liteLLM 
            # Use ollama_chat provider for better responses per best practices
            lm = dspy.LM(model=self.chat_model, api_base=ollama_base, max_tokens=500)
            dspy.configure(lm=lm)
            
            # Initialize DSPy modules
            self.router_predictor = dspy.ChainOfThought(RouterSignature)
            
        except Exception as e:
            print(f"Warning: DSPy + Ollama setup failed, falling back to simple routing: {e}")
            self.router_predictor = None

    def route(self, inp: RouterInput) -> RouteDecision:
        if self.router_predictor is None:
            return self._simple_route(inp)
        
        try:
            # Use DSPy for intelligent routing
            result = self.router_predictor(
                user_input=inp.user_input,
                history=str(inp.history),
                context=str(inp.context)
            )
            
            # Parse the results
            task_type = self._parse_task_type(result.task_type)
            requires_cu = self._parse_computer_use(result.requires_computer_use)
            confidence = self._parse_confidence(result.confidence)
            suggested_tools = self._parse_tools(result.suggested_tools)
            
            # Select appropriate model based on task type and CUA requirements
            selected_model = self._select_model(task_type, requires_cu or inp.requires_computer_use)
            computer_use_model = self.computer_use_model if (requires_cu or inp.requires_computer_use) else None
            
            return RouteDecision(
                task=task_type,
                model=selected_model,
                computer_use_model=computer_use_model,
                reason=result.reasoning,
                confidence=confidence,
                suggested_tools=suggested_tools
            )
            
        except Exception as e:
            print(f"Warning: DSPy routing failed, falling back to simple routing: {e}")
            return self._simple_route(inp)
    
    def _select_model(self, task_type: TaskType, requires_computer_use: bool) -> str:
        """Select appropriate Ollama model based on task type and computer use requirements"""
        
        # For computer use actions, always use the computer use model
        if requires_computer_use or task_type == TaskType.COMPUTER_USE:
            return self.computer_use_model
        
        # For complex reasoning, use the reasoning model
        if task_type == TaskType.REASONING:
            return self.reasoning_model
            
        # For planning tasks that might involve computer use, use computer use model
        if task_type == TaskType.PLANNING:
            return self.computer_use_model
            
        # Default to chat model for simple interactions
        return self.chat_model
    
    def _simple_route(self, inp: RouterInput) -> RouteDecision:
        """Fallback routing logic when DSPy is unavailable"""
        user_input_lower = inp.user_input.lower()
        
        # Detect computer use requirements
        computer_use_keywords = ["click", "type", "screenshot", "mouse", "keyboard", "screen", "window", "desktop", "gui"]
        requires_cu = any(keyword in user_input_lower for keyword in computer_use_keywords) or inp.requires_computer_use
        
        # Simple keyword-based routing
        if requires_cu:
            task = TaskType.COMPUTER_USE
        elif any(word in user_input_lower for word in ["plan", "step", "strategy", "approach"]):
            task = TaskType.PLANNING
        elif any(word in user_input_lower for word in ["remember", "recall", "previous", "memory"]):
            task = TaskType.MEMORY_QUERY
        elif any(word in user_input_lower for word in ["execute", "run", "do", "perform"]):
            task = TaskType.TOOL_EXECUTION
        elif any(word in user_input_lower for word in ["think", "analyze", "reason", "explain"]):
            task = TaskType.REASONING
        else:
            task = TaskType.CHAT
        
        selected_model = self._select_model(task, requires_cu)
        computer_use_model = self.computer_use_model if requires_cu else None
        
        return RouteDecision(
            task=task,
            model=selected_model,
            computer_use_model=computer_use_model,
            reason=f"Simple keyword-based routing determined task: {task.value}, computer use: {requires_cu}",
            confidence=0.7,
            suggested_tools=[]
        )
    
    def _parse_task_type(self, task_str: str) -> TaskType:
        """Parse task type from DSPy output"""
        task_str = task_str.lower().strip()
        for task in TaskType:
            if task.value in task_str:
                return task
        return TaskType.CHAT
    
    def _parse_computer_use(self, cu_str: str) -> bool:
        """Parse computer use requirement from DSPy output"""
        cu_str = cu_str.lower().strip()
        return cu_str in ["true", "yes", "1"] or "computer" in cu_str
    
    def _parse_confidence(self, confidence_str: str) -> float:
        """Parse confidence score from DSPy output"""
        try:
            confidence_str = confidence_str.strip()
            if confidence_str.replace(".", "").isdigit():
                return float(confidence_str)
            import re
            match = re.search(r'\d+\.?\d*', confidence_str)
            if match:
                return min(1.0, max(0.0, float(match.group())))
        except:
            pass
        return 0.5
    
    def _parse_tools(self, tools_str: str) -> List[str]:
        """Parse suggested tools from DSPy output"""
        if not tools_str or tools_str.lower() in ["none", "n/a", "not applicable"]:
            return []
        
        tools = [tool.strip() for tool in tools_str.split(",")]
        return [tool for tool in tools if tool and tool.lower() not in ["none", "n/a"]]
