"""
CUA-compliant tools for Columbus
"""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel


class ToolResult(BaseModel):
    """CUA-compliant tool result format"""
    success: bool = True
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    output_type: Literal["computer_call", "computer_call_output", "reasoning", "message"] = "message"
    
    def model_dump(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "output_type": self.output_type
        }


class ComputerInterface:
    """CUA-style computer interface for basic operations"""
    
    async def screenshot(self) -> Dict[str, Any]:
        """Take screenshot (simulated)"""
        return {
            "success": True,
            "file_path": "/tmp/screenshot.png",
            "message": "Screenshot captured (simulated)"
        }
    
    async def left_click(self, x: int, y: int) -> Dict[str, Any]:
        """Click at coordinates (simulated)"""
        return {
            "success": True,
            "x": x,
            "y": y,
            "message": f"Clicked at ({x}, {y}) (simulated)"
        }
    
    async def type(self, text: str) -> Dict[str, Any]:
        """Type text (simulated)"""
        return {
            "success": True,
            "text": text,
            "message": f"Typed '{text}' (simulated)"
        }


class ToolRegistry:
    """CUA-compliant tool registry"""
    
    def __init__(self):
        self.tools = {}
        self.computer = ComputerInterface()
    
    def get_available_tools(self) -> List[str]:
        """Return list of available CUA tools"""
        return [
            "computer",  # Core computer interface
            "screenshot", "mouse", "keyboard", 
            "reasoning", "file_system", "web_browser"
        ]
    
    async def execute(self, tool: str, action: str, execution_context: Optional[Dict] = None, **kwargs) -> ToolResult:
        """Execute a tool action following CUA patterns"""
        
        # Computer interface actions
        if tool == "computer":
            if action == "screenshot":
                result = await self.computer.screenshot()
                return ToolResult(
                    success=result["success"],
                    message=result["message"],
                    data=result,
                    output_type="computer_call_output"
                )
            
            elif action == "left_click":
                x = kwargs.get("x", 100)
                y = kwargs.get("y", 100)
                result = await self.computer.left_click(x, y)
                return ToolResult(
                    success=result["success"],
                    message=result["message"], 
                    data=result,
                    output_type="computer_call_output"
                )
            
            elif action == "type":
                text = kwargs.get("text", "")
                result = await self.computer.type(text)
                return ToolResult(
                    success=result["success"],
                    message=result["message"],
                    data=result,
                    output_type="computer_call_output"
                )
        
        # Legacy tool compatibility
        elif tool == "screenshot" and action == "capture":
            result = await self.computer.screenshot()
            return ToolResult(
                success=result["success"],
                message=result["message"],
                data=result,
                output_type="computer_call_output"
            )
        
        elif tool == "reasoning" and action == "analyze":
            return ToolResult(
                success=True,
                message="Analysis completed (simulated)",
                data={"analysis": "Task analyzed using CUA reasoning patterns"},
                output_type="reasoning"
            )
        
        # Default handler for other tools
        else:
            return ToolResult(
                success=True,
                message=f"Tool {tool}.{action} executed (simulated)",
                data={"tool": tool, "action": action, "args": kwargs},
                output_type="computer_call"
            )
