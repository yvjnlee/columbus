"""
Enhanced planning layer with DSPy and CUA integration
"""

import dspy
import litellm
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from columbus.config.settings import Config


class PlanStep(BaseModel):
    step_number: int
    thought: str
    tool: Optional[str] = None
    action: Optional[str] = None
    args: Optional[Dict[str, Any]] = Field(default_factory=dict)
    requires_computer_use: bool = False
    expected_outcome: str = ""


class PlanningContext(BaseModel):
    goal: str
    context: Optional[str] = None
    available_tools: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)


class PlannerSignature(dspy.Signature):
    """Create a detailed step-by-step plan to accomplish the given goal."""
    
    goal = dspy.InputField(desc="The main goal to accomplish")
    context = dspy.InputField(desc="Additional context about the task")
    available_tools = dspy.InputField(desc="List of available tools and their capabilities")
    constraints = dspy.InputField(desc="Any constraints or limitations to consider")
    
    plan_steps = dspy.OutputField(desc="Detailed step-by-step plan as JSON array with fields: step_number, thought, tool, action, args, requires_computer_use, expected_outcome")
    reasoning = dspy.OutputField(desc="Explanation of the planning approach and key considerations")


class ComputerUseAnalysisSignature(dspy.Signature):
    """Analyze if a plan step requires computer use interactions."""
    
    step_description = dspy.InputField(desc="Description of the planned step")
    tool_name = dspy.InputField(desc="Tool being used in this step")
    action_name = dspy.InputField(desc="Specific action being performed")
    
    requires_computer_use = dspy.OutputField(desc="True if step involves GUI interaction, clicking, typing, screenshots, or computer control")
    computer_use_type = dspy.OutputField(desc="Type of computer interaction: none, gui_interaction, file_system, screenshot, keyboard_input, mouse_action")
    reasoning = dspy.OutputField(desc="Explanation for the computer use determination")


class Planner:
    def __init__(self, model: str = "ollama_chat/llama3.2:8b", cfg: Optional[Config] = None):
        self.model = model
        self.cfg = cfg or Config()
        
        # Configure Ollama connection
        ollama_base = getattr(self.cfg, 'ollama_base_url', 'http://localhost:11434')
        
        try:
            # Configure DSPy with Ollama for planning
            lm = dspy.LM(model=self.model, api_base=ollama_base, max_tokens=1000)
            dspy.configure(lm=lm)
            
            # Initialize DSPy modules
            self.planner = dspy.ChainOfThought(PlannerSignature)
            self.cu_analyzer = dspy.ChainOfThought(ComputerUseAnalysisSignature)
            
        except Exception as e:
            print(f"Warning: DSPy planner setup failed, using fallback planning: {e}")
            self.planner = None
            self.cu_analyzer = None
    
    def plan(self, goal: str, context: Optional[str] = None, available_tools: Optional[List[str]] = None) -> List[PlanStep]:
        """Create a comprehensive plan for achieving the goal"""
        planning_context = PlanningContext(
            goal=goal,
            context=context or "",
            available_tools=available_tools or self._get_default_tools(),
            constraints=self._get_constraints(),
            user_preferences=self._get_user_preferences()
        )
        
        if self.planner is None:
            return self._simple_plan(planning_context)
        
        try:
            # Use DSPy for intelligent planning
            result = self.planner(
                goal=planning_context.goal,
                context=planning_context.context,
                available_tools=str(planning_context.available_tools),
                constraints=str(planning_context.constraints)
            )
            
            # Parse and enhance the plan
            steps = self._parse_plan_steps(result.plan_steps)
            
            # Analyze each step for computer use requirements
            enhanced_steps = []
            for step in steps:
                if self.cu_analyzer:
                    try:
                        cu_result = self.cu_analyzer(
                            step_description=step.thought,
                            tool_name=step.tool or "none",
                            action_name=step.action or "none"
                        )
                        step.requires_computer_use = self._parse_cu_requirement(cu_result.requires_computer_use)
                    except:
                        # Fallback computer use detection
                        step.requires_computer_use = self._detect_computer_use_fallback(step)
                else:
                    step.requires_computer_use = self._detect_computer_use_fallback(step)
                
                enhanced_steps.append(step)
            
            return enhanced_steps
            
        except Exception as e:
            print(f"Warning: DSPy planning failed, using simple planning: {e}")
            return self._simple_plan(planning_context)
    
    def _simple_plan(self, context: PlanningContext) -> List[PlanStep]:
        """Fallback planning when DSPy is unavailable"""
        goal_lower = context.goal.lower()
        steps = []
        
        # Basic planning heuristics
        if "screenshot" in goal_lower or "capture" in goal_lower:
            steps.append(PlanStep(
                step_number=1,
                thought="Take a screenshot to capture current screen state",
                tool="screenshot",
                action="capture",
                requires_computer_use=True,
                expected_outcome="Screenshot saved successfully"
            ))
        
        if any(word in goal_lower for word in ["click", "press", "button"]):
            steps.append(PlanStep(
                step_number=len(steps) + 1,
                thought="Perform clicking action on specified element",
                tool="mouse",
                action="click",
                requires_computer_use=True,
                expected_outcome="Element clicked successfully"
            ))
        
        if any(word in goal_lower for word in ["type", "enter", "input"]):
            steps.append(PlanStep(
                step_number=len(steps) + 1,
                thought="Type text input as specified",
                tool="keyboard",
                action="type",
                requires_computer_use=True,
                expected_outcome="Text entered successfully"
            ))
        
        # Default step if no specific actions detected
        if not steps:
            steps.append(PlanStep(
                step_number=1,
                thought=f"Analyze and approach the goal: {context.goal}",
                tool="reasoning",
                action="analyze",
                requires_computer_use=False,
                expected_outcome="Goal understood and approach determined"
            ))
        
        return steps
    
    def _parse_plan_steps(self, plan_json: str) -> List[PlanStep]:
        """Parse plan steps from JSON string"""
        import json
        
        try:
            # Try to parse as JSON
            if plan_json.strip().startswith('['):
                plan_data = json.loads(plan_json)
            else:
                # If not valid JSON, try to extract from text
                plan_data = self._extract_steps_from_text(plan_json)
            
            steps = []
            for i, step_data in enumerate(plan_data, 1):
                if isinstance(step_data, dict):
                    step = PlanStep(
                        step_number=step_data.get('step_number', i),
                        thought=step_data.get('thought', ''),
                        tool=step_data.get('tool'),
                        action=step_data.get('action'),
                        args=step_data.get('args', {}),
                        requires_computer_use=step_data.get('requires_computer_use', False),
                        expected_outcome=step_data.get('expected_outcome', '')
                    )
                    steps.append(step)
            
            return steps
            
        except Exception as e:
            print(f"Error parsing plan steps: {e}")
            # Return a single analysis step as fallback
            return [PlanStep(
                step_number=1,
                thought="Analyze the goal and determine next steps",
                tool="reasoning",
                action="analyze",
                expected_outcome="Understanding of the task requirements"
            )]
    
    def _extract_steps_from_text(self, text: str) -> List[Dict]:
        """Extract steps from unstructured text"""
        steps = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines, 1):
            if line.strip() and not line.startswith('#'):
                steps.append({
                    'step_number': i,
                    'thought': line.strip(),
                    'expected_outcome': 'Step completed successfully'
                })
        
        return steps
    
    def _detect_computer_use_fallback(self, step: PlanStep) -> bool:
        """Fallback computer use detection"""
        computer_use_keywords = [
            "click", "type", "screenshot", "mouse", "keyboard", "screen", 
            "window", "desktop", "gui", "interface", "button", "menu",
            "drag", "scroll", "select", "copy", "paste"
        ]
        
        text_to_check = f"{step.thought} {step.tool or ''} {step.action or ''}".lower()
        return any(keyword in text_to_check for keyword in computer_use_keywords)
    
    def _parse_cu_requirement(self, cu_str: str) -> bool:
        """Parse computer use requirement from DSPy output"""
        cu_str = cu_str.lower().strip()
        return cu_str in ["true", "yes", "1"] or "computer" in cu_str or "gui" in cu_str
    
    def _get_default_tools(self) -> List[str]:
        """Get default available tools"""
        return [
            "screenshot", "mouse", "keyboard", "file_system", 
            "web_browser", "text_editor", "terminal", "reasoning"
        ]
    
    def _get_constraints(self) -> List[str]:
        """Get default constraints"""
        return [
            "Prioritize user safety and data privacy",
            "Minimize system disruption",
            "Use appropriate models for computer use actions",
            "Verify actions before execution when possible"
        ]
    
    def _get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from config"""
        return {
            "verbose_output": getattr(self.cfg, 'verbose_output', False),
            "confirm_destructive_actions": getattr(self.cfg, 'confirm_destructive_actions', True),
            "max_steps": getattr(self.cfg, 'max_planning_steps', 10)
        }
