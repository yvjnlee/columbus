"""
cua agent
"""

import logging

# Import the unified agent class and types
from agent import ComputerAgent
from computer import Computer, VMProviderType
from agent.callbacks import (
    ImageRetentionCallback,
    TrajectorySaverCallback,
    BudgetManagerCallback,
)

from columbus.config.settings import Config
from columbus.planning.planning import Planner

# Set up logging
logging.basicConfig(level=logging.INFO)


from agent.decorators import register_agent


@register_agent(models=r".*columbus.*", priority=10)
class ColumbusConfig:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.computer = Computer(
            os_type="linux",
            api_key=cfg.cua_api_key,
            name=getattr(cfg, "cua_container_name", None),
            provider_type=VMProviderType.CLOUD,
        )
        self.agent = ComputerAgent(
            model="omniparser+ollama_chat/llama3.2:3b-it-q4_K_M",
            instructions=(
                "You are a meticulous software operator. Prefer safe, deterministic actions. "
                "Always confirm via on-screen text before proceeding."
            ),
            tools=[self.computer],
            callbacks=[
                ImageRetentionCallback(only_n_most_recent_images=3),
                TrajectorySaverCallback("./trajectories"),
                BudgetManagerCallback(max_budget=10.0, raise_error=True),
            ],
            only_n_most_recent_images=3,
            verbosity=logging.DEBUG,
            trajectory_dir="trajectories",
            use_prompt_caching=True,
            max_trajectory_budget=1.0,
        )
        self.tasks = []
        self.history = []

    async def predict_step(self, messages, model, tools, **kwargs):
        # Format messages for the agent
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                formatted_messages.append({"role": msg.role, "content": msg.content})
            else:
                formatted_messages.append(msg)

        # Use the computer agent to process the step
        try:
            # Extract the latest user message as instruction
            instruction = formatted_messages[-1]["content"] if formatted_messages else ""

            # Run the agent with the instruction
            result = await self.agent.run_async(instruction)

            # Convert to expected output schema
            return {
                "output": [{"type": "text", "text": str(result)}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }
        except Exception as e:
            return {
                "output": [{"type": "text", "text": f"Error: {str(e)}"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}
            }

    async def predict_click(self, model, image_b64, instruction):
        # Optional: click-only capability
        return None

    def get_capabilities(self):
        return ["step"]


def main():
    cfg = ColumbusConfig(cfg=Config())
    cfg.agent.run("go to github and take a screenshot of the home page")


if __name__ == "__main__":
    main()
