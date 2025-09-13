"""
HUD OSWorld verified test integration for Columbus
"""

import os
from typing import Optional, Any, Dict, List

try:
    import hud

    HUD_AVAILABLE = True
except ImportError as e:
    HUD_AVAILABLE = False
    HUD_ERROR = str(e)

from columbus.cua_agent.agent import ComputerAgent


class HUDCompatibleAgent(ComputerAgent):
    """HUD-compatible wrapper for Columbus ComputerAgent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add HUD-expected attributes
        self.transfer_gyms = []  # HUD expects this for gym transfer
        self.action_space = None  # HUD expects this
        self.observation_space = None  # HUD expects this

    async def step(self, action: Any) -> Dict[str, Any]:
        """HUD-compatible step method"""
        if isinstance(action, str):
            # Convert string action to our format
            result = await self.run(goal=action)
            return {
                "observation": result.get("transcript", []),
                "reward": 1.0 if result.get("success", False) else 0.0,
                "done": result.get("success", False),
                "info": result,
            }
        return {"observation": [], "reward": 0.0, "done": False, "info": {}}

    def reset(self) -> Dict[str, Any]:
        """HUD-compatible reset method"""
        return {"observation": [], "info": {}}

    def close(self):
        """HUD-compatible close method"""
        pass


async def run_full_dataset(
    agent,
    dataset="OSWorld-Verified",
    split="train[:3]",
    max_concurrent=20,
    max_steps=50,
):
    """Run HUD OSWorld verified benchmark with Columbus agent"""
    if not HUD_AVAILABLE:
        raise ImportError(f"HUD not available: {HUD_ERROR}")

    # Check for HUD API key
    hud_api_key = os.getenv("HUD_API_KEY")
    if not hud_api_key:
        raise ImportError(
            "HUD_API_KEY environment variable not set. Get one from https://hud.ai"
        )

    try:
        # Load the OSWorld dataset
        taskset = await hud.load_taskset(taskset_id=dataset, api_key=hud_api_key)

        # Parse split parameter (e.g., "train[:3]" means first 3 tasks)
        if ":" in split:
            split_parts = split.split(":")
            if len(split_parts) > 1 and split_parts[1].rstrip("]"):
                limit = int(split_parts[1].rstrip("]"))
                # Limit the taskset to first N tasks
                if (
                    hasattr(taskset, "tasks")
                    and isinstance(taskset.tasks, list)
                    and len(taskset.tasks) > limit
                ):
                    taskset.tasks = taskset.tasks[:limit]

        # Run the benchmark using HUD's run_job with our HUD-compatible agent
        job_result = await hud.run_job(
            agent_cls=HUDCompatibleAgent,
            task_or_taskset=taskset,
            job_name=f"columbus_osworld_benchmark",
            agent_kwargs={
                "model": getattr(
                    agent.settings, "model_reasoning", "ollama_chat/llama3.2:8b"
                ),
                # Pass our Columbus agent's configuration
                "cfg": agent.cfg,
                "settings": agent.settings,
            },
            max_steps_per_task=max_steps,
            run_parallel=True,
            max_concurrent_tasks=max_concurrent,
            show_progress=True,
            verbose=True,
        )

        # Format results for Columbus
        results = []
        for task_result in job_result.task_results:
            results.append(
                {
                    "task_name": getattr(task_result.task, "name", "Unknown Task"),
                    "task_id": getattr(task_result.task, "id", "unknown"),
                    "success": task_result.success
                    if hasattr(task_result, "success")
                    else False,
                    "score": getattr(task_result, "score", None),
                    "steps": len(getattr(task_result, "trajectory", [])),
                    "error": getattr(task_result, "error", None),
                }
            )

        return results

    except Exception as e:
        raise ImportError(f"Failed to run HUD benchmark: {e}")
