import os
import logging
from pathlib import Path
from agent import ComputerAgent

import uuid
from pprint import pprint
from agent.integrations.hud import run_full_dataset

from litellm import get_valid_models

# valid_models = get_valid_models(check_provider_endpoint=True)
# print(valid_models)

# Here you can set the model and tools for your agent.
# Computer use models: https://www.trycua.com/docs/agent-sdk/supported-agents/computer-use-agents
# Composed agent models: https://www.trycua.com/docs/agent-sdk/supported-agents/composed-agents
# Custom tools: https://www.trycua.com/docs/agent-sdk/custom-tools


async def main():
    agent_config = {
        "model": "anthropic/claude-3-7-sonnet-20250219",
        "trajectory_dir": str(Path("trajectories")),
        "only_n_most_recent_images": 3,
        "verbosity": logging.INFO,
    }

    from computer import Computer, VMProviderType

    # Connect to your existing cloud container
    computer = Computer(
        os_type="linux",
        provider_type=VMProviderType.CLOUD,
        name=os.getenv("CUA_CONTAINER_NAME") or "",
        api_key=os.getenv("CUA_API_KEY"),
        verbosity=logging.INFO,
    )

    agent_config["tools"] = [computer]

    # Create agent
    agent = ComputerAgent(**agent_config)

    tasks = [
        "Open the web browser and search for a repository named trycua/cua on GitHub."
    ]

    for i, task in enumerate(tasks):
        print(f"\nExecuting task {i}/{len(tasks)}: {task}")
        async for result in agent.run(task):
            print(result)
            pass

        print(f"\n✅ Task {i + 1}/{len(tasks)} completed: {task}")

    job_name = f"osworld-test-{str(uuid.uuid4())[:4]}"

    # Full dataset evaluation (runs via HUD's run_dataset under the hood)
    # See the documentation here: https://docs.trycua.com/docs/agent-sdk/integrations/hud#running-a-full-dataset
    results = await run_full_dataset(
        dataset="ddupont/OSWorld-Tiny-Public",
        job_name=job_name,
        **agent_config,
        max_concurrent=20,
        max_steps=50,
        # split="train[:5]"
    )

    # results is a list from hud.datasets.run_dataset; inspect/aggregate as needed
    print(f"Job: {job_name}")
    print(f"Total results: {len(results)}")
    pprint(results[:3])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
