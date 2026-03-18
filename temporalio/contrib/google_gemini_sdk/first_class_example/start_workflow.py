# ABOUTME: Client script to start the first-class Gemini agent workflow.
# No GOOGLE_API_KEY needed here — only the worker requires it.

import asyncio
import sys
import uuid

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

TASK_QUEUE = "gemini-first-class"


async def main() -> None:
    client = await Client.connect(
        "localhost:7233",
        data_converter=pydantic_data_converter,
    )

    query = sys.argv[1] if len(sys.argv) > 1 else "What's the weather like right now?"

    result = await client.execute_workflow(
        "WeatherAgentWorkflow",
        query,
        id=f"gemini-first-class-{uuid.uuid4()}",
        task_queue=TASK_QUEUE,
    )
    print(f"\nResult:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
