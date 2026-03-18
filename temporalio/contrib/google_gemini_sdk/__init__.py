"""First-class Temporal integration for the Google Gemini SDK.

.. warning::
    This module is experimental and may change in future versions.
    Use with caution in production environments.

Quickstart::

    from temporalio.contrib.google_gemini_sdk import (
        GeminiAgent,
        GeminiPlugin,
        activity_as_tool,
        run_agent,
    )

    @workflow.defn
    class MyAgentWorkflow:
        @workflow.run
        async def run(self, query: str) -> str:
            return await run_agent(
                GeminiAgent(
                    model="gemini-2.5-flash",
                    system_instruction="You are a helpful assistant.",
                    tools=[
                        activity_as_tool(my_tool, start_to_close_timeout=timedelta(seconds=30)),
                    ],
                ),
                query,
            )
"""

from temporalio.contrib.google_gemini_sdk._gemini_plugin import GeminiPlugin
from temporalio.contrib.google_gemini_sdk._model_activity_parameters import (
    ModelActivityParameters,
)
from temporalio.contrib.google_gemini_sdk.workflow import (
    ActivityTool,
    GeminiAgent,
    GeminiAgentWorkflowError,
    GeminiToolSerializationError,
    activity_as_tool,
    run_agent,
)
from temporalio.contrib.google_gemini_sdk import testing, workflow

__all__ = [
    "ActivityTool",
    "GeminiAgent",
    "GeminiAgentWorkflowError",
    "GeminiPlugin",
    "GeminiToolSerializationError",
    "ModelActivityParameters",
    "activity_as_tool",
    "run_agent",
    "testing",
    "workflow",
]
