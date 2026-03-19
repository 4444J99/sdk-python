"""First-class Temporal integration for the Google Gemini SDK.

.. warning::
    This module is experimental and may change in future versions.
    Use with caution in production environments.

This integration lets you use the Gemini SDK **exactly as you normally would**
while making every network call and every tool invocation **durable Temporal
activities**.

- :class:`GeminiPlugin` — creates and owns the ``genai.Client``, registers
  the HTTP transport activity, and configures the worker.  Pass the same args
  you would pass to ``genai.Client()`` — the plugin handles ``http_options``
  internally.
- :func:`activity_as_tool` — convert any ``@activity.defn`` function into a
  Gemini tool callable; Gemini's AFC invokes it as a Temporal activity.
- :func:`get_gemini_client` — retrieve the client inside a workflow.

Quickstart::

    # ---- worker setup (outside sandbox) ----
    plugin = GeminiPlugin(api_key=os.environ["GOOGLE_API_KEY"])

    @activity.defn
    async def get_weather(state: str) -> str: ...

    # ---- workflow (sandbox-safe) ----
    @workflow.defn
    class AgentWorkflow:
        @workflow.run
        async def run(self, query: str) -> str:
            client = get_gemini_client()
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[
                        activity_as_tool(
                            get_weather,
                            start_to_close_timeout=timedelta(seconds=30),
                        ),
                    ],
                ),
            )
            return response.text
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from temporalio.contrib.google_gemini_sdk._gemini_plugin import GeminiPlugin
from temporalio.contrib.google_gemini_sdk.workflow import (
    GeminiAgentWorkflowError,
    GeminiToolSerializationError,
    activity_as_tool,
)

__all__ = [
    "GeminiAgentWorkflowError",
    "GeminiPlugin",
    "GeminiToolSerializationError",
    "activity_as_tool",
]
