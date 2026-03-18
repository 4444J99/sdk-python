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

# --- Type-checking imports (never executed at runtime) ---
# These give IDEs and type checkers full visibility into the lazy-loaded
# symbols so that autocomplete, go-to-definition, and hover docs work.
if TYPE_CHECKING:
    from temporalio.contrib.google_gemini_sdk._gemini_plugin import (
        GeminiPlugin as GeminiPlugin,
    )
    from temporalio.contrib.google_gemini_sdk._temporal_httpx_client import (
        TemporalHttpxClient as TemporalHttpxClient,
        temporal_http_options as temporal_http_options,
    )

# --- Sandbox-safe imports (loaded eagerly at runtime) ---
# These modules have NO httpx / google.genai imports and are safe to load
# inside the Temporal workflow sandbox.
from temporalio.contrib.google_gemini_sdk._client_store import get_gemini_client
from temporalio.contrib.google_gemini_sdk.workflow import (
    GeminiAgentWorkflowError,
    GeminiToolSerializationError,
    activity_as_tool,
)

__all__ = [
    "GeminiAgentWorkflowError",
    "GeminiPlugin",
    "GeminiToolSerializationError",
    "TemporalHttpxClient",
    "activity_as_tool",
    "get_gemini_client",
    "temporal_http_options",
]


# --- Lazy imports for httpx-dependent symbols ---
# GeminiPlugin, TemporalHttpxClient, and temporal_http_options all transitively
# import httpx (via _gemini_plugin → _http_activity → httpx, and via
# _temporal_httpx_client → httpx).  They must NOT be loaded inside the workflow
# sandbox.  They are imported lazily so that sandbox-safe imports like
# ``from temporalio.contrib.google_gemini_sdk import activity_as_tool``
# never trigger an httpx import.
def __getattr__(name: str):  # type: ignore[override]
    _lazy = {
        "GeminiPlugin": (
            "temporalio.contrib.google_gemini_sdk._gemini_plugin",
            "GeminiPlugin",
        ),
        "TemporalHttpxClient": (
            "temporalio.contrib.google_gemini_sdk._temporal_httpx_client",
            "TemporalHttpxClient",
        ),
        "temporal_http_options": (
            "temporalio.contrib.google_gemini_sdk._temporal_httpx_client",
            "temporal_http_options",
        ),
    }
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name]
        mod = importlib.import_module(module_path)
        value = getattr(mod, attr)
        # Cache on the module so __getattr__ is only called once per name.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
