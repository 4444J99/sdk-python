# ABOUTME: First-class Temporal + Gemini SDK integration demo.
#
# Key differences from example/durable_agent_worker.py:
#   - AFC is ENABLED: Gemini's SDK owns the agentic loop, no manual while-True
#   - Tools are plain @activity.defn functions; no registry, no dynamic activities
#   - activity_as_tool() makes each tool call a durable Temporal activity
#   - temporal_http_options() makes each model HTTP call a durable Temporal activity
#   - genai.Client is created once in main(), stored via GeminiPlugin
#   - Workflow retrieves it with get_gemini_client() — no os.environ in sandbox
#   - No run_agent(), no inspect hackery, no print() logging

import asyncio
import json
import os
from datetime import timedelta

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    import httpx
    from google.genai import types

from temporalio.contrib.google_gemini_sdk import (
    GeminiPlugin,
    activity_as_tool,
    get_gemini_client,
)


# =============================================================================
# System Instructions
# =============================================================================

SYSTEM_INSTRUCTIONS = """
You are a helpful agent that can use tools to help the user.
You will be given an input from the user and a list of tools to use.
You may or may not need to use the tools to satisfy the user ask.
If no tools are needed, respond in haikus.
"""

# =============================================================================
# Tool Definitions — plain @activity.defn functions, no registry required
# =============================================================================

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


class GetWeatherAlertsRequest(BaseModel):
    """Request model for getting weather alerts."""

    state: str = Field(description="Two-letter US state code (e.g. CA, NY)")


@activity.defn
async def get_weather_alerts(request: GetWeatherAlertsRequest) -> str:
    """Get weather alerts for a US state.

    Args:
        request: The request object containing:
            - state: Two-letter US state code (e.g. CA, NY)
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    url = f"{NWS_API_BASE}/alerts/active/area/{request.state}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        return json.dumps(response.json())


@activity.defn
async def get_ip_address() -> str:
    """Get the public IP address of the current machine."""
    async with httpx.AsyncClient() as client:
        response = await client.get("https://icanhazip.com")
        response.raise_for_status()
        return response.text.strip()


class GetLocationRequest(BaseModel):
    """Request model for getting location info from an IP address."""

    ipaddress: str = Field(description="An IP address")


@activity.defn
async def get_location_info(request: GetLocationRequest) -> str:
    """Get the location information for an IP address including city, state, and country.

    Args:
        request: The request object containing:
            - ipaddress: An IP address to look up
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://ip-api.com/json/{request.ipaddress}")
        response.raise_for_status()
        result = response.json()
        return f"{result['city']}, {result['regionName']}, {result['country']}"


# =============================================================================
# Workflow — natural Gemini SDK usage; AFC drives the loop; all calls are durable
# =============================================================================

TASK_QUEUE = "gemini-first-class"


@workflow.defn
class WeatherAgentWorkflow:
    """Durable agentic workflow powered by Gemini SDK and Temporal.

    The Gemini SDK's automatic function calling (AFC) drives the multi-turn
    agentic loop.  temporal_http_options() ensures every model HTTP call is a
    durable Temporal activity.  activity_as_tool() ensures every tool invocation
    is also a durable Temporal activity.  Together, every step of the agentic
    loop is visible in the workflow event history and recoverable after a crash.
    """

    @workflow.run
    async def run(self, query: str) -> str:
        # Retrieve the pre-built genai.Client that was created at worker
        # startup and stored via GeminiPlugin.  We cannot instantiate
        # genai.Client here because its constructor always reads os.environ
        # (for the API key, project ID, etc.), which Temporal's workflow
        # sandbox forbids.  get_gemini_client() reads from a passthrough'd
        # module, so the sandbox sees the real, pre-configured client object.
        client = get_gemini_client()
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                tools=[
                    activity_as_tool(
                        get_weather_alerts,
                        start_to_close_timeout=timedelta(seconds=30),
                    ),
                    activity_as_tool(
                        get_ip_address,
                        start_to_close_timeout=timedelta(seconds=10),
                    ),
                    activity_as_tool(
                        get_location_info,
                        start_to_close_timeout=timedelta(seconds=10),
                    ),
                ],
            ),
        )
        return response.text


# =============================================================================
# Worker — plugin owns client creation, start worker
# =============================================================================


async def main() -> None:
    load_dotenv()

    # GeminiPlugin creates the genai.Client internally, ensuring it is always
    # wired with temporal_http_options() so every LLM HTTP call runs as a
    # durable Temporal activity.  Pass the same args you'd pass to
    # genai.Client() — the plugin handles http_options for you.
    #
    # The client is created at worker startup (outside the sandbox) because
    # genai.Client() reads os.environ internally, which Temporal's workflow
    # sandbox forbids.  Workflows retrieve the pre-built client via
    # get_gemini_client().
    #
    # ── Sensitive activity field encryption ────────────────────────────
    # By default, credential headers (x-goog-api-key, authorization) in
    # activity inputs are encrypted in Temporal's event history so they're
    # never visible in plaintext.  Set SENSITIVE_ACTIVITY_FIELDS_ENCRYPTION_KEY in
    # your environment to enable this:
    #
    #   Generate a key (once, store in secret manager):
    #     python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    #
    # If you don't care about encrypting credentials (e.g. local dev with
    # a private Temporal server), pass sensitive_activity_fields=None:
    #
    #   plugin = GeminiPlugin(
    #       api_key=os.environ["GOOGLE_API_KEY"],
    #       sensitive_activity_fields=None,
    #   )
    # ─────────────────────────────────────────────────────────────────
    sensitive_fields_key = os.environ.get("SENSITIVE_ACTIVITY_FIELDS_ENCRYPTION_KEY")
    if sensitive_fields_key:
        # Encrypt credential headers in Temporal's event history.
        plugin = GeminiPlugin(
            api_key=os.environ["GOOGLE_API_KEY"],
            sensitive_activity_fields_encryption_key=sensitive_fields_key.encode(),
            start_to_close_timeout=timedelta(seconds=60),
        )
    else:
        # No encryption — credentials will be visible in the Temporal UI.
        plugin = GeminiPlugin(
            api_key=os.environ["GOOGLE_API_KEY"],
            sensitive_activity_fields=None,
            start_to_close_timeout=timedelta(seconds=60),
        )

    config = ClientConfig.load_client_connect_config()
    config.setdefault("target_host", "localhost:7233")
    client = await Client.connect(**config, plugins=[plugin])

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WeatherAgentWorkflow],
        activities=[get_weather_alerts, get_ip_address, get_location_info],
    ):
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
