# ABOUTME: First-class Temporal + Gemini SDK integration demo.
# Demonstrates the clean developer experience: 3-line workflow, no manual loop,
# no dynamic activities, no tool registry, and no inspect hackery.

import asyncio
import json
from datetime import timedelta

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.envconfig import ClientConfig
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    import httpx

from temporalio.contrib.google_gemini_sdk import GeminiAgent, GeminiPlugin, activity_as_tool, run_agent


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
# Workflow — 3 lines of real logic, no manual loop, no print() debugging
# =============================================================================

TASK_QUEUE = "gemini-first-class"


@workflow.defn
class WeatherAgentWorkflow:
    """Durable agentic workflow powered by Gemini SDK and Temporal."""

    @workflow.run
    async def run(self, query: str) -> str:
        return await run_agent(
            GeminiAgent(
                model="gemini-2.5-flash",
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
            query,
        )


# =============================================================================
# Worker — register plugin + user activities, nothing else required
# =============================================================================


async def main() -> None:
    load_dotenv()

    plugin = GeminiPlugin()

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
