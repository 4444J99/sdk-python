"""Testing utilities for the Google Gemini SDK Temporal integration.

.. warning::
    This module is experimental and may change in future versions.
    Use with caution in production environments.

Example::

    from temporalio.contrib.google_gemini_sdk.testing import (
        GeminiEnvironment,
        MockGeminiResponse,
    )

    async def test_weather_agent():
        responses = [
            MockGeminiResponse.tool_call("get_weather_alerts", {"request": {"state": "CA"}}),
            MockGeminiResponse.text("No active weather alerts in California."),
        ]
        async with GeminiEnvironment(responses=responses) as env:
            client = await Client.connect("localhost:7233", plugins=[env.plugin])
            # ... run your workflow test
"""

from __future__ import annotations

from google.genai import types

from temporalio import activity
from temporalio.contrib.google_gemini_sdk._invoke_model_activity import (
    ActivityModelInput,
    ActivityModelOutput,
    FunctionCallOutput,
)
from temporalio.contrib.google_gemini_sdk._gemini_plugin import GeminiPlugin


class MockGeminiResponse:
    """Factory for constructing :class:`ActivityModelOutput` test fixtures.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.
    """

    __test__ = False

    @staticmethod
    def text(text: str) -> ActivityModelOutput:
        """Return an output that simulates a plain-text model response."""
        return ActivityModelOutput(
            text=text,
            function_calls=[],
            model_content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=text)],
            ),
        )

    @staticmethod
    def tool_call(name: str, args: dict) -> ActivityModelOutput:
        """Return an output that simulates a model requesting a tool call."""
        return ActivityModelOutput(
            text=None,
            function_calls=[FunctionCallOutput(name=name, args=args)],
            model_content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name=name, args=args)
                    )
                ],
            ),
        )


class TestGeminiModelActivity:
    """A mock replacement for :class:`GeminiModelActivity` that returns pre-configured responses.

    Responses are consumed in FIFO order. If no responses remain, raises ``IndexError``.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Example::

        responses = [
            MockGeminiResponse.tool_call("lookup", {"lookup": {"query": "CA"}}),
            MockGeminiResponse.text("Here is the answer."),
        ]
        activity_instance = TestGeminiModelActivity(responses)
        plugin = GeminiPlugin(_model_activity=activity_instance)
    """

    __test__ = False

    def __init__(self, responses: list[ActivityModelOutput]) -> None:
        self._responses = list(responses)

    @activity.defn
    async def generate_content_activity(
        self, input: ActivityModelInput
    ) -> ActivityModelOutput:
        """Return the next pre-configured response, ignoring the actual input."""
        if not self._responses:
            raise IndexError(
                "TestGeminiModelActivity has no more responses. "
                "Add more responses to the list passed to the constructor."
            )
        return self._responses.pop(0)


class GeminiEnvironment:
    """A test environment that wires up a mock Gemini model activity.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Example::

        responses = [
            MockGeminiResponse.text("Hello!"),
        ]
        async with GeminiEnvironment(responses=responses) as env:
            client = await Client.connect("localhost:7233", plugins=[env.plugin])
            async with Worker(
                client,
                task_queue="test-queue",
                workflows=[MyWorkflow],
                activities=[my_tool],
            ):
                result = await client.execute_workflow(...)
    """

    __test__ = False

    def __init__(
        self,
        responses: list[ActivityModelOutput] | None = None,
    ) -> None:
        test_activity = TestGeminiModelActivity(list(responses or []))
        self._plugin = GeminiPlugin(_model_activity=test_activity)

    async def __aenter__(self) -> GeminiEnvironment:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def applied_on_client(self, client: object) -> object:
        """Return the client unchanged (plugin is applied at connect time via ``plugins=``).

        This method exists for API symmetry with other test environment helpers.
        """
        return client

    @property
    def plugin(self) -> GeminiPlugin:
        """The :class:`GeminiPlugin` configured with the mock model activity."""
        return self._plugin
