"""Gemini model invocation activity for Temporal workflows."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from google import genai
from google.genai import types

from temporalio import activity
from temporalio.exceptions import ApplicationError

from temporalio.contrib.google_gemini_sdk._heartbeat_decorator import _auto_heartbeater


class FunctionCallOutput(BaseModel):
    """A single function call returned by the model."""

    name: str
    args: dict[str, Any]


class ActivityModelInput(BaseModel):
    """Input for the Gemini model invocation activity."""

    model: str
    system_instruction: str | None = None
    contents: list[types.Content]
    function_declarations: list[types.FunctionDeclaration] = []
    generation_config: types.GenerateContentConfig | None = None


class ActivityModelOutput(BaseModel):
    """Output from the Gemini model invocation activity."""

    text: str | None
    function_calls: list[FunctionCallOutput]
    model_content: types.Content  # Model turn to append to conversation history


def _default_client_factory() -> genai.Client:
    try:
        api_key = os.environ["GOOGLE_API_KEY"]
    except KeyError:
        raise ApplicationError(
            "GOOGLE_API_KEY environment variable is not set",
            non_retryable=True,
        )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _map_google_exception(exc: Exception) -> None:
    """Map google.api_core exceptions to ApplicationError with correct retryability."""
    try:
        from google.api_core import exceptions as google_exceptions
    except ImportError:
        return

    if isinstance(exc, google_exceptions.ResourceExhausted):
        raise ApplicationError(
            str(exc),
            type="ResourceExhausted",
            non_retryable=False,
        ) from exc
    elif isinstance(
        exc,
        (
            google_exceptions.DeadlineExceeded,
            google_exceptions.ServiceUnavailable,
            google_exceptions.InternalServerError,
        ),
    ):
        raise ApplicationError(
            str(exc), type=type(exc).__name__, non_retryable=False
        ) from exc
    elif isinstance(
        exc,
        (
            google_exceptions.InvalidArgument,
            google_exceptions.PermissionDenied,
            google_exceptions.NotFound,
        ),
    ):
        raise ApplicationError(
            str(exc), type=type(exc).__name__, non_retryable=True
        ) from exc


class GeminiModelActivity:
    """Temporal activity class for invoking the Gemini model.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.
    """

    def __init__(
        self,
        client_factory: Callable[[], genai.Client] | None = None,
    ) -> None:
        self._client_factory = client_factory or _default_client_factory

    @activity.defn
    @_auto_heartbeater
    async def generate_content_activity(
        self, input: ActivityModelInput
    ) -> ActivityModelOutput:
        """Invoke the Gemini model and return text and/or function calls."""
        try:
            client = self._client_factory()

            # Merge user's generation_config with required agent settings.
            # AFC must be disabled so Temporal owns tool execution.
            base_config = (
                input.generation_config
                if input.generation_config is not None
                else types.GenerateContentConfig()
            )
            config = base_config.model_copy(
                update={
                    "system_instruction": input.system_instruction,
                    "tools": (
                        [types.Tool(function_declarations=input.function_declarations)]
                        if input.function_declarations
                        else None
                    ),
                    "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                }
            )

            response = await client.aio.models.generate_content(
                model=input.model,
                contents=input.contents,
                config=config,
            )

            function_calls: list[FunctionCallOutput] = []
            text_parts: list[str] = []
            model_content: types.Content

            if response.candidates and response.candidates[0].content:
                model_content = response.candidates[0].content
                for part in model_content.parts:
                    if part.function_call:
                        function_calls.append(
                            FunctionCallOutput(
                                name=part.function_call.name,
                                args=(
                                    dict(part.function_call.args)
                                    if part.function_call.args
                                    else {}
                                ),
                            )
                        )
                    elif part.text:
                        text_parts.append(part.text)
            else:
                model_content = types.Content(role="model", parts=[])

            # Only include text if there are no function calls (avoids SDK warning)
            text = "".join(text_parts) if text_parts and not function_calls else None

            return ActivityModelOutput(
                text=text,
                function_calls=function_calls,
                model_content=model_content,
            )

        except Exception as exc:
            _map_google_exception(exc)
            raise
