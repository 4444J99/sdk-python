"""Workflow utilities for Google Gemini SDK integration with Temporal.

This module is loaded inside the Temporal workflow sandbox and therefore must
**not** import ``httpx``, ``google.genai``, or any other module that the
sandbox cannot handle.  HTTP-transport helpers live in
``_temporal_httpx_client.py`` which is loaded lazily outside the sandbox.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import google.auth.credentials
from google.genai import Client as GeminiClient
from google.genai.client import DebugConfig
from google.genai.types import HttpOptions, HttpOptionsDict

from temporalio import activity
from temporalio import workflow as temporal_workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.contrib.google_gemini_sdk._temporal_httpx_client import (
    temporal_http_options,
)
from temporalio.exceptions import ApplicationError, TemporalError
from temporalio.workflow import ActivityCancellationType, VersioningIntent


def activity_as_tool(
    fn: Callable,
    *,
    task_queue: str | None = None,
    schedule_to_close_timeout: timedelta | None = None,
    schedule_to_start_timeout: timedelta | None = None,
    start_to_close_timeout: timedelta | None = timedelta(seconds=30),
    heartbeat_timeout: timedelta | None = None,
    retry_policy: RetryPolicy | None = None,
    cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL,
    versioning_intent: VersioningIntent | None = None,
    priority: Priority = Priority.default,
) -> Callable:
    """Convert a Temporal activity into a Gemini-compatible async tool callable.

    .. warning::
        This API is experimental and may change in future versions.
        Use with caution in production environments.

    Returns an async callable with the same name, docstring, and type signature as
    ``fn``. When Gemini's automatic function calling (AFC) invokes the returned
    callable from within a Temporal workflow, the call is executed as a Temporal
    activity via :func:`workflow.execute_activity`. Each tool invocation therefore
    appears as a separate, durable entry in the workflow event history.

    Because AFC is left **enabled**, the Gemini SDK owns the agentic loop — no
    manual ``while`` loop or ``run_agent()`` helper is required. Pass the returned
    callable directly to ``GenerateContentConfig(tools=[...])``.

    For undocumented arguments see :py:meth:`workflow.execute_activity`.

    Args:
        fn: A Temporal activity function decorated with ``@activity.defn``.

    Returns:
        An async callable suitable for use as a Gemini tool.

    Raises:
        ApplicationError: If ``fn`` is not decorated with ``@activity.defn`` or
            has no activity name.
    """
    ret = activity._Definition.from_callable(fn)
    if not ret:
        raise ApplicationError(
            "Bare function without @activity.defn decorator is not supported",
            "invalid_tool",
        )
    if ret.name is None:
        raise ApplicationError(
            "Activity must have a name to be used as a Gemini tool",
            "invalid_tool",
        )

    # For class-based activities the first parameter is 'self'.  Partially apply
    # it so that Gemini inspects only the user-facing parameters when building
    # the function-call schema, while the worker resolves the real instance at
    # execution time.
    params = list(inspect.signature(fn).parameters.keys())
    schema_fn: Callable = fn
    if params and params[0] == "self":
        partial = functools.partial(fn, None)
        setattr(partial, "__name__", fn.__name__)
        partial.__annotations__ = getattr(fn, "__annotations__", {})
        setattr(
            partial,
            "__temporal_activity_definition",
            getattr(fn, "__temporal_activity_definition", None),
        )
        partial.__doc__ = fn.__doc__
        schema_fn = partial

    activity_name: str = ret.name
    schedule_kwargs: dict[str, Any] = {
        "task_queue": task_queue,
        "schedule_to_close_timeout": schedule_to_close_timeout,
        "schedule_to_start_timeout": schedule_to_start_timeout,
        "start_to_close_timeout": start_to_close_timeout,
        "heartbeat_timeout": heartbeat_timeout,
        "retry_policy": retry_policy,
        "cancellation_type": cancellation_type,
        "versioning_intent": versioning_intent,
        "priority": priority,
    }

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(schema_fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        activity_args = list(bound.arguments.values())
        return await temporal_workflow.execute_activity(
            activity_name,
            args=activity_args,
            **schedule_kwargs,
        )

    wrapper.__name__ = schema_fn.__name__
    wrapper.__doc__ = schema_fn.__doc__
    setattr(wrapper, "__signature__", inspect.signature(schema_fn))
    wrapper.__annotations__ = getattr(schema_fn, "__annotations__", {})

    return wrapper


def gemini_client(
    *,
    vertexai: bool | None = None,
    api_key: str | None = None,
    credentials: google.auth.credentials.Credentials | None = None,
    project: str | None = None,
    location: str | None = None,
    debug_config: DebugConfig | None = None,
    http_options: HttpOptions | HttpOptionsDict | None = None,
) -> GeminiClient:
    return GeminiClient(
        http_options=temporal_http_options(http_options=http_options),
        vertexai=vertexai,
        api_key=api_key,
        credentials=credentials,
        project=project,
        location=location,
        debug_config=debug_config,
    )


class GeminiAgentWorkflowError(TemporalError):
    """Raised when a Gemini-driven agentic workflow cannot complete normally.

    .. warning::
        This exception is experimental and may change in future versions.
        Use with caution in production environments.
    """


class GeminiToolSerializationError(TemporalError):
    """Raised when a tool's return value cannot be converted to a string.

    .. warning::
        This exception is experimental and may change in future versions.
        Use with caution in production environments.
    """
