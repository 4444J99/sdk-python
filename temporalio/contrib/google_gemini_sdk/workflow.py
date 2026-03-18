"""Workflow utilities for Gemini SDK integration with Temporal."""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from collections.abc import Callable

from google.genai import types

from temporalio import activity
from temporalio import workflow as temporal_workflow
from temporalio.common import Priority, RetryPolicy
from temporalio.exceptions import ApplicationError, TemporalError
from temporalio.workflow import ActivityCancellationType, VersioningIntent

from temporalio.contrib.google_gemini_sdk._invoke_model_activity import (
    ActivityModelInput,
    GeminiModelActivity,
)
from temporalio.contrib.google_gemini_sdk._model_activity_parameters import (
    ModelActivityParameters,
)


@dataclass
class ActivityTool:
    """A Temporal activity wrapped as a Gemini tool.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.
    """

    function_declaration: types.FunctionDeclaration
    activity_name: str
    schedule_kwargs: dict[str, Any]


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
    api_option: str = "GEMINI_API",
) -> ActivityTool:
    """Convert a Temporal activity function into a Gemini tool for use with :func:`run_agent`.

    .. warning::
        This API is experimental and may change in future versions.
        Use with caution in production environments.

    Args:
        fn: A Temporal activity function decorated with ``@activity.defn``.
        task_queue: Specific task queue to use for this tool's activity.
        schedule_to_close_timeout: Maximum time from scheduling to completion.
        schedule_to_start_timeout: Maximum time from scheduling to starting.
        start_to_close_timeout: Maximum time for the activity to complete.
        heartbeat_timeout: Maximum time between heartbeats.
        retry_policy: Policy for retrying failed activities.
        cancellation_type: How the activity handles cancellation.
        versioning_intent: Versioning intent for the activity.
        priority: Priority for the activity execution.
        api_option: Gemini API option for schema generation. Defaults to ``"GEMINI_API"``.

    Returns:
        An :class:`ActivityTool` wrapping the activity with its Gemini schema.

    Raises:
        ApplicationError: If the function is not properly decorated as a Temporal activity.

    Example:
        >>> @activity.defn
        ... async def get_weather(request: WeatherRequest) -> str: ...
        >>>
        >>> tool = activity_as_tool(
        ...     get_weather,
        ...     start_to_close_timeout=timedelta(seconds=30),
        ... )
    """
    ret = activity._Definition.from_callable(fn)
    if not ret:
        raise ApplicationError(
            "Bare function without @activity.defn decorator is not supported",
            "invalid_tool",
        )
    if ret.name is None:
        raise ApplicationError(
            "Activity must have a name to be used as a tool",
            "invalid_tool",
        )

    # If the callable has a 'self' parameter (class-based activity), partially apply it
    # so that FunctionDeclaration schema generation ignores the self param.
    # The actual instance is resolved at execution time by the worker.
    params = list(inspect.signature(fn).parameters.keys())
    schema_fn = fn
    if len(params) > 0 and params[0] == "self":
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

    function_declaration = types.FunctionDeclaration.from_callable_with_api_option(
        callable=schema_fn,
        api_option=api_option,
    )

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

    return ActivityTool(
        function_declaration=function_declaration,
        activity_name=ret.name,
        schedule_kwargs=schedule_kwargs,
    )


@dataclass
class GeminiAgent:
    """Configuration for a Gemini-powered agentic loop.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Example:
        >>> agent = GeminiAgent(
        ...     model="gemini-2.5-flash",
        ...     system_instruction="You are a helpful assistant.",
        ...     tools=[
        ...         activity_as_tool(get_weather, start_to_close_timeout=timedelta(seconds=30)),
        ...     ],
        ... )
    """

    model: str
    system_instruction: str | None = None
    tools: list[ActivityTool] = field(default_factory=list)
    generation_config: types.GenerateContentConfig | None = None
    api_option: str = "GEMINI_API"


async def run_agent(
    agent: GeminiAgent,
    initial_message: str,
    *,
    model_params: ModelActivityParameters | None = None,
    max_turns: int = 10,
) -> str:
    """Run the Gemini agentic loop inside a Temporal workflow.

    .. warning::
        This API is experimental and may change in future versions.
        Use with caution in production environments.

    Each model call and each tool invocation becomes a separate Temporal activity,
    giving full workflow history visibility and crash recovery.

    Args:
        agent: The :class:`GeminiAgent` configuration.
        initial_message: The user's initial query.
        model_params: Optional parameters for configuring model activity execution.
        max_turns: Maximum number of model call + tool execution rounds before raising.

    Returns:
        The model's final text response.

    Raises:
        GeminiAgentWorkflowError: If ``max_turns`` is exceeded or an unknown tool is called.
    """
    if model_params is None:
        model_params = ModelActivityParameters()

    history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=initial_message)])
    ]

    function_declarations = [t.function_declaration for t in agent.tools]
    activity_tools: dict[str, ActivityTool] = {
        t.function_declaration.name: t for t in agent.tools
    }

    # Build kwargs for the model activity execution
    model_kwargs: dict[str, Any] = {
        "task_queue": model_params.task_queue,
        "schedule_to_close_timeout": model_params.schedule_to_close_timeout,
        "schedule_to_start_timeout": model_params.schedule_to_start_timeout,
        "start_to_close_timeout": model_params.start_to_close_timeout,
        "heartbeat_timeout": model_params.heartbeat_timeout,
        "retry_policy": model_params.retry_policy,
        "cancellation_type": model_params.cancellation_type,
        "versioning_intent": model_params.versioning_intent,
        "priority": model_params.priority,
    }

    for _ in range(max_turns):
        model_input = ActivityModelInput(
            model=agent.model,
            system_instruction=agent.system_instruction,
            contents=history,
            function_declarations=function_declarations,
            generation_config=agent.generation_config,
        )

        if model_params.use_local_activity:
            result = await temporal_workflow.execute_local_activity_method(
                GeminiModelActivity.generate_content_activity,
                model_input,
                **{
                    k: v
                    for k, v in model_kwargs.items()
                    if k
                    not in (
                        "task_queue",
                        "schedule_to_start_timeout",
                        "versioning_intent",
                    )
                },
            )
        else:
            result = await temporal_workflow.execute_activity_method(
                GeminiModelActivity.generate_content_activity,
                model_input,
                **model_kwargs,
            )

        if result.function_calls:
            history.append(result.model_content)

            for fc in result.function_calls:
                tool = activity_tools.get(fc.name)
                if tool is None:
                    raise GeminiAgentWorkflowError(
                        f"Model called unknown tool '{fc.name}'. "
                        f"Available tools: {list(activity_tools.keys())}"
                    )

                # Extract positional args from the dict Gemini returns.
                # Gemini wraps each parameter under its name, e.g.:
                #   get_weather_alerts(request: WeatherRequest) → {"request": {"state": "CA"}}
                # list(args.values()) unwraps to [{"state": "CA"}], which Temporal
                # then deserializes as the activity's Pydantic parameter type.
                # For no-arg activities, args={} → dispatch_args=[] (no args passed).
                dispatch_args = list(fc.args.values())

                try:
                    tool_result = await temporal_workflow.execute_activity(
                        tool.activity_name,
                        args=dispatch_args,
                        **tool.schedule_kwargs,
                    )
                except Exception as exc:
                    raise GeminiAgentWorkflowError(
                        f"Tool '{fc.name}' raised an error: {exc}"
                    ) from exc

                try:
                    result_str = str(tool_result)
                except Exception as exc:
                    raise GeminiToolSerializationError(
                        f"Tool '{fc.name}' returned a value that could not be converted to str"
                    ) from exc

                history.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=fc.name,
                                response={"result": result_str},
                            )
                        ],
                    )
                )
        else:
            return result.text or ""

    raise GeminiAgentWorkflowError(
        f"Agent exceeded maximum turns ({max_turns}) without producing a final response."
    )


class GeminiAgentWorkflowError(TemporalError):
    """Raised when the Gemini agent loop cannot complete normally.

    .. warning::
        This exception is experimental and may change in future versions.
        Use with caution in production environments.

    This is raised when:
    - The agent exceeds ``max_turns`` without returning a text response.
    - The model calls a tool that was not registered.
    - A tool activity raises an unexpected error.
    """


class GeminiToolSerializationError(TemporalError):
    """Raised when a tool's return value cannot be converted to a string.

    .. warning::
        This exception is experimental and may change in future versions.
        Use with caution in production environments.

    All tool outputs are converted to strings before being sent back to the model.
    If ``str(result)`` raises, this exception is raised instead.
    """
