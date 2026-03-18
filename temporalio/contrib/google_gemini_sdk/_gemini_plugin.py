"""Temporal plugin for Google Gemini SDK integration."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from google import genai

from temporalio.contrib.google_gemini_sdk._invoke_model_activity import (
    GeminiModelActivity,
)
from temporalio.contrib.google_gemini_sdk._model_activity_parameters import (
    ModelActivityParameters,
)
from temporalio.contrib.google_gemini_sdk.workflow import GeminiAgentWorkflowError
from temporalio.contrib.pydantic import (
    PydanticPayloadConverter as _DefaultPydanticPayloadConverter,
)
from temporalio.converter import DataConverter, DefaultPayloadConverter
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner


class GeminiPlugin(SimplePlugin):
    """A Temporal Worker Plugin configured for the Google Gemini SDK.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    This plugin configures:
    - Pydantic Payload Converter (required for Gemini SDK types).
    - Sandbox passthrough for ``google.genai`` and ``google.api_core`` modules.
    - The ``generate_content_activity`` model invocation activity.
    - ``GeminiAgentWorkflowError`` as a workflow failure exception type.

    Example:
        >>> plugin = GeminiPlugin()
        >>> client = await Client.connect("localhost:7233", plugins=[plugin])
        >>> async with Worker(
        ...     client,
        ...     task_queue="my-queue",
        ...     workflows=[MyAgentWorkflow],
        ...     activities=[my_tool_activity],
        ... ):
        ...     await asyncio.Event().wait()
    """

    def __init__(
        self,
        model_params: ModelActivityParameters | None = None,
        client_factory: Callable[[], genai.Client] | None = None,
        _model_activity: GeminiModelActivity | None = None,
    ) -> None:
        """Initialize the Gemini plugin.

        Args:
            model_params: Optional default parameters for model activity execution.
                Currently accepted but not applied automatically; pass ``model_params``
                directly to :func:`~temporalio.contrib.google_gemini_sdk.workflow.run_agent`.
            client_factory: Optional factory function for creating the Gemini client.
                Defaults to reading ``GOOGLE_API_KEY`` from the environment.
            _model_activity: Internal override for testing. Prefer using
                :class:`~temporalio.contrib.google_gemini_sdk.testing.GeminiEnvironment`
                instead of setting this directly.
        """
        model_activity = _model_activity or GeminiModelActivity(client_factory)

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if not runner:
                raise ValueError("No WorkflowRunner provided to GeminiPlugin.")
            if isinstance(runner, SandboxedWorkflowRunner):
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        "google.genai", "google.api_core"
                    ),
                )
            return runner

        super().__init__(
            name="GeminiPlugin",
            data_converter=self._configure_data_converter,
            activities=[model_activity.generate_content_activity],
            workflow_runner=workflow_runner,
            workflow_failure_exception_types=[GeminiAgentWorkflowError],
        )

    def _configure_data_converter(
        self, converter: DataConverter | None
    ) -> DataConverter:
        if converter is None:
            return DataConverter(
                payload_converter_class=_DefaultPydanticPayloadConverter
            )
        elif converter.payload_converter_class is DefaultPayloadConverter:
            return dataclasses.replace(
                converter,
                payload_converter_class=_DefaultPydanticPayloadConverter,
            )
        return converter
