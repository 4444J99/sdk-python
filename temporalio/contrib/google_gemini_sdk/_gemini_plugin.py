"""Temporal plugin for Google Gemini SDK integration."""

from __future__ import annotations

import dataclasses
from typing import Any

from temporalio.contrib.google_gemini_sdk import _client_store
from temporalio.contrib.google_gemini_sdk._http_activity import gemini_api_call
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

    This plugin:

    - Stores the pre-built ``genai.Client`` so that workflows can retrieve it
      via :func:`~temporalio.contrib.google_gemini_sdk.get_gemini_client`
      without accessing ``os.environ`` or creating heavy objects in the sandbox.
    - Registers ``gemini_api_call`` — the durable HTTP transport invoked
      by :class:`~temporalio.contrib.google_gemini_sdk.TemporalHttpxClient`.
    - Configures the Pydantic data converter and sandbox passthrough modules.

    Example::

        gemini_client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"],
            http_options=temporal_http_options(
                start_to_close_timeout=timedelta(seconds=60),
            ),
        )
        plugin = GeminiPlugin(gemini_client=gemini_client)
        client = await Client.connect("localhost:7233", plugins=[plugin])
    """

    def __init__(
        self,
        gemini_client: Any,
    ) -> None:
        """Initialize the Gemini plugin.

        Args:
            gemini_client: A pre-built ``genai.Client`` instance.  Create it at
                worker startup (where ``os.environ`` is available) with
                ``http_options=temporal_http_options(...)`` so that its HTTP
                calls are routed through Temporal activities.
        """
        # Store the client in the passthrough'd module so the sandbox can see it.
        _client_store._gemini_client = gemini_client

        def workflow_runner(runner: WorkflowRunner | None) -> WorkflowRunner:
            if not runner:
                raise ValueError("No WorkflowRunner provided to GeminiPlugin.")
            if isinstance(runner, SandboxedWorkflowRunner):
                return dataclasses.replace(
                    runner,
                    restrictions=runner.restrictions.with_passthrough_modules(
                        # google.genai — so the workflow can call methods on the
                        # pre-built genai.Client and use types.GenerateContentConfig
                        "google.genai",
                        "google.api_core",
                        # pydantic internals — avoids "imported after initial
                        # workflow load" warnings when google.genai types
                        # trigger lazy pydantic schema compilation.
                        "pydantic_core",
                        "pydantic",
                        "annotated_types",
                        # The client store module — passthrough'd so the sandbox
                        # sees the same _gemini_client reference the worker set.
                        "temporalio.contrib.google_gemini_sdk._client_store",
                    ),
                )
            return runner

        super().__init__(
            name="GeminiPlugin",
            data_converter=self._configure_data_converter,
            activities=[gemini_api_call],
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
