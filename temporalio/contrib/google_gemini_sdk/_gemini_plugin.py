"""Temporal plugin for Google Gemini SDK integration."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from typing import Any

from temporalio.common import RetryPolicy
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

    This plugin creates and owns the ``genai.Client`` instance, ensuring it
    is always configured with :func:`temporal_http_options` so that every HTTP
    call (LLM model calls, streaming, etc.) is routed through a Temporal
    activity.  Workflows retrieve the client via :func:`get_gemini_client`.

    It also:

    - Registers the ``gemini_api_call`` activity (the durable HTTP transport).
    - Configures the Pydantic data converter and sandbox passthrough modules.

    All ``genai.Client`` constructor arguments (``api_key``, ``vertexai``,
    ``project``, ``credentials``, etc.) are forwarded via ``**kwargs`` — the
    plugin automatically handles ``http_options``.  If the Gemini SDK adds new
    constructor parameters in a future release, they are forwarded without
    any changes to this plugin.

    Example::

        plugin = GeminiPlugin(api_key=os.environ["GOOGLE_API_KEY"])
        client = await Client.connect("localhost:7233", plugins=[plugin])
        async with Worker(client, task_queue="q", workflows=[MyWorkflow],
                          activities=[my_tool]):
            ...

    Vertex AI example::

        plugin = GeminiPlugin(vertexai=True, project="my-project", location="us-central1")
    """

    def __init__(
        self,
        *,
        # ── Temporal activity configuration for model HTTP calls ─────────
        start_to_close_timeout: timedelta = timedelta(seconds=60),
        schedule_to_close_timeout: timedelta | None = None,
        heartbeat_timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        # ── genai.Client constructor args ────────────────────────────────
        # Forwarded directly to genai.Client().  The plugin adds
        # http_options automatically — do NOT pass it here.
        # See genai.Client docs for available kwargs: api_key, vertexai,
        # project, location, credentials, etc.
        **gemini_client_kwargs: Any,
    ) -> None:
        """Initialize the Gemini plugin.

        Args:
            start_to_close_timeout: Maximum time for a single model HTTP call
                activity.  Defaults to 60 seconds.
            schedule_to_close_timeout: Maximum time from scheduling to
                completion for model HTTP call activities.
            heartbeat_timeout: Maximum time between heartbeats for model HTTP
                call activities.
            retry_policy: Retry policy for failed model HTTP call activities.
            **gemini_client_kwargs: Forwarded to ``genai.Client()``.  Do NOT
                pass ``http_options`` — the plugin manages it internally.
                See ``genai.Client`` for available options (``api_key``,
                ``vertexai``, ``project``, ``location``, ``credentials``, …).
        """
        if "http_options" in gemini_client_kwargs:
            raise ValueError(
                "Do not pass http_options to GeminiPlugin — it configures "
                "temporal_http_options() internally.  Pass other genai.Client "
                "args (api_key, vertexai, project, etc.) directly."
            )

        # Create the genai.Client with temporal_http_options() so that all
        # HTTP calls go through a Temporal activity.  This happens at worker
        # startup (outside the sandbox) where os.environ is available.
        #
        # When no kwargs are provided (e.g. in test environments), skip client
        # creation — get_gemini_client() will raise at workflow runtime.
        gemini_client = None
        if gemini_client_kwargs:
            from google.genai import Client as GeminiClient

            from temporalio.contrib.google_gemini_sdk._temporal_httpx_client import (
                temporal_http_options,
            )

            gemini_client = GeminiClient(
                http_options=temporal_http_options(
                    start_to_close_timeout=start_to_close_timeout,
                    schedule_to_close_timeout=schedule_to_close_timeout,
                    heartbeat_timeout=heartbeat_timeout,
                    retry_policy=retry_policy,
                ),
                **gemini_client_kwargs,
            )

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
