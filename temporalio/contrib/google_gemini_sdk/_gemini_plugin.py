"""Temporal plugin for Google Gemini SDK integration."""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from typing import Any

from temporalio.common import RetryPolicy
from temporalio.contrib.google_gemini_sdk import _client_store
from temporalio.contrib.google_gemini_sdk._http_activity import (
    HttpRequestData,
    gemini_api_call,
)
from temporalio.contrib.google_gemini_sdk.workflow import GeminiAgentWorkflowError
from temporalio.contrib.pydantic import (
    PydanticPayloadConverter as _DefaultPydanticPayloadConverter,
)
from typing import Sequence

import temporalio.api.common.v1
from temporalio.converter import DataConverter, DefaultPayloadConverter, PayloadCodec
from temporalio.plugin import SimplePlugin
from temporalio.worker import WorkflowRunner
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

#: Default set of HTTP header keys that contain credentials and should be
#: encrypted in Temporal's event history.  Pass to ``sensitive_activity_fields``
#: to use these defaults, or provide your own set.
DEFAULT_SENSITIVE_HEADER_KEYS: set[str] = {"x-goog-api-key", "authorization"}

_ENCRYPTION_KEY_HELP = """\
sensitive_activity_fields_encryption_key must be a Fernet key (44 URL-safe base64 bytes).

To generate one:

  Local dev / quick start:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

  Production:
    Store the key in a secret manager (e.g. Google Secret Manager, AWS Secrets
    Manager, HashiCorp Vault) and load it at worker startup.  The same key must
    be used by every worker and client that reads/writes this Temporal namespace.

Then pass it to GeminiPlugin:

    plugin = GeminiPlugin(
        api_key=os.environ["GOOGLE_API_KEY"],
        sensitive_activity_fields_encryption_key=os.environ["SENSITIVE_ACTIVITY_FIELDS_ENCRYPTION_KEY"].encode(),
    )

If you don't need credential encryption (e.g. local dev with a private Temporal
server), pass sensitive_activity_fields=None to skip it entirely:

    plugin = GeminiPlugin(
        api_key=os.environ["GOOGLE_API_KEY"],
        sensitive_activity_fields=None,
    )"""


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
    - Optionally configures a **field-level encryption codec** that encrypts
      credential headers (``x-goog-api-key``, ``authorization``) within
      ``HttpRequestData`` payloads so they never appear in plaintext in
      Temporal's event history.  All other fields remain human-readable.
    - Configures sandbox passthrough modules.

    All ``genai.Client`` constructor arguments (``api_key``, ``vertexai``,
    ``project``, ``credentials``, etc.) are forwarded via ``**kwargs`` — the
    plugin automatically handles ``http_options``.

    Example with encryption (recommended for production)::

        plugin = GeminiPlugin(
            api_key=os.environ["GOOGLE_API_KEY"],
            sensitive_activity_fields_encryption_key=os.environ["SENSITIVE_ACTIVITY_FIELDS_ENCRYPTION_KEY"].encode(),
        )

    Example without encryption (local dev with private Temporal server)::

        plugin = GeminiPlugin(
            api_key=os.environ["GOOGLE_API_KEY"],
            sensitive_activity_fields=None,
        )

    Vertex AI example::

        plugin = GeminiPlugin(
            vertexai=True, project="my-project", location="us-central1",
            sensitive_activity_fields_encryption_key=os.environ["SENSITIVE_ACTIVITY_FIELDS_ENCRYPTION_KEY"].encode(),
        )
    """

    def __init__(
        self,
        *,
        # ── Temporal activity configuration for model HTTP calls ─────────
        start_to_close_timeout: timedelta = timedelta(seconds=60),
        schedule_to_close_timeout: timedelta | None = None,
        heartbeat_timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
        # ── Sensitive activity field encryption ──────────────────────────
        sensitive_activity_fields: set[str] | None = DEFAULT_SENSITIVE_HEADER_KEYS,
        sensitive_activity_fields_encryption_key: bytes | None = None,
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
            sensitive_activity_fields: Set of HTTP header keys to encrypt in
                Temporal's event history.  Defaults to
                ``DEFAULT_SENSITIVE_HEADER_KEYS`` (``{"x-goog-api-key",
                "authorization"}``).  Pass ``None`` to disable encryption
                entirely (e.g. local dev with a private Temporal server).
            sensitive_activity_fields_encryption_key: A Fernet key for
                encrypting the fields specified above.  Generate one with
                ``cryptography.fernet.Fernet.generate_key()``.  Must be the
                same across all workers and clients for a given namespace.
                Required when ``sensitive_activity_fields`` is not ``None``.
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

        # ── Build the DataConverter ──────────────────────────────────────
        if sensitive_activity_fields is not None:
            if sensitive_activity_fields_encryption_key is None:
                raise ValueError(
                    "sensitive_activity_fields_encryption_key is required when "
                    "sensitive_activity_fields is set.\n\n" + _ENCRYPTION_KEY_HELP
                )

            from temporalio.contrib.google_gemini_sdk._sensitive_fields_codec import (
                make_sensitive_fields_data_converter,
            )

            self._data_converter = make_sensitive_fields_data_converter(
                model_configs={
                    HttpRequestData: {"headers": sensitive_activity_fields},
                },
                encryption_key=sensitive_activity_fields_encryption_key,
            )
        else:
            # No encryption — use plain Pydantic converter.
            # Credential headers will be visible in Temporal's event history.
            self._data_converter = DataConverter(
                payload_converter_class=_DefaultPydanticPayloadConverter
            )

        # ── Create the genai.Client ──────────────────────────────────────
        # Uses temporal_http_options() so all HTTP calls go through a Temporal
        # activity.  Created at worker startup (outside the sandbox) where
        # os.environ is available.
        #
        # When no kwargs are provided (e.g. test environments), skip client
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
                        "google.genai",
                        "google.api_core",
                        "pydantic_core",
                        "pydantic",
                        "annotated_types",
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
        if converter is not None and converter.payload_codec is not None:
            # Caller has their own codec — chain ours first, then theirs.
            if self._data_converter.payload_codec is not None:
                return dataclasses.replace(
                    self._data_converter,
                    payload_codec=_CompositeCodec(
                        [self._data_converter.payload_codec, converter.payload_codec]
                    ),
                )
        return self._data_converter


class _CompositeCodec(PayloadCodec):
    """Chains multiple codecs in order (encode: left→right, decode: right→left)."""

    def __init__(self, codecs: list[PayloadCodec]) -> None:
        self._codecs = codecs

    async def encode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:
        result = list(payloads)
        for codec in self._codecs:
            result = await codec.encode(result)
        return result

    async def decode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:
        result = list(payloads)
        for codec in reversed(self._codecs):
            result = await codec.decode(result)
        return result
