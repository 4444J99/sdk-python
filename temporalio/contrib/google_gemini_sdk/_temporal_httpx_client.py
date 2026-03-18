"""Temporal-aware httpx client and helpers.

This module imports ``httpx`` and must **never** be loaded inside the Temporal
workflow sandbox.  It is imported lazily by ``__init__.py`` and at module level
in worker code (outside the sandbox).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import httpx
from google.genai import types

from temporalio import workflow as temporal_workflow
from temporalio.common import RetryPolicy

from temporalio.contrib.google_gemini_sdk._http_activity import (
    HttpRequestData,
    gemini_api_call,
)


class _NoOpAsyncTransport(httpx.AsyncBaseTransport):
    """Placeholder transport; never called because TemporalHttpxClient.send() intercepts all requests."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise RuntimeError(
            "_NoOpAsyncTransport.handle_async_request() should never be "
            "called.  All requests are routed through "
            "workflow.execute_activity via TemporalHttpxClient.send()."
        )


class TemporalHttpxClient(httpx.AsyncClient):
    """An ``httpx.AsyncClient`` that routes all HTTP calls through a Temporal activity.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    Pass an instance to ``genai.Client(http_options=types.HttpOptions(httpx_async_client=...))``.
    Every HTTP request the Gemini SDK makes (model calls, streaming responses, etc.)
    is dispatched as
    :func:`~temporalio.contrib.google_gemini_sdk._http_activity.gemini_api_call`,
    making it durable and visible in the workflow event history.

    The :func:`temporal_http_options` helper constructs this for you.

    Args:
        start_to_close_timeout: Maximum time for a single HTTP request activity.
        schedule_to_close_timeout: Maximum time from scheduling to completion.
        heartbeat_timeout: Maximum time between activity heartbeats.
        retry_policy: Retry policy for failed HTTP request activities.
    """

    def __init__(
        self,
        *,
        start_to_close_timeout: timedelta | None = timedelta(seconds=60),
        schedule_to_close_timeout: timedelta | None = None,
        heartbeat_timeout: timedelta | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        # Use a no-op transport to avoid SSL cert file I/O at construction time.
        # The transport is never invoked because send() is fully overridden.
        super().__init__(transport=_NoOpAsyncTransport())
        self._activity_kwargs: dict[str, Any] = {
            "start_to_close_timeout": start_to_close_timeout,
            "schedule_to_close_timeout": schedule_to_close_timeout,
            "heartbeat_timeout": heartbeat_timeout,
            "retry_policy": retry_policy,
        }

    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        """Dispatch the HTTP request as a Temporal activity instead of making a direct call."""
        try:
            content = request.content
        except httpx.RequestNotRead:
            content = b""

        # Extract the timeout the SDK set on this request.  httpx stores it
        # as request.extensions["timeout"] (a dict with pool/connect/read/write
        # keys).  We pass the read timeout through so the activity's httpx
        # client doesn't use the default 5 s.
        timeout: float | None = None
        timeout_ext = request.extensions.get("timeout", {})
        if isinstance(timeout_ext, dict):
            # Prefer read timeout (waiting for response), fall back to pool
            timeout = timeout_ext.get("read") or timeout_ext.get("pool")

        # Headers are passed through as-is.  Sensitive headers (x-goog-api-key,
        # authorization) are encrypted transparently by SensitiveFieldsCodec
        # before the payload reaches Temporal's event history, and decrypted
        # before the activity receives it.  No manual stripping needed.

        req_data = HttpRequestData(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            content=content,
            timeout=timeout,
        )

        resp_data = await temporal_workflow.execute_activity(
            gemini_api_call,
            req_data,
            **self._activity_kwargs,
        )

        return httpx.Response(
            status_code=resp_data.status_code,
            headers=resp_data.headers,
            content=resp_data.content,
            request=request,
        )


def temporal_http_options(
    *,
    start_to_close_timeout: timedelta = timedelta(seconds=60),
    schedule_to_close_timeout: timedelta | None = None,
    heartbeat_timeout: timedelta | None = None,
    retry_policy: RetryPolicy | None = None,
) -> types.HttpOptions:
    """Create ``HttpOptions`` that route all Gemini SDK HTTP calls through Temporal.

    .. warning::
        This API is experimental and may change in future versions.
        Use with caution in production environments.

    Pass the result to ``genai.Client(http_options=...)`` so that every model
    call made from within a Temporal workflow is durably recorded as a Temporal
    activity.

    Create the ``genai.Client`` at **module level** (outside the workflow
    sandbox) so that ``os.environ`` is read at import time, not inside the
    workflow.  The workflow then references the pre-built client.

    Gemini SDK retries are disabled (``attempts=1``) because Temporal owns the
    retry policy instead.

    Args:
        start_to_close_timeout: Maximum time for a single HTTP request activity.
            Defaults to 60 seconds.
        schedule_to_close_timeout: Maximum time from scheduling to completion.
        heartbeat_timeout: Maximum time between heartbeats.
        retry_policy: Retry policy for failed HTTP request activities.

    Returns:
        A :class:`~google.genai.types.HttpOptions` instance with a
        :class:`TemporalHttpxClient` set as the async HTTP backend.

    Example::

        # ---- module level (outside workflow sandbox) ----
        gemini_client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"],
            http_options=temporal_http_options(
                start_to_close_timeout=timedelta(seconds=60),
            ),
        )

        # ---- inside the workflow ----
        @workflow.defn
        class AgentWorkflow:
            @workflow.run
            async def run(self, query: str) -> str:
                response = await gemini_client.aio.models.generate_content(...)
                return response.text
    """
    return types.HttpOptions(
        httpx_async_client=TemporalHttpxClient(
            start_to_close_timeout=start_to_close_timeout,
            schedule_to_close_timeout=schedule_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            retry_policy=retry_policy,
        ),
        # Temporal owns retries; disable SDK-level retries to avoid interference.
        retry_options=types.HttpRetryOptions(attempts=1),
    )
