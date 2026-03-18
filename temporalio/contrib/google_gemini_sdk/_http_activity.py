"""Temporal activity for executing HTTP requests on behalf of the Gemini SDK.

The Gemini SDK makes HTTP calls internally through an httpx.AsyncClient.
:class:`~temporalio.contrib.google_gemini_sdk.workflow.TemporalHttpxClient`
overrides that client's ``send()`` method to dispatch through this activity
instead of making direct network calls.  This ensures every Gemini model call
is durably recorded in Temporal's workflow event history and benefits from
Temporal's retry/timeout semantics.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from temporalio import activity


class HttpRequestData(BaseModel):
    """Serialized form of an httpx.Request for transport across the activity boundary."""

    method: str
    url: str
    headers: dict[str, str]
    content: bytes
    timeout: float | None = None


class HttpResponseData(BaseModel):
    """Serialized form of an httpx.Response for transport across the activity boundary."""

    status_code: int
    headers: dict[str, str]
    content: bytes


@activity.defn
async def gemini_api_call(req: HttpRequestData) -> HttpResponseData:
    """Execute an HTTP request and return the serialized response.

    .. warning::
        This activity is experimental and may change in future versions.
        Use with caution in production environments.

    This activity is registered automatically by
    :class:`~temporalio.contrib.google_gemini_sdk.GeminiPlugin` and is invoked
    by :class:`~temporalio.contrib.google_gemini_sdk.workflow.TemporalHttpxClient`
    whenever the Gemini SDK needs to make a network call from within a workflow.

    Do not call this activity directly.
    """
    import os

    # ── Credential re-injection ──────────────────────────────────────────
    # TemporalHttpxClient.send() (in _temporal_httpx_client.py) strips the
    # "x-goog-api-key" header from the serialized request so that the API
    # key never appears in Temporal's event history.  Here — inside the
    # activity, which runs on the worker outside the workflow sandbox — we
    # read the API key from the worker's environment and put it back before
    # making the real HTTP call.
    #
    # NOTE: The "authorization" header (used by Vertex AI OAuth flows) is
    # NOT stripped because those tokens are short-lived and refreshed
    # per-request by the SDK — we cannot reconstruct them here.
    #
    # This means the API key must be set in the worker's environment
    # (GOOGLE_API_KEY or GEMINI_API_KEY) for API-key auth to work.
    # ─────────────────────────────────────────────────────────────────────
    headers = dict(req.headers)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key and "x-goog-api-key" not in headers:
        headers["x-goog-api-key"] = api_key

    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=req.method,
            url=req.url,
            headers=headers,
            content=req.content,
            timeout=req.timeout,
        )
        # response.content is already decoded (decompressed) by httpx.
        # Strip content-encoding so the reconstructed Response on the
        # workflow side doesn't try to decompress it a second time.
        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower() not in ("content-encoding", "content-length")
        }
        return HttpResponseData(
            status_code=response.status_code,
            headers=headers,
            content=response.content,
        )
