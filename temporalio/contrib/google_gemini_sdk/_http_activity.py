"""Temporal activity for executing HTTP requests on behalf of the Gemini SDK.

The Gemini SDK makes HTTP calls internally through an httpx.AsyncClient.
:class:`~temporalio.contrib.google_gemini_sdk._temporal_httpx_client.TemporalHttpxClient`
overrides that client's ``send()`` method to dispatch through this activity
instead of making direct network calls.  This ensures every Gemini model call
is durably recorded in Temporal's workflow event history and benefits from
Temporal's retry/timeout semantics.

Credential headers (``x-goog-api-key``, ``authorization``) are **not** stripped
or re-injected here.  They are encrypted transparently by
:class:`~temporalio.contrib.google_gemini_sdk._sensitive_fields_codec.SensitiveFieldsCodec`
before the payload reaches Temporal's event history, and decrypted before the
activity receives it.
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
    by :class:`~temporalio.contrib.google_gemini_sdk._temporal_httpx_client.TemporalHttpxClient`
    whenever the Gemini SDK needs to make a network call from within a workflow.

    Credential headers arrive fully intact (decrypted by the codec before the
    activity receives the payload).  No manual re-injection is needed.

    Do not call this activity directly.
    """
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=req.method,
            url=req.url,
            headers=req.headers,
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
