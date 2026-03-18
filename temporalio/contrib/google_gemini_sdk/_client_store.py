"""Shared storage for the pre-built ``genai.Client`` instance.

This module is added to the Temporal sandbox passthrough list by
:class:`~temporalio.contrib.google_gemini_sdk.GeminiPlugin`.  Because it is
passthrough'd, the module-level ``_gemini_client`` variable is shared between
the real runtime (where the worker sets it) and the sandboxed workflow (where
:func:`get_gemini_client` reads it).

This module intentionally has **no** ``httpx`` or ``google.genai`` imports so
that it can also be loaded safely by the sandbox's restricted importer if the
passthrough hasn't been configured yet.
"""

from __future__ import annotations

from typing import Any

# Set by GeminiPlugin.__init__ before the worker starts.
_gemini_client: Any = None


def get_gemini_client() -> Any:
    """Return the ``genai.Client`` stored by :class:`GeminiPlugin`.

    .. warning::
        This function is experimental and may change in future versions.
        Use with caution in production environments.

    Call this inside a workflow to obtain the pre-built Gemini client that was
    passed to :class:`~temporalio.contrib.google_gemini_sdk.GeminiPlugin` at
    worker setup time.  The client is created **once** outside the workflow
    sandbox, so ``os.environ`` access, SSL cert loading, etc. happen at startup
    — not during workflow execution.

    Raises:
        RuntimeError: If no client has been configured (i.e. ``GeminiPlugin``
            was not initialised with a ``gemini_client``).
    """
    if _gemini_client is None:
        raise RuntimeError(
            "No Gemini client configured.  Pass gemini_client= to GeminiPlugin()."
        )
    return _gemini_client
