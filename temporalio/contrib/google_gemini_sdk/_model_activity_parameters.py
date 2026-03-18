"""Parameters for configuring Temporal activity execution for Gemini model calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio.common import Priority, RetryPolicy
from temporalio.workflow import ActivityCancellationType, VersioningIntent


@dataclass
class ModelActivityParameters:
    """Parameters for configuring Temporal activity execution for Gemini model calls.

    .. warning::
        This class is experimental and may change in future versions.
        Use with caution in production environments.

    This class encapsulates all the parameters that can be used to configure
    how Temporal activities are executed when making Gemini model calls.
    """

    task_queue: str | None = None
    """Specific task queue to use for model activities."""

    schedule_to_close_timeout: timedelta | None = None
    """Maximum time from scheduling to completion."""

    schedule_to_start_timeout: timedelta | None = None
    """Maximum time from scheduling to starting."""

    start_to_close_timeout: timedelta | None = timedelta(seconds=60)
    """Maximum time for the activity to complete."""

    heartbeat_timeout: timedelta | None = None
    """Maximum time between heartbeats."""

    retry_policy: RetryPolicy | None = None
    """Policy for retrying failed activities."""

    cancellation_type: ActivityCancellationType = ActivityCancellationType.TRY_CANCEL
    """How the activity handles cancellation."""

    versioning_intent: VersioningIntent | None = None
    """Versioning intent for the activity."""

    priority: Priority = field(default_factory=lambda: Priority.default)
    """Priority for the activity execution."""

    use_local_activity: bool = False
    """Whether to use a local activity. Changing mid-workflow breaks determinism."""
