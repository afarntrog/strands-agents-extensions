"""Rate limiter plugin for throttling model and tool calls.

This module provides a plugin that enforces rate limits on model and tool
invocations using a sliding window algorithm. It supports configurable
limits and actions when limits are hit.

Example Usage:
    ```python
    from strands import Agent
    from strands_agents_extensions.plugins import OnLimitAction, RateLimiterPlugin

    plugin = RateLimiterPlugin(
        model_calls_per_minute=60,
        tool_calls_per_minute=100,
        on_limit=OnLimitAction.WAIT,
    )
    agent = Agent(plugins=[plugin])
    ```
"""

import logging
import threading
import time
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING

from strands.hooks.events import BeforeModelCallEvent, BeforeToolCallEvent
from strands.plugins import Plugin

if TYPE_CHECKING:
    from strands.agent import Agent

logger = logging.getLogger(__name__)


class OnLimitAction(str, Enum):
    """Action to take when a rate limit is hit."""

    WAIT = "wait"
    ERROR = "error"


class RateLimitType(str, Enum):
    """Type of rate limit."""

    MODEL = "model"
    TOOL = "tool"


class RateLimitExceededException(Exception):
    """Exception raised when a rate limit is exceeded and on_limit is OnLimitAction.ERROR.

    Attributes:
        limit_type: The type of limit exceeded.
        calls_per_minute: The configured rate limit.
    """

    def __init__(self, limit_type: RateLimitType, calls_per_minute: int) -> None:
        """Initialize the exception.

        Args:
            limit_type: The type of limit exceeded.
            calls_per_minute: The configured rate limit.
        """
        self.limit_type = limit_type
        self.calls_per_minute = calls_per_minute
        super().__init__(f"Rate limit exceeded: {limit_type.value} calls exceed {calls_per_minute}/minute")


class _SlidingWindowCounter:
    """Thread-safe sliding window rate counter.

    Tracks timestamps of events within a rolling time window and provides
    methods to check if the limit has been reached and wait until a slot
    is available.
    """

    def __init__(self, max_calls: int, window_seconds: float = 60.0) -> None:
        """Initialize the sliding window counter.

        Args:
            max_calls: Maximum number of calls allowed within the window.
            window_seconds: Size of the sliding window in seconds.
        """
        self._max_calls = max_calls
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def _evict_expired(self, now: float) -> None:
        """Remove timestamps that have fallen outside the window.

        Args:
            now: Current timestamp.
        """
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def try_acquire(self) -> bool:
        """Attempt to record a call within the rate limit.

        Returns:
            True if the call is within the limit and was recorded, False otherwise.
        """
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            if len(self._timestamps) < self._max_calls:
                self._timestamps.append(now)
                return True
            return False

    def wait_and_acquire(self) -> float:
        """Block until a slot is available, then record the call.

        Returns:
            The number of seconds spent waiting (0.0 if no wait was needed).
        """
        waited = 0.0
        while True:
            now = time.monotonic()
            with self._lock:
                self._evict_expired(now)
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return waited
                # Calculate how long until the oldest entry expires
                sleep_time = self._timestamps[0] + self._window_seconds - now

            if sleep_time > 0:
                logger.debug(
                    "sleep_time=<%.3f> | rate limiter waiting for slot",
                    sleep_time,
                )
                time.sleep(sleep_time)
                waited += sleep_time


class RateLimiterPlugin(Plugin):
    """Plugin that enforces rate limits on model and tool calls.

    Uses a sliding window algorithm to track call frequency and enforce
    configurable per-minute limits on model inference and tool execution.

    Attributes:
        name: Plugin identifier ("rate-limiter").

    Example:
        ```python
        from strands import Agent
        from strands_agents_extensions.plugins import OnLimitAction, RateLimiterPlugin

        # Wait when limit is hit
        plugin = RateLimiterPlugin(
            model_calls_per_minute=60,
            tool_calls_per_minute=100,
            on_limit=OnLimitAction.WAIT,
        )
        agent = Agent(plugins=[plugin])

        # Error when limit is hit
        plugin = RateLimiterPlugin(
            model_calls_per_minute=30,
            on_limit=OnLimitAction.ERROR,
        )
        ```
    """

    name: str = "rate-limiter"

    def __init__(
        self,
        model_calls_per_minute: int | None = None,
        tool_calls_per_minute: int | None = None,
        on_limit: OnLimitAction = OnLimitAction.WAIT,
    ) -> None:
        """Initialize the rate limiter plugin.

        Args:
            model_calls_per_minute: Maximum model calls per minute.
                None means no model call rate limit.
            tool_calls_per_minute: Maximum tool calls per minute.
                None means no tool call rate limit.
            on_limit: Action to take when a limit is hit.
                OnLimitAction.WAIT sleeps until a slot is available.
                OnLimitAction.ERROR raises RateLimitExceededException.

        Raises:
            ValueError: If on_limit is not a valid OnLimitAction, or if no limits are configured.
        """
        on_limit = OnLimitAction(on_limit)

        if model_calls_per_minute is None and tool_calls_per_minute is None:
            raise ValueError("at least one of model_calls_per_minute or tool_calls_per_minute must be set")

        self._on_limit = on_limit
        self._model_limiter: _SlidingWindowCounter | None = None
        self._tool_limiter: _SlidingWindowCounter | None = None

        if model_calls_per_minute is not None:
            self._model_limiter = _SlidingWindowCounter(model_calls_per_minute)

        if tool_calls_per_minute is not None:
            self._tool_limiter = _SlidingWindowCounter(tool_calls_per_minute)

        super().__init__()

    def init_agent(self, agent: "Agent") -> None:
        """Register hooks for rate limiting on the agent.

        Args:
            agent: The agent instance to attach rate limiting to.
        """
        if self._model_limiter is not None:
            agent.add_hook(self._on_before_model_call)

        if self._tool_limiter is not None:
            agent.add_hook(self._on_before_tool_call)

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Enforce model call rate limit.

        Args:
            event: The before model call event.

        Raises:
            RateLimitExceededException: If on_limit is "error" and limit is exceeded.
        """
        if self._model_limiter is None:
            return

        self._enforce_limit(self._model_limiter, RateLimitType.MODEL)

    def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Enforce tool call rate limit.

        Args:
            event: The before tool call event.

        Raises:
            RateLimitExceededException: If on_limit is "error" and limit is exceeded.
        """
        if self._tool_limiter is None:
            return

        tool_name = event.tool_use.get("name", "unknown")
        logger.debug("tool_name=<%s> | checking tool rate limit", tool_name)
        self._enforce_limit(self._tool_limiter, RateLimitType.TOOL)

    def _enforce_limit(self, limiter: _SlidingWindowCounter, limit_type: RateLimitType) -> None:
        """Enforce a rate limit using the given counter.

        Args:
            limiter: The sliding window counter to check.
            limit_type: The type of limit for logging/errors.

        Raises:
            RateLimitExceededException: If on_limit is OnLimitAction.ERROR and limit is exceeded.
        """
        if self._on_limit is OnLimitAction.WAIT:
            waited = limiter.wait_and_acquire()
            if waited > 0:
                logger.debug(
                    "limit_type=<%s>, waited=<%.3f> | rate limit wait completed",
                    limit_type,
                    waited,
                )
        else:
            if not limiter.try_acquire():
                logger.warning(
                    "limit_type=<%s> | rate limit exceeded",
                    limit_type,
                )
                raise RateLimitExceededException(limit_type, limiter._max_calls)
