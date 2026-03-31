"""Retry plugin for automatic same-model retries on transient failures.

This module provides a plugin that retries the same model when it fails with
a matching exception. It supports configurable delay, exponential backoff,
max retry attempts, and exception type filtering.

Unlike ModelFallbackPlugin which swaps to a different model, RetryPlugin
retries the current model — useful for transient errors like throttling,
network timeouts, or temporary service unavailability.

Example Usage:
    ```python
    from strands import Agent
    from strands_agents_extensions.plugins import RetryPlugin

    plugin = RetryPlugin(
        max_retries=3,
        retry_on=[ConnectionError, TimeoutError],
        delay_seconds=1.0,
        backoff_factor=2.0,
    )
    agent = Agent(plugins=[plugin])
    ```
"""

import logging
import time
from typing import TYPE_CHECKING

from strands.hooks.events import AfterModelCallEvent
from strands.plugins import Plugin

if TYPE_CHECKING:
    from strands.agent import Agent

logger = logging.getLogger(__name__)


class RetryPlugin(Plugin):
    """Plugin that retries the same model on transient failures.

    When a model call fails with a matching exception, this plugin sets
    ``event.retry = True`` to trigger a retry.
    Supports configurable delay between retries and exponential backoff.

    Attributes:
        name: Plugin identifier ("retry").

    Example:
        ```python
        from strands import Agent
        from strands_agents_extensions.plugins import RetryPlugin

        # Retry up to 3 times with exponential backoff
        plugin = RetryPlugin(
            max_retries=3,
            retry_on=[ConnectionError, TimeoutError],
            delay_seconds=1.0,
            backoff_factor=2.0,
        )
        agent = Agent(plugins=[plugin])

        # Retry up to 5 times with no delay
        plugin = RetryPlugin(max_retries=5)
        ```
    """

    name: str = "retry"

    def __init__(
        self,
        max_retries: int,
        retry_on: list[type[Exception]] | None = None,
        delay_seconds: float = 0.0,
        backoff_factor: float = 1.0,
    ) -> None:
        """Initialize the retry plugin.

        Args:
            max_retries: Maximum number of retry attempts per model call sequence.
                Must be at least 1.
            retry_on: Exception types that trigger a retry. Subclasses of listed
                types also match. Defaults to [Exception] (all exceptions).
            delay_seconds: Base delay in seconds before each retry. 0 means no delay.
            backoff_factor: Multiplier applied to delay on each successive retry.
                With delay_seconds=1.0 and backoff_factor=2.0, delays are 1s, 2s, 4s, ...
                Use 1.0 for fixed delay.

        Raises:
            ValueError: If max_retries is less than 1.
        """
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self._max_retries = max_retries
        self._retry_on: list[type[Exception]] = retry_on if retry_on is not None else [Exception]
        self._delay_seconds = delay_seconds
        self._backoff_factor = backoff_factor

        self._attempt_count: int = 0

        super().__init__()

    def init_agent(self, agent: "Agent") -> None:
        """Register hooks for retry on the agent.

        Args:
            agent: The agent instance to attach retry behavior to.
        """
        agent.add_hook(self._on_after_model_call)

    @property
    def attempt_count(self) -> int:
        """Number of retry attempts made in the current sequence."""
        return self._attempt_count

    def _on_after_model_call(self, event: AfterModelCallEvent) -> None:
        """Handle model call completion and trigger retry if needed.

        On success, resets the attempt counter. On failure, checks exception
        type and attempt budget, then optionally sleeps and sets retry.

        Args:
            event: The after model call event.
        """
        # Success — reset counter
        if event.exception is None:
            self._attempt_count = 0
            return

        # Another hook already set retry — don't interfere
        if event.retry:
            return

        # Check if this exception type should trigger retry
        if not any(isinstance(event.exception, exc_type) for exc_type in self._retry_on):
            return

        # Check if we've exhausted retries
        if self._attempt_count >= self._max_retries:
            logger.debug(
                "attempt_count=<%d>, max_retries=<%d> | retry attempts exhausted",
                self._attempt_count,
                self._max_retries,
            )
            return

        # Apply delay with backoff
        if self._delay_seconds > 0:
            delay = self._delay_seconds * (self._backoff_factor ** self._attempt_count)
            logger.debug(
                "attempt=<%d>, delay=<%.3f> | sleeping before retry",
                self._attempt_count + 1,
                delay,
            )
            time.sleep(delay)

        # Trigger retry (same model — no swap)
        event.retry = True
        self._attempt_count += 1

        logger.info(
            "attempt=<%d>, max_retries=<%d> | retrying same model",
            self._attempt_count,
            self._max_retries,
        )
