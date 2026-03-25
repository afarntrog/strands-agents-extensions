"""Model fallback plugin for automatic failover across model providers.

This module provides a plugin that automatically falls back to alternative models
when the primary model fails. It supports ordered fallback chains, exception type
filtering, and provider cooldown to avoid repeatedly hitting failed providers.

Example Usage:
    ```python
    from strands import Agent
    from strands.models.bedrock import BedrockModel
    from strands.models.anthropic import AnthropicModel
    from strands_agents_extensions.plugins import ModelFallbackPlugin

    plugin = ModelFallbackPlugin(
        fallback_models=[
            BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
            AnthropicModel(model_id="claude-sonnet-4-20250514"),
        ],
    )
    agent = Agent(model=primary_model, plugins=[plugin])
    ```
"""

import logging
import time
from typing import TYPE_CHECKING, Any

from strands.hooks.events import AfterInvocationEvent, AfterModelCallEvent
from strands.plugins import Plugin

if TYPE_CHECKING:
    from strands.agent import Agent

logger = logging.getLogger(__name__)


class ModelFallbackPlugin(Plugin):
    """Plugin that automatically falls back to alternative models on failure.

    When the primary model fails with a matching exception, this plugin swaps
    the agent's model to the next fallback in the chain and triggers a retry.
    After the invocation completes, the original model is restored.

    Attributes:
        name: Plugin identifier ("model-fallback").

    Example:
        ```python
        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        plugin = ModelFallbackPlugin(
            fallback_models=[
                BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"),
            ],
            retry_on=[Exception],
            cooldown_seconds=60.0,
        )
        agent = Agent(model=primary_model, plugins=[plugin])
        ```
    """

    name: str = "model-fallback"

    def __init__(
        self,
        fallback_models: list[Any],
        retry_on: list[type[Exception]] | None = None,
        max_fallback_attempts: int | None = None,
        cooldown_seconds: float = 0.0,
    ) -> None:
        """Initialize the model fallback plugin.

        Args:
            fallback_models: Ordered list of fallback models. When the current model
                fails, the plugin tries the next model in this list.
            retry_on: Exception types that trigger a fallback. Subclasses of listed
                types also match. Defaults to [Exception] (all exceptions).
            max_fallback_attempts: Maximum number of fallback swaps per invocation.
                Defaults to the number of fallback models.
            cooldown_seconds: How long (in seconds) to skip a model after it fails.
                0 means no cooldown.

        Raises:
            ValueError: If fallback_models is empty.
        """
        if not fallback_models:
            raise ValueError("at least one fallback model must be provided")

        self._fallback_models = fallback_models
        self._retry_on: list[type[Exception]] = retry_on if retry_on is not None else [Exception]
        self._max_fallback_attempts = (
            max_fallback_attempts if max_fallback_attempts is not None else len(fallback_models)
        )
        self._cooldown_seconds = cooldown_seconds

        # Per-invocation state
        self._original_model: Any | None = None
        self._fallback_index: int = 0
        self._attempt_count: int = 0

        # Cooldown tracking: maps fallback index to the time when cooldown expires
        self._cooldown_until: dict[int, float] = {}

        super().__init__()

    def init_agent(self, agent: "Agent") -> None:
        """Register hooks for model fallback on the agent.

        Args:
            agent: The agent instance to attach fallback behavior to.
        """
        agent.add_hook(self._on_after_model_call)
        agent.add_hook(self._on_after_invocation)

    def _on_after_model_call(self, event: AfterModelCallEvent) -> None:
        """Handle model call completion and trigger fallback if needed.

        Args:
            event: The after model call event.
        """
        # No failure — nothing to do
        if event.exception is None:
            return

        # Another hook already set retry — don't interfere
        if event.retry:
            return

        # Check if this exception type should trigger fallback
        if not any(isinstance(event.exception, exc_type) for exc_type in self._retry_on):
            return

        # Check if we've exhausted our fallback attempts
        if self._attempt_count >= self._max_fallback_attempts:
            logger.debug(
                "attempt_count=<%d>, max_attempts=<%d> | fallback attempts exhausted",
                self._attempt_count,
                self._max_fallback_attempts,
            )
            return

        # Find the next available fallback (respecting cooldowns)
        fallback = self._select_next_fallback()
        if fallback is None:
            logger.debug("no available fallback models (all exhausted or cooled down)")
            return

        # Save the original model on first fallback
        if self._original_model is None:
            self._original_model = event.agent.model

        # Swap the model and trigger retry
        event.agent.model = fallback
        event.retry = True
        self._attempt_count += 1

        logger.info(
            "fallback_index=<%d>, attempt=<%d> | switched to fallback model",
            self._fallback_index - 1,
            self._attempt_count,
        )

    def _select_next_fallback(self) -> Any | None:
        """Find the next available fallback model, skipping cooled-down ones.

        Returns:
            The next fallback model, or None if all are exhausted or cooled down.
        """
        now = time.monotonic()

        while self._fallback_index < len(self._fallback_models):
            idx = self._fallback_index
            self._fallback_index += 1

            # Check cooldown
            cooldown_expiry = self._cooldown_until.get(idx, 0.0)
            if now < cooldown_expiry:
                logger.debug(
                    "fallback_index=<%d>, cooldown_remaining=<%.1f> | skipping cooled-down model",
                    idx,
                    cooldown_expiry - now,
                )
                continue

            # Mark the previous model (the one that just failed) for cooldown
            if self._cooldown_seconds > 0 and idx > 0:
                self._cooldown_until[idx - 1] = now + self._cooldown_seconds

            return self._fallback_models[idx]

        return None

    def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Restore the original model and reset state after invocation.

        Args:
            event: The after invocation event.
        """
        if self._original_model is not None:
            event.agent.model = self._original_model
            logger.debug("original model restored after invocation")

        self._original_model = None
        self._fallback_index = 0
        self._attempt_count = 0
