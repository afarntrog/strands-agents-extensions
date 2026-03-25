"""Budget plugin for tracking and enforcing token/cost limits.

This module provides a plugin that monitors token usage across agent invocations
and enforces configurable budget limits. It supports per-invocation and cumulative
(session-level) cost tracking with configurable actions when limits are exceeded.

Example Usage:
    ```python
    from strands import Agent
    from strands_agents_extensions.plugins import BudgetPlugin, OnExceedAction

    plugin = BudgetPlugin(
        max_cost_per_invocation=0.50,
        max_cost_per_session=5.00,
        on_exceed=OnExceedAction.STOP,
    )
    agent = Agent(plugins=[plugin])
    ```
"""

import logging
from enum import Enum
from typing import TYPE_CHECKING

from litellm import completion_cost
from strands.hooks.events import AfterInvocationEvent, BeforeInvocationEvent, BeforeModelCallEvent
from strands.plugins import Plugin
from strands.types.event_loop import Usage

if TYPE_CHECKING:
    from strands.agent import Agent

logger = logging.getLogger(__name__)

# Default pricing per 1M tokens (USD) for common models.
# Users can override with custom pricing via the `pricing` parameter.
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "default": {
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
    },
}


class OnExceedAction(str, Enum):
    """Action to take when a budget limit is exceeded."""

    STOP = "stop"
    WARN = "warn"


class BudgetLimitType(str, Enum):
    """Type of budget limit."""

    INVOCATION = "invocation"
    SESSION = "session"


class BudgetExceededException(Exception):
    """Exception raised when a budget limit is exceeded.

    Attributes:
        limit_type: The type of limit that was exceeded ("invocation" or "session").
        current_cost: The current accumulated cost when the limit was hit.
        limit: The configured limit that was exceeded.
    """

    def __init__(self, limit_type: BudgetLimitType, current_cost: float, limit: float) -> None:
        """Initialize the exception.

        Args:
            limit_type: The type of limit exceeded.
            current_cost: The current accumulated cost.
            limit: The configured limit that was exceeded.
        """
        self.limit_type = limit_type
        self.current_cost = current_cost
        self.limit = limit
        super().__init__(f"Budget exceeded: {limit_type.value} cost ${current_cost:.4f} exceeds limit ${limit:.4f}")


class BudgetPlugin(Plugin):
    """Plugin that tracks token usage costs and enforces budget limits.

    Monitors token consumption across model calls and estimates costs using
    configurable pricing. Supports per-invocation and session-level limits
    with configurable actions when limits are exceeded.

    Attributes:
        name: Plugin identifier ("budget").

    Example:
        ```python
        from strands import Agent
        from strands_agents_extensions.plugins import BudgetPlugin, OnExceedAction

        # Stop agent when cost exceeds limits
        plugin = BudgetPlugin(
            max_cost_per_invocation=0.50,
            max_cost_per_session=5.00,
            on_exceed=OnExceedAction.STOP,
        )
        agent = Agent(plugins=[plugin])

        # Warn but continue
        plugin = BudgetPlugin(
            max_cost_per_invocation=1.00,
            on_exceed=OnExceedAction.WARN,
        )

        # Custom pricing for a specific model
        plugin = BudgetPlugin(
            max_cost_per_session=10.00,
            pricing={"input_per_1m": 1.00, "output_per_1m": 5.00},
        )
        ```
    """

    name: str = "budget"

    def __init__(
        self,
        max_cost_per_invocation: float | None = None,
        max_cost_per_session: float | None = None,
        on_exceed: OnExceedAction = OnExceedAction.STOP,
        pricing: dict[str, float] | None = None,
    ) -> None:
        """Initialize the budget plugin.

        Args:
            max_cost_per_invocation: Maximum allowed cost per agent invocation in USD.
                None means no per-invocation limit.
            max_cost_per_session: Maximum allowed cumulative cost across all invocations in USD.
                None means no session limit.
            on_exceed: Action to take when a limit is exceeded.
                OnExceedAction.STOP raises BudgetExceededException.
                OnExceedAction.WARN logs a warning and continues.
            pricing: Token pricing as {"input_per_1m": float, "output_per_1m": float}.
                Prices are per 1 million tokens in USD. If not provided, uses default pricing.

        Raises:
            ValueError: If on_exceed is not a valid OnExceedAction.
        """
        on_exceed = OnExceedAction(on_exceed)

        self._max_cost_per_invocation = max_cost_per_invocation
        self._max_cost_per_session = max_cost_per_session
        self._on_exceed = on_exceed
        self._pricing = pricing or DEFAULT_PRICING["default"]
        self._use_litellm = pricing is None

        self._session_cost: float = 0.0
        self._invocation_cost: float = 0.0
        self._session_usage: Usage = Usage(inputTokens=0, outputTokens=0, totalTokens=0)

        super().__init__()

    def init_agent(self, agent: "Agent") -> None:
        """Register hooks for budget tracking on the agent.

        Args:
            agent: The agent instance to attach budget tracking to.
        """
        agent.add_hook(self._on_before_invocation)
        agent.add_hook(self._on_before_model_call)
        agent.add_hook(self._on_after_invocation)

    @property
    def session_cost(self) -> float:
        """Current cumulative session cost in USD."""
        return self._session_cost

    @property
    def invocation_cost(self) -> float:
        """Current invocation cost in USD."""
        return self._invocation_cost

    @property
    def session_usage(self) -> Usage:
        """Cumulative token usage across all invocations."""
        return self._session_usage

    def estimate_cost(self, usage: Usage) -> float:
        """Estimate cost in USD for the given token usage.

        Args:
            usage: Token usage to estimate cost for.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (usage["inputTokens"] / 1_000_000) * self._pricing["input_per_1m"]
        output_cost = (usage["outputTokens"] / 1_000_000) * self._pricing["output_per_1m"]
        return input_cost + output_cost

    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Reset per-invocation cost tracking at the start of each invocation.

        Args:
            event: The before invocation event.
        """
        self._invocation_cost = 0.0

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Check budget limits before each model call using completed cycle data.

        Catches budget overruns mid-invocation so agents with multiple model
        calls (tool-use loops) don't blow past the budget unchecked.

        Args:
            event: The before model call event.

        Raises:
            BudgetExceededException: If on_exceed is "stop" and a limit is exceeded.
        """
        agent = event.agent
        latest_invocation = agent.event_loop_metrics.latest_agent_invocation
        if latest_invocation is None or not latest_invocation.cycles:
            return

        # Estimate cost from all completed cycles so far in this invocation
        running_invocation_cost = sum(
            self._estimate_cycle_cost(agent, cycle.usage) for cycle in latest_invocation.cycles
        )
        running_session_cost = self._session_cost + running_invocation_cost

        if self._max_cost_per_invocation is not None and running_invocation_cost > self._max_cost_per_invocation:
            self._handle_exceeded(BudgetLimitType.INVOCATION, running_invocation_cost, self._max_cost_per_invocation)

        if self._max_cost_per_session is not None and running_session_cost > self._max_cost_per_session:
            self._handle_exceeded(BudgetLimitType.SESSION, running_session_cost, self._max_cost_per_session)

    def _estimate_cycle_cost(self, agent: "Agent", usage: Usage) -> float:
        """Estimate cost for a single cycle's token usage.

        Uses litellm's pricing database when no custom pricing is provided.
        Falls back to manual pricing if litellm fails.

        Args:
            agent: The agent instance (used to read the model ID for litellm).
            usage: Token usage for the cycle.

        Returns:
            Estimated cost in USD.
        """
        if not self._use_litellm:
            return self.estimate_cost(usage)

        try:
            model_id = agent.model.config.get("model_id", "")
            return completion_cost(
                model=model_id,
                prompt_tokens=usage["inputTokens"],
                completion_tokens=usage["outputTokens"],
            )
        except Exception:
            return self.estimate_cost(usage)

    def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Track costs after each invocation and enforce limits.

        Uses cycle usage data which is only populated after the model call completes.

        Args:
            event: The after invocation event.

        Raises:
            BudgetExceededException: If on_exceed is "stop" and a limit is exceeded.
        """
        agent = event.agent
        latest_invocation = agent.event_loop_metrics.latest_agent_invocation
        if latest_invocation is None or not latest_invocation.cycles:
            return

        # Calculate invocation cost from all cycles
        self._invocation_cost = sum(self._estimate_cycle_cost(agent, cycle.usage) for cycle in latest_invocation.cycles)

        # Accumulate session usage from all cycles in this invocation
        for cycle in latest_invocation.cycles:
            self._accumulate_usage(self._session_usage, cycle.usage)

        # Update session cost
        self._session_cost += self._invocation_cost

        logger.debug(
            "invocation_cost=<%.6f>, session_cost=<%.6f> | budget tracking update",
            self._invocation_cost,
            self._session_cost,
        )

        self._check_limits()

    def _check_limits(self) -> None:
        """Check if any budget limits have been exceeded.

        Raises:
            BudgetExceededException: If on_exceed is "stop" and a limit is exceeded.
        """
        if self._max_cost_per_invocation is not None and self._invocation_cost > self._max_cost_per_invocation:
            self._handle_exceeded(BudgetLimitType.INVOCATION, self._invocation_cost, self._max_cost_per_invocation)

        if self._max_cost_per_session is not None and self._session_cost > self._max_cost_per_session:
            self._handle_exceeded(BudgetLimitType.SESSION, self._session_cost, self._max_cost_per_session)

    def _handle_exceeded(self, limit_type: BudgetLimitType, current_cost: float, limit: float) -> None:
        """Handle a budget limit being exceeded.

        Args:
            limit_type: The type of limit exceeded.
            current_cost: The current accumulated cost.
            limit: The configured limit.

        Raises:
            BudgetExceededException: If on_exceed is OnExceedAction.STOP.
        """
        if self._on_exceed is OnExceedAction.STOP:
            logger.warning(
                "limit_type=<%s>, current_cost=<%.6f>, limit=<%.6f> | budget exceeded, stopping",
                limit_type,
                current_cost,
                limit,
            )
            raise BudgetExceededException(limit_type, current_cost, limit)

        logger.warning(
            "limit_type=<%s>, current_cost=<%.6f>, limit=<%.6f> | budget exceeded, continuing",
            limit_type,
            current_cost,
            limit,
        )

    @staticmethod
    def _accumulate_usage(target: Usage, source: Usage) -> None:
        """Accumulate token usage from source into target.

        Args:
            target: The usage dict to accumulate into.
            source: The usage dict to accumulate from.
        """
        target["inputTokens"] += source["inputTokens"]
        target["outputTokens"] += source["outputTokens"]
        target["totalTokens"] += source["totalTokens"]
