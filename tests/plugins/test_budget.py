"""Tests for the BudgetPlugin."""

from unittest.mock import MagicMock, patch

import pytest
from strands.hooks.events import AfterInvocationEvent, BeforeInvocationEvent, BeforeModelCallEvent
from strands.telemetry.metrics import EventLoopCycleMetric, EventLoopMetrics
from strands.types.event_loop import Usage

from strands_agents_extensions.plugins import BudgetExceededException, BudgetLimitType, BudgetPlugin, OnExceedAction


@pytest.fixture
def mock_agent():
    """Create a mock agent with event_loop_metrics."""
    agent = MagicMock()
    agent.event_loop_metrics = EventLoopMetrics()
    agent.event_loop_metrics.reset_usage_metrics()
    return agent


@pytest.fixture
def plugin():
    """Create a BudgetPlugin with default pricing."""
    return BudgetPlugin(
        max_cost_per_invocation=0.10,
        max_cost_per_session=1.00,
        on_exceed=OnExceedAction.STOP,
    )


def _add_cycle_usage(agent: MagicMock, usage: Usage) -> None:
    """Add a cycle with usage data to the agent's latest invocation metrics."""
    invocation = agent.event_loop_metrics.agent_invocations[-1]
    invocation.cycles.append(EventLoopCycleMetric(event_loop_cycle_id="test-cycle", usage=usage))


class TestBudgetPluginInit:
    """Tests for BudgetPlugin initialization."""

    def test_valid_on_exceed_stop(self):
        plugin = BudgetPlugin(max_cost_per_session=1.0, on_exceed=OnExceedAction.STOP)
        assert plugin._on_exceed is OnExceedAction.STOP

    def test_valid_on_exceed_warn(self):
        plugin = BudgetPlugin(max_cost_per_session=1.0, on_exceed=OnExceedAction.WARN)
        assert plugin._on_exceed is OnExceedAction.WARN

    def test_invalid_on_exceed_raises(self):
        with pytest.raises(ValueError):
            BudgetPlugin(max_cost_per_session=1.0, on_exceed="invalid")

    def test_name(self):
        plugin = BudgetPlugin(max_cost_per_session=1.0)
        assert plugin.name == "budget"

    def test_custom_pricing(self):
        pricing = {"input_per_1m": 1.0, "output_per_1m": 2.0}
        plugin = BudgetPlugin(max_cost_per_session=1.0, pricing=pricing)
        assert plugin._pricing == pricing

    def test_default_pricing(self):
        plugin = BudgetPlugin(max_cost_per_session=1.0)
        assert plugin._pricing["input_per_1m"] == 3.00
        assert plugin._pricing["output_per_1m"] == 15.00


class TestBudgetPluginHookRegistration:
    """Tests for hook registration via init_agent."""

    def test_init_agent_registers_hooks(self):
        plugin = BudgetPlugin(max_cost_per_session=1.0)
        agent = MagicMock()
        plugin.init_agent(agent)
        assert agent.add_hook.call_count == 3


class TestBudgetPluginCostEstimation:
    """Tests for cost estimation."""

    def test_estimate_cost_default_pricing(self):
        plugin = BudgetPlugin(max_cost_per_session=1.0)
        usage = Usage(inputTokens=1_000_000, outputTokens=1_000_000, totalTokens=2_000_000)
        cost = plugin.estimate_cost(usage)
        assert cost == pytest.approx(18.00)

    def test_estimate_cost_custom_pricing(self):
        plugin = BudgetPlugin(
            max_cost_per_session=1.0,
            pricing={"input_per_1m": 1.0, "output_per_1m": 2.0},
        )
        usage = Usage(inputTokens=500_000, outputTokens=500_000, totalTokens=1_000_000)
        cost = plugin.estimate_cost(usage)
        assert cost == pytest.approx(1.50)

    def test_estimate_cost_zero_tokens(self):
        plugin = BudgetPlugin(max_cost_per_session=1.0)
        usage = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
        assert plugin.estimate_cost(usage) == 0.0


class TestBudgetPluginTracking:
    """Tests for cost tracking across model calls."""

    def test_invocation_cost_resets_on_new_invocation(self, plugin, mock_agent):
        plugin.init_agent(mock_agent)
        before_event = BeforeInvocationEvent(agent=mock_agent)
        plugin._on_before_invocation(before_event)
        assert plugin.invocation_cost == 0.0

    def test_session_cost_accumulates(self, mock_agent):
        plugin = BudgetPlugin(max_cost_per_session=100.0, on_exceed=OnExceedAction.WARN)
        plugin.init_agent(mock_agent)

        usage = Usage(inputTokens=1000, outputTokens=500, totalTokens=1500)
        _add_cycle_usage(mock_agent, usage)
        plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))

        first_cost = plugin.session_cost
        assert first_cost > 0

        mock_agent.event_loop_metrics.reset_usage_metrics()
        _add_cycle_usage(mock_agent, usage)
        plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))

        assert plugin.session_cost == pytest.approx(first_cost * 2)

    def test_session_usage_accumulates(self, mock_agent):
        plugin = BudgetPlugin(max_cost_per_session=100.0)
        plugin.init_agent(mock_agent)

        usage = Usage(inputTokens=100, outputTokens=50, totalTokens=150)
        _add_cycle_usage(mock_agent, usage)
        plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))

        assert plugin.session_usage["inputTokens"] == 100
        assert plugin.session_usage["outputTokens"] == 50

    def test_skips_tracking_when_no_cycles(self, plugin, mock_agent):
        plugin.init_agent(mock_agent)
        # No cycles added — should not track anything
        plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))
        assert plugin.session_cost == 0.0


class TestBudgetPluginLimits:
    """Tests for budget limit enforcement."""

    def test_invocation_limit_stop(self, mock_agent):
        plugin = BudgetPlugin(max_cost_per_invocation=0.0001, on_exceed=OnExceedAction.STOP)
        plugin.init_agent(mock_agent)

        usage = Usage(inputTokens=100_000, outputTokens=100_000, totalTokens=200_000)
        _add_cycle_usage(mock_agent, usage)

        with pytest.raises(BudgetExceededException, match="invocation"):
            plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))

    def test_session_limit_stop(self, mock_agent):
        plugin = BudgetPlugin(max_cost_per_session=0.0001, on_exceed=OnExceedAction.STOP)
        plugin.init_agent(mock_agent)

        usage = Usage(inputTokens=100_000, outputTokens=100_000, totalTokens=200_000)
        _add_cycle_usage(mock_agent, usage)

        with pytest.raises(BudgetExceededException, match="session"):
            plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))

    def test_invocation_limit_warn(self, mock_agent):
        plugin = BudgetPlugin(max_cost_per_invocation=0.0001, on_exceed=OnExceedAction.WARN)
        plugin.init_agent(mock_agent)

        usage = Usage(inputTokens=100_000, outputTokens=100_000, totalTokens=200_000)
        _add_cycle_usage(mock_agent, usage)

        plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))
        assert plugin.invocation_cost > 0.0001

    def test_no_limit_configured(self, mock_agent):
        plugin = BudgetPlugin()
        plugin.init_agent(mock_agent)

        usage = Usage(inputTokens=10_000_000, outputTokens=10_000_000, totalTokens=20_000_000)
        _add_cycle_usage(mock_agent, usage)

        plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))


class TestBudgetPluginMidInvocationCheck:
    """Tests for budget enforcement before each model call."""

    def test_before_model_call_stops_when_over_budget(self, mock_agent):
        """BeforeModelCallEvent should raise when completed cycles exceed budget."""
        plugin = BudgetPlugin(max_cost_per_invocation=0.0001, on_exceed=OnExceedAction.STOP)
        plugin.init_agent(mock_agent)

        # Simulate a completed cycle with high usage
        usage = Usage(inputTokens=100_000, outputTokens=100_000, totalTokens=200_000)
        _add_cycle_usage(mock_agent, usage)

        # The next model call should be blocked
        with pytest.raises(BudgetExceededException, match="invocation"):
            plugin._on_before_model_call(BeforeModelCallEvent(agent=mock_agent))

    def test_before_model_call_checks_session_limit(self, mock_agent):
        """BeforeModelCallEvent should check session limit including prior invocations."""
        # Use a session limit that fits one invocation but not two
        plugin = BudgetPlugin(max_cost_per_session=2.00, on_exceed=OnExceedAction.STOP)
        plugin.init_agent(mock_agent)

        # First invocation: costs ~1.80 (under 2.00 limit)
        usage = Usage(inputTokens=100_000, outputTokens=100_000, totalTokens=200_000)
        _add_cycle_usage(mock_agent, usage)
        plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))

        # Start a second invocation with another cycle
        mock_agent.event_loop_metrics.reset_usage_metrics()
        _add_cycle_usage(mock_agent, usage)

        # The before-model-call check should see prior session cost + current cycle > limit
        with pytest.raises(BudgetExceededException, match="session"):
            plugin._on_before_model_call(BeforeModelCallEvent(agent=mock_agent))

    def test_before_model_call_allows_when_under_budget(self, mock_agent):
        """BeforeModelCallEvent should not raise when within budget."""
        plugin = BudgetPlugin(max_cost_per_invocation=100.0, on_exceed=OnExceedAction.STOP)
        plugin.init_agent(mock_agent)

        usage = Usage(inputTokens=100, outputTokens=50, totalTokens=150)
        _add_cycle_usage(mock_agent, usage)

        # Should not raise
        plugin._on_before_model_call(BeforeModelCallEvent(agent=mock_agent))

    def test_before_model_call_skips_when_no_cycles(self, mock_agent):
        """BeforeModelCallEvent should be a no-op on the first model call (no prior cycles)."""
        plugin = BudgetPlugin(max_cost_per_invocation=0.0001, on_exceed=OnExceedAction.STOP)
        plugin.init_agent(mock_agent)

        # No cycles yet — should not raise
        plugin._on_before_model_call(BeforeModelCallEvent(agent=mock_agent))


class TestBudgetPluginLiteLLMPricing:
    """Tests for litellm-based cost estimation."""

    def test_uses_litellm_when_no_custom_pricing(self, mock_agent):
        """When no pricing dict is provided, litellm.completion_cost should be used."""
        plugin = BudgetPlugin(max_cost_per_session=100.0)
        plugin.init_agent(mock_agent)

        mock_agent.model.config = {"model_id": "anthropic.claude-sonnet-4-20250514-v1:0"}

        usage = Usage(inputTokens=1000, outputTokens=500, totalTokens=1500)
        _add_cycle_usage(mock_agent, usage)

        with patch("strands_agents_extensions.plugins.budget.completion_cost", return_value=0.042) as mock_cost:
            plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))
            assert mock_cost.call_count >= 1

        assert plugin.session_cost == pytest.approx(0.042)

    def test_skips_litellm_when_custom_pricing_provided(self, mock_agent):
        """When pricing dict is provided, manual calculation should be used instead of litellm."""
        plugin = BudgetPlugin(
            max_cost_per_session=100.0,
            pricing={"input_per_1m": 1.0, "output_per_1m": 2.0},
        )
        plugin.init_agent(mock_agent)

        usage = Usage(inputTokens=1_000_000, outputTokens=1_000_000, totalTokens=2_000_000)
        _add_cycle_usage(mock_agent, usage)

        with patch("strands_agents_extensions.plugins.budget.completion_cost") as mock_cost:
            plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))
            mock_cost.assert_not_called()

        assert plugin.session_cost == pytest.approx(3.0)

    def test_litellm_receives_correct_model_and_tokens(self, mock_agent):
        """Verify litellm is called with the correct model ID and token counts."""
        plugin = BudgetPlugin(max_cost_per_session=100.0)
        plugin.init_agent(mock_agent)

        mock_agent.model.config = {"model_id": "amazon.nova-pro-v1:0"}

        usage = Usage(inputTokens=500, outputTokens=200, totalTokens=700)
        _add_cycle_usage(mock_agent, usage)

        with patch("strands_agents_extensions.plugins.budget.completion_cost", return_value=0.01) as mock_cost:
            plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))
            mock_cost.assert_called_with(
                model="amazon.nova-pro-v1:0",
                prompt_tokens=500,
                completion_tokens=200,
            )

    def test_falls_back_to_manual_pricing_on_litellm_error(self, mock_agent):
        """If litellm raises an exception, fall back to default manual pricing."""
        plugin = BudgetPlugin(max_cost_per_session=100.0)
        plugin.init_agent(mock_agent)

        mock_agent.model.config = {"model_id": "unknown-model"}

        usage = Usage(inputTokens=1_000_000, outputTokens=1_000_000, totalTokens=2_000_000)
        _add_cycle_usage(mock_agent, usage)

        with patch(
            "strands_agents_extensions.plugins.budget.completion_cost", side_effect=Exception("model not found")
        ):
            plugin._on_after_invocation(AfterInvocationEvent(agent=mock_agent))

        # Falls back to default pricing: (1M * 3.00/1M) + (1M * 15.00/1M) = 18.00
        assert plugin.session_cost == pytest.approx(18.00)

    def test_estimate_cost_still_works_with_manual_pricing(self):
        """The public estimate_cost method should still work with manual pricing."""
        plugin = BudgetPlugin(
            max_cost_per_session=1.0,
            pricing={"input_per_1m": 1.0, "output_per_1m": 2.0},
        )
        usage = Usage(inputTokens=500_000, outputTokens=500_000, totalTokens=1_000_000)
        cost = plugin.estimate_cost(usage)
        assert cost == pytest.approx(1.50)


class TestBudgetExceededException:
    """Tests for BudgetExceededException."""

    def test_attributes(self):
        exc = BudgetExceededException(BudgetLimitType.SESSION, 5.50, 5.00)
        assert exc.limit_type is BudgetLimitType.SESSION
        assert exc.current_cost == 5.50
        assert exc.limit == 5.00

    def test_message(self):
        exc = BudgetExceededException(BudgetLimitType.INVOCATION, 1.23, 1.00)
        assert "invocation" in str(exc)
        assert "1.2300" in str(exc)
