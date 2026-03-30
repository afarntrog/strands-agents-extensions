"""Integration tests for BudgetPlugin with a real Bedrock LLM."""

import pytest
from strands import Agent
from strands.models.bedrock import BedrockModel

from strands_agents_extensions.plugins import BudgetExceededException, BudgetPlugin, OnExceedAction

from .conftest import MODEL_ID, retry_on_flaky


class TestBudgetTracking:
    """Verify budget tracking."""

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_single_invocation_tracks_cost(self, model):
        """A single agent call should record non-zero invocation and session costs."""
        plugin = BudgetPlugin(max_cost_per_session=100.0)
        agent = Agent(model=model, plugins=[plugin])

        agent("Say hello in one word.")

        assert plugin.invocation_cost > 0
        assert plugin.session_cost > 0
        assert plugin.session_usage["inputTokens"] > 0
        assert plugin.session_usage["outputTokens"] > 0

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_session_cost_accumulates_across_invocations(self, model):
        """Session cost should grow with each agent invocation."""
        plugin = BudgetPlugin(max_cost_per_session=100.0)
        agent = Agent(model=model, plugins=[plugin])

        agent("Say hi.")
        first_cost = plugin.session_cost
        assert first_cost > 0

        agent("Say bye.")
        assert plugin.session_cost > first_cost

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_session_usage_tokens_accumulate(self, model):
        """Session usage should track total token counts across invocations."""
        plugin = BudgetPlugin(max_cost_per_session=100.0)
        agent = Agent(model=model, plugins=[plugin])

        agent("Say one word.")
        first_input = plugin.session_usage["inputTokens"]
        first_output = plugin.session_usage["outputTokens"]

        agent("Say another word.")
        assert plugin.session_usage["inputTokens"] > first_input
        assert plugin.session_usage["outputTokens"] > first_output


class TestBudgetLimitsStop:
    """Verify STOP action raises BudgetExceededException."""

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_session_limit_exceeded_raises(self, model):
        """Agent call should raise when session budget is exceeded."""
        # Set absurdly low budget so any real call exceeds it
        plugin = BudgetPlugin(
            max_cost_per_session=0.0000001,
            on_exceed=OnExceedAction.STOP,
        )
        agent = Agent(model=model, plugins=[plugin])

        with pytest.raises(BudgetExceededException, match="session"):
            agent("Hello!")

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_invocation_limit_exceeded_raises(self, model):
        """Agent call should raise when per-invocation budget is exceeded."""
        plugin = BudgetPlugin(
            max_cost_per_invocation=0.0000001,
            on_exceed=OnExceedAction.STOP,
        )
        agent = Agent(model=model, plugins=[plugin])

        with pytest.raises(BudgetExceededException, match="invocation"):
            agent("Hello!")


class TestBudgetLimitsWarn:
    """Verify WARN action continues execution."""

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_session_limit_exceeded_continues(self, model):
        """Agent should complete when WARN mode is used, even over budget."""
        plugin = BudgetPlugin(
            max_cost_per_session=0.0000001,
            on_exceed=OnExceedAction.WARN,
        )
        agent = Agent(model=model, plugins=[plugin])

        result = agent("Say hello in one word.")
        assert result is not None
        assert plugin.session_cost > 0.0000001
