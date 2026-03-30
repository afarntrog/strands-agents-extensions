"""Integration tests for multiple plugins composed on a single Agent with real Bedrock LLM."""

import pytest
from strands import Agent
from strands.models.bedrock import BedrockModel

from strands_agents_extensions.plugins import (
    BudgetExceededException,
    BudgetPlugin,
    MessageRedactionPlugin,
    ModelFallbackPlugin,
    OnExceedAction,
    OnLimitAction,
    RateLimiterPlugin,
)

from .conftest import MODEL_ID, retry_on_flaky


class TestBudgetAndRateLimiter:
    """Verify budget and rate limiter plugins work together with."""

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_both_plugins_active(self, model):
        """Agent with both plugins should track budget and enforce rate limits."""
        budget = BudgetPlugin(max_cost_per_session=100.0)
        rate_limiter = RateLimiterPlugin(model_calls_per_minute=10, on_limit=OnLimitAction.ERROR)
        agent = Agent(model=model, plugins=[budget, rate_limiter])

        agent("Say hello in one word.")

        assert budget.session_cost > 0
        assert budget.session_usage["inputTokens"] > 0


class TestBudgetAndRedaction:
    """Verify budget and redaction plugins work together with."""

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_redaction_and_budget_both_active(self, model):
        """PII should be redacted and costs should be tracked simultaneously."""
        budget = BudgetPlugin(max_cost_per_session=100.0)
        redaction = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
        )
        agent = Agent(model=model, plugins=[budget, redaction])

        agent("My SSN is 123-45-6789")

        # Budget should have tracked costs
        assert budget.session_cost > 0

        # Redaction should have happened
        assert redaction.redaction_count >= 1
        user_msg = agent.messages[0]
        assert "123-45-6789" not in user_msg["content"][0]["text"]


class TestBudgetAndFallback:
    """Verify budget and fallback plugins work together with."""

    @retry_on_flaky("LLM responses may throttle")
    def test_fallback_with_budget_tracking(self):
        """Budget should track costs even when fallback model is used."""
        primary = BedrockModel(model_id="us.anthropic.claude-nonexistent-v1:0")
        fallback = BedrockModel(model_id=MODEL_ID)

        budget = BudgetPlugin(max_cost_per_session=100.0)
        fallback_plugin = ModelFallbackPlugin(fallback_models=[fallback])
        agent = Agent(model=primary, plugins=[budget, fallback_plugin])

        agent("Say hello in one word.")

        # Budget should have tracked the fallback model's costs
        assert budget.session_cost > 0


class TestAllPlugins:
    """Verify all plugins can work together on a single agent with."""

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_all_plugins_composed(self, model):
        """Agent with all plugins should function correctly."""
        fallback_model = BedrockModel(model_id=MODEL_ID)

        budget = BudgetPlugin(max_cost_per_session=100.0)
        rate_limiter = RateLimiterPlugin(model_calls_per_minute=100, on_limit=OnLimitAction.ERROR)
        redaction = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
        )
        fallback_plugin = ModelFallbackPlugin(fallback_models=[fallback_model])

        agent = Agent(model=model, plugins=[budget, rate_limiter, redaction, fallback_plugin])

        agent("My SSN is 123-45-6789")

        assert budget.session_cost > 0
        assert redaction.redaction_count >= 1
