"""Integration tests for RateLimiterPlugin with a real Bedrock LLM."""

import pytest
from strands import Agent

from strands_agents_extensions.plugins import OnLimitAction, RateLimiterPlugin, RateLimitExceededException

from .conftest import retry_on_flaky


class TestRateLimiterError:
    """Verify ERROR action raises when rate limit is exceeded with."""

    @retry_on_flaky("LLM responses may throttle")
    def test_model_calls_exceed_limit_raises(self, model):
        """Rapid agent calls should trigger rate limit error."""
        plugin = RateLimiterPlugin(model_calls_per_minute=2, on_limit=OnLimitAction.ERROR)
        agent = Agent(model=model, plugins=[plugin])

        agent("Say hi.")
        agent("Say bye.")

        with pytest.raises(RateLimitExceededException, match="model"):
            agent("Say hello.")

    @retry_on_flaky("LLM responses may throttle")
    def test_model_calls_within_limit_succeed(self, model):
        """Agent calls within the rate limit should all succeed."""
        plugin = RateLimiterPlugin(model_calls_per_minute=10, on_limit=OnLimitAction.ERROR)
        agent = Agent(model=model, plugins=[plugin])

        for _ in range(3):
            result = agent("Say one word.")
            assert result is not None
