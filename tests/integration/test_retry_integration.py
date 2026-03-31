"""Integration tests for RetryPlugin with a real Bedrock LLM."""

import pytest
from strands import Agent
from strands.models.bedrock import BedrockModel

from strands_agents_extensions.plugins import RetryPlugin

from .conftest import retry_on_flaky


class TestRetryExhaustion:
    """Verify retry exhaustion with a failing model."""

    def test_retries_exhaust_then_raises(self):
        """Agent should retry and then raise after max_retries with a bad model."""
        bad_model = BedrockModel(model_id="us.anthropic.claude-nonexistent-v1:0")
        plugin = RetryPlugin(max_retries=2)

        agent = Agent(model=bad_model, plugins=[plugin])

        with pytest.raises(Exception):
            agent("Say hello.")

        assert plugin.attempt_count == 2


class TestRetryNoFailure:
    """Verify no retries when the model succeeds."""

    @retry_on_flaky("LLM responses may throttle")
    def test_successful_call_no_retries(self, model):
        """A successful agent call should not trigger any retries."""
        plugin = RetryPlugin(max_retries=3)
        agent = Agent(model=model, plugins=[plugin])

        result = agent("Say hello in one word.")

        assert result is not None
        assert plugin.attempt_count == 0


class TestRetryExceptionFiltering:
    """Verify exception type filtering with real models."""

    def test_non_matching_exception_skips_retry(self):
        """When the exception doesn't match retry_on, no retries should occur."""
        bad_model = BedrockModel(model_id="us.anthropic.claude-nonexistent-v1:0")

        # Use a narrow exception type that won't match the actual AWS error
        plugin = RetryPlugin(max_retries=3, retry_on=[FileNotFoundError])

        agent = Agent(model=bad_model, plugins=[plugin])

        with pytest.raises(Exception):
            agent("Say hello.")

        # No retries because the exception type didn't match
        assert plugin.attempt_count == 0
