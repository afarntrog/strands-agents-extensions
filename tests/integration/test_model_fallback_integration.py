"""Integration tests for ModelFallbackPlugin with real Bedrock LLMs."""

from strands import Agent
from strands.models.bedrock import BedrockModel

from strands_agents_extensions.plugins import ModelFallbackPlugin

from .conftest import MODEL_ID, retry_on_flaky


class TestFallbackOnFailure:
    """Verify fallback triggers when primary model fails with real models."""

    @retry_on_flaky("LLM responses may throttle")
    def test_primary_fails_fallback_succeeds(self):
        """Agent should return a result from the fallback when primary fails."""
        # Use a nonexistent model ID to guarantee failure
        primary = BedrockModel(model_id="us.anthropic.claude-nonexistent-v1:0")
        fallback = BedrockModel(model_id=MODEL_ID)

        plugin = ModelFallbackPlugin(fallback_models=[fallback])
        agent = Agent(model=primary, plugins=[plugin])

        result = agent("Say hello in one word.")
        assert result is not None

    @retry_on_flaky("LLM responses may throttle")
    def test_original_model_restored_after_fallback(self):
        """After invocation, the original model should be restored."""
        primary = BedrockModel(model_id="us.anthropic.claude-nonexistent-v1:0")
        fallback = BedrockModel(model_id=MODEL_ID)

        plugin = ModelFallbackPlugin(fallback_models=[fallback])
        agent = Agent(model=primary, plugins=[plugin])

        agent("Say hello.")

        # Agent's model should be restored to primary after invocation
        assert agent.model is primary


class TestFallbackNoFailure:
    """Verify no fallback when primary succeeds with real model."""

    @retry_on_flaky("LLM responses may throttle")
    def test_successful_call_uses_primary(self, model):
        """When primary succeeds, fallback should not be used."""
        fallback = BedrockModel(model_id=MODEL_ID)

        plugin = ModelFallbackPlugin(fallback_models=[fallback])
        agent = Agent(model=model, plugins=[plugin])

        result = agent("Say hello in one word.")
        assert result is not None
        # Plugin state should show no fallback occurred
        assert plugin._original_model is None
