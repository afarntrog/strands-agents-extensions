"""Tests for the ModelFallbackPlugin."""

from unittest.mock import MagicMock

import pytest
from strands.hooks.events import AfterInvocationEvent, AfterModelCallEvent
from strands.types.exceptions import ModelThrottledException


class TestModelFallbackPluginInit:
    """Tests for ModelFallbackPlugin initialization."""

    def test_name(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(fallback_models=[fallback])
        assert plugin.name == "model-fallback"

    def test_empty_fallback_list_raises(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        with pytest.raises(ValueError, match="at least one"):
            ModelFallbackPlugin(fallback_models=[])

    def test_default_retry_on(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(fallback_models=[fallback])
        assert plugin._retry_on == [Exception]

    def test_custom_retry_on(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(
            fallback_models=[fallback],
            retry_on=[ModelThrottledException, ValueError],
        )
        assert plugin._retry_on == [ModelThrottledException, ValueError]

    def test_default_max_fallback_attempts(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallbacks = [MagicMock(), MagicMock(), MagicMock()]
        plugin = ModelFallbackPlugin(fallback_models=fallbacks)
        assert plugin._max_fallback_attempts == 3

    def test_custom_max_fallback_attempts(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallbacks = [MagicMock(), MagicMock(), MagicMock()]
        plugin = ModelFallbackPlugin(fallback_models=fallbacks, max_fallback_attempts=1)
        assert plugin._max_fallback_attempts == 1

    def test_default_cooldown(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(fallback_models=[fallback])
        assert plugin._cooldown_seconds == 0.0


class TestModelFallbackPluginHookRegistration:
    """Tests for hook registration."""

    def test_registers_hooks(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(fallback_models=[fallback])
        agent = MagicMock()
        plugin.init_agent(agent)
        assert agent.add_hook.call_count == 2


class TestModelFallbackOnSuccess:
    """Tests that successful calls are not affected."""

    def test_successful_call_no_fallback(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(fallback_models=[fallback])

        agent = MagicMock()
        stop_response = AfterModelCallEvent.ModelStopResponse(
            message={"role": "assistant", "content": [{"text": "Hello"}]},
            stop_reason="end_turn",
        )
        event = AfterModelCallEvent(agent=agent, stop_response=stop_response)
        plugin._on_after_model_call(event)

        assert event.retry is False
        # Model should not have been swapped
        assert agent.model == agent.model


class TestModelFallbackOnFailure:
    """Tests for fallback behavior on model failure."""

    def test_single_fallback_swaps_model(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        primary_model = MagicMock(name="primary")
        fallback_model = MagicMock(name="fallback")
        plugin = ModelFallbackPlugin(fallback_models=[fallback_model])

        agent = MagicMock()
        agent.model = primary_model

        event = AfterModelCallEvent(agent=agent, exception=RuntimeError("provider down"))
        plugin._on_after_model_call(event)

        assert event.retry is True
        assert agent.model is fallback_model

    def test_original_model_saved(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        primary_model = MagicMock(name="primary")
        fallback_model = MagicMock(name="fallback")
        plugin = ModelFallbackPlugin(fallback_models=[fallback_model])

        agent = MagicMock()
        agent.model = primary_model

        event = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event)

        assert plugin._original_model is primary_model

    def test_fallback_chain_a_fails_b_fails_c_used(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        primary = MagicMock(name="primary")
        fallback_a = MagicMock(name="fallback_a")
        fallback_b = MagicMock(name="fallback_b")
        fallback_c = MagicMock(name="fallback_c")
        plugin = ModelFallbackPlugin(fallback_models=[fallback_a, fallback_b, fallback_c])

        agent = MagicMock()
        agent.model = primary

        # First failure: should switch to fallback_a
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        assert agent.model is fallback_a
        assert event1.retry is True

        # Second failure (fallback_a failed): should switch to fallback_b
        event2 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail again"))
        plugin._on_after_model_call(event2)
        assert agent.model is fallback_b
        assert event2.retry is True

        # Third failure (fallback_b failed): should switch to fallback_c
        event3 = AfterModelCallEvent(agent=agent, exception=RuntimeError("still failing"))
        plugin._on_after_model_call(event3)
        assert agent.model is fallback_c
        assert event3.retry is True

    def test_max_attempts_respected(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback_a = MagicMock(name="fallback_a")
        fallback_b = MagicMock(name="fallback_b")
        plugin = ModelFallbackPlugin(
            fallback_models=[fallback_a, fallback_b],
            max_fallback_attempts=1,
        )

        agent = MagicMock()
        agent.model = MagicMock(name="primary")

        # First failure: uses one attempt
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        assert event1.retry is True

        # Second failure: max attempts reached, no retry
        event2 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail again"))
        plugin._on_after_model_call(event2)
        assert event2.retry is False

    def test_all_fallbacks_exhausted(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock(name="fallback")
        plugin = ModelFallbackPlugin(fallback_models=[fallback])

        agent = MagicMock()
        agent.model = MagicMock(name="primary")

        # First failure: switches to fallback
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        assert event1.retry is True

        # Second failure: no more fallbacks
        event2 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event2)
        assert event2.retry is False


class TestModelFallbackExceptionFiltering:
    """Tests for exception type filtering."""

    def test_matching_exception_triggers_fallback(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(
            fallback_models=[fallback],
            retry_on=[ModelThrottledException],
        )

        agent = MagicMock()
        agent.model = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=ModelThrottledException("throttled"))
        plugin._on_after_model_call(event)
        assert event.retry is True

    def test_non_matching_exception_no_fallback(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(
            fallback_models=[fallback],
            retry_on=[ModelThrottledException],
        )

        agent = MagicMock()
        agent.model = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=ValueError("bad input"))
        plugin._on_after_model_call(event)
        assert event.retry is False

    def test_subclass_exception_matches(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(
            fallback_models=[fallback],
            retry_on=[Exception],
        )

        agent = MagicMock()
        agent.model = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=ModelThrottledException("throttled"))
        plugin._on_after_model_call(event)
        assert event.retry is True


class TestModelFallbackRetryConflict:
    """Tests for interaction with other retry hooks."""

    def test_skips_if_retry_already_set(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(fallback_models=[fallback])

        agent = MagicMock()
        agent.model = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        event.retry = True  # Another hook already set retry
        plugin._on_after_model_call(event)

        # Should not have swapped model since retry was already handled
        assert agent.model is agent.model  # unchanged


class TestModelFallbackCooldown:
    """Tests for provider cooldown."""

    def test_cooled_down_model_skipped(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback_a = MagicMock(name="fallback_a")
        fallback_b = MagicMock(name="fallback_b")
        plugin = ModelFallbackPlugin(
            fallback_models=[fallback_a, fallback_b],
            cooldown_seconds=60.0,
        )

        agent = MagicMock()
        primary = MagicMock(name="primary")
        agent.model = primary

        # First failure: switches to fallback_a
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        assert agent.model is fallback_a

        # fallback_a fails: should skip back to fallback_a (it's cooled down),
        # and go to fallback_b
        event2 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event2)
        assert agent.model is fallback_b


class TestModelFallbackRestoreOnInvocationEnd:
    """Tests for restoring original model after invocation."""

    def test_original_model_restored(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        primary = MagicMock(name="primary")
        fallback = MagicMock(name="fallback")
        plugin = ModelFallbackPlugin(fallback_models=[fallback])

        agent = MagicMock()
        agent.model = primary

        # Trigger fallback
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        assert agent.model is fallback

        # Invocation ends
        event2 = AfterInvocationEvent(agent=agent)
        plugin._on_after_invocation(event2)

        assert agent.model is primary

    def test_state_reset_after_invocation(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(fallback_models=[fallback])

        agent = MagicMock()
        agent.model = MagicMock()

        # Trigger fallback
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)

        # Invocation ends
        event2 = AfterInvocationEvent(agent=agent)
        plugin._on_after_invocation(event2)

        # State should be reset
        assert plugin._original_model is None
        assert plugin._fallback_index == 0
        assert plugin._attempt_count == 0

    def test_no_restore_if_no_fallback_occurred(self):
        from strands_agents_extensions.plugins import ModelFallbackPlugin

        fallback = MagicMock()
        plugin = ModelFallbackPlugin(fallback_models=[fallback])

        agent = MagicMock()
        original = MagicMock(name="original")
        agent.model = original

        # Invocation ends without any fallback
        event = AfterInvocationEvent(agent=agent)
        plugin._on_after_invocation(event)

        # Model should still be the original
        assert agent.model is original
