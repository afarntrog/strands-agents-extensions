"""Tests for the RetryPlugin."""

import time
from unittest.mock import MagicMock, patch

import pytest
from strands.hooks.events import AfterModelCallEvent
from strands.types.exceptions import ModelThrottledException


class TestRetryPluginInit:
    """Tests for RetryPlugin initialization."""

    def test_name(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        assert plugin.name == "retry"

    def test_default_retry_on(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        assert plugin._retry_on == [Exception]

    def test_custom_retry_on(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(
            max_retries=2,
            retry_on=[ModelThrottledException, ConnectionError],
        )
        assert plugin._retry_on == [ModelThrottledException, ConnectionError]

    def test_zero_max_retries_raises(self):
        from strands_agents_extensions.plugins import RetryPlugin

        with pytest.raises(ValueError, match="max_retries must be at least 1"):
            RetryPlugin(max_retries=0)

    def test_negative_max_retries_raises(self):
        from strands_agents_extensions.plugins import RetryPlugin

        with pytest.raises(ValueError, match="max_retries must be at least 1"):
            RetryPlugin(max_retries=-1)

    def test_default_delay(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        assert plugin._delay_seconds == 0.0

    def test_custom_delay(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3, delay_seconds=1.5)
        assert plugin._delay_seconds == 1.5

    def test_default_backoff_factor(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        assert plugin._backoff_factor == 1.0

    def test_custom_backoff_factor(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3, backoff_factor=2.0)
        assert plugin._backoff_factor == 2.0


class TestRetryPluginHookRegistration:
    """Tests for hook registration."""

    def test_registers_hooks(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        agent = MagicMock()
        plugin.init_agent(agent)
        assert agent.add_hook.call_count == 1


class TestRetryOnSuccess:
    """Tests that successful calls are not affected."""

    def test_successful_call_no_retry(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)

        agent = MagicMock()
        stop_response = AfterModelCallEvent.ModelStopResponse(
            message={"role": "assistant", "content": [{"text": "Hello"}]},
            stop_reason="end_turn",
        )
        event = AfterModelCallEvent(agent=agent, stop_response=stop_response)
        plugin._on_after_model_call(event)

        assert event.retry is False


class TestRetryOnFailure:
    """Tests for retry behavior on model failure."""

    def test_failure_triggers_retry(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        agent = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=RuntimeError("transient"))
        plugin._on_after_model_call(event)

        assert event.retry is True

    def test_model_not_swapped(self):
        """Retry should keep the same model, not swap it."""
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        agent = MagicMock()
        original_model = agent.model

        event = AfterModelCallEvent(agent=agent, exception=RuntimeError("transient"))
        plugin._on_after_model_call(event)

        assert agent.model is original_model

    def test_max_retries_respected(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=2)
        agent = MagicMock()

        # First failure: retry
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        assert event1.retry is True

        # Second failure: retry
        event2 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event2)
        assert event2.retry is True

        # Third failure: exhausted, no retry
        event3 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event3)
        assert event3.retry is False

    def test_attempt_count_tracks(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        agent = MagicMock()

        assert plugin.attempt_count == 0

        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        assert plugin.attempt_count == 1

        event2 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event2)
        assert plugin.attempt_count == 2

    def test_attempt_count_resets_on_success(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        agent = MagicMock()

        # Fail once
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        assert plugin.attempt_count == 1

        # Succeed
        stop_response = AfterModelCallEvent.ModelStopResponse(
            message={"role": "assistant", "content": [{"text": "ok"}]},
            stop_reason="end_turn",
        )
        event2 = AfterModelCallEvent(agent=agent, stop_response=stop_response)
        plugin._on_after_model_call(event2)
        assert plugin.attempt_count == 0


class TestRetryExceptionFiltering:
    """Tests for exception type filtering."""

    def test_matching_exception_triggers_retry(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3, retry_on=[ModelThrottledException])
        agent = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=ModelThrottledException("throttled"))
        plugin._on_after_model_call(event)
        assert event.retry is True

    def test_non_matching_exception_no_retry(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3, retry_on=[ModelThrottledException])
        agent = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=ValueError("bad input"))
        plugin._on_after_model_call(event)
        assert event.retry is False

    def test_subclass_exception_matches(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3, retry_on=[Exception])
        agent = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=ModelThrottledException("throttled"))
        plugin._on_after_model_call(event)
        assert event.retry is True


class TestRetryConflict:
    """Tests for interaction with other retry hooks."""

    def test_skips_if_retry_already_set(self):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        agent = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        event.retry = True  # Another hook already set retry
        plugin._on_after_model_call(event)

        # Should not have consumed an attempt
        assert plugin.attempt_count == 0


class TestRetryDelay:
    """Tests for delay and exponential backoff."""

    @patch("time.sleep")
    def test_fixed_delay(self, mock_sleep):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3, delay_seconds=1.0)
        agent = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event)

        mock_sleep.assert_called_once_with(1.0)

    @patch("time.sleep")
    def test_no_delay_by_default(self, mock_sleep):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=3)
        agent = MagicMock()

        event = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event)

        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_exponential_backoff(self, mock_sleep):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=4, delay_seconds=1.0, backoff_factor=2.0)
        agent = MagicMock()

        # Attempt 1: delay = 1.0 * 2.0^0 = 1.0
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        mock_sleep.assert_called_with(1.0)

        # Attempt 2: delay = 1.0 * 2.0^1 = 2.0
        event2 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event2)
        mock_sleep.assert_called_with(2.0)

        # Attempt 3: delay = 1.0 * 2.0^2 = 4.0
        event3 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event3)
        mock_sleep.assert_called_with(4.0)

    @patch("time.sleep")
    def test_no_delay_when_exhausted(self, mock_sleep):
        from strands_agents_extensions.plugins import RetryPlugin

        plugin = RetryPlugin(max_retries=1, delay_seconds=1.0)
        agent = MagicMock()

        # First: retries with delay
        event1 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event1)
        assert mock_sleep.call_count == 1

        # Second: exhausted, no retry, no delay
        event2 = AfterModelCallEvent(agent=agent, exception=RuntimeError("fail"))
        plugin._on_after_model_call(event2)
        assert mock_sleep.call_count == 1  # no additional sleep
