"""Tests for the RateLimiterPlugin."""

import time
from unittest.mock import MagicMock

import pytest
from strands.hooks.events import BeforeModelCallEvent, BeforeToolCallEvent

from strands_agents_extensions.plugins import (
    OnLimitAction,
    RateLimiterPlugin,
    RateLimitExceededException,
    RateLimitType,
)
from strands_agents_extensions.plugins.rate_limiter import _SlidingWindowCounter


class TestSlidingWindowCounter:
    """Tests for the _SlidingWindowCounter."""

    def test_try_acquire_within_limit(self):
        counter = _SlidingWindowCounter(max_calls=3, window_seconds=60.0)
        assert counter.try_acquire() is True
        assert counter.try_acquire() is True
        assert counter.try_acquire() is True

    def test_try_acquire_exceeds_limit(self):
        counter = _SlidingWindowCounter(max_calls=2, window_seconds=60.0)
        assert counter.try_acquire() is True
        assert counter.try_acquire() is True
        assert counter.try_acquire() is False

    def test_window_expiry(self):
        counter = _SlidingWindowCounter(max_calls=1, window_seconds=0.1)
        assert counter.try_acquire() is True
        assert counter.try_acquire() is False
        time.sleep(0.15)
        assert counter.try_acquire() is True

    def test_wait_and_acquire_no_wait(self):
        counter = _SlidingWindowCounter(max_calls=5, window_seconds=60.0)
        waited = counter.wait_and_acquire()
        assert waited == 0.0

    def test_wait_and_acquire_waits(self):
        counter = _SlidingWindowCounter(max_calls=1, window_seconds=0.1)
        counter.try_acquire()
        waited = counter.wait_and_acquire()
        assert waited >= 0.05


class TestRateLimiterPluginInit:
    """Tests for RateLimiterPlugin initialization."""

    def test_name(self):
        plugin = RateLimiterPlugin(model_calls_per_minute=60)
        assert plugin.name == "rate-limiter"

    def test_invalid_on_limit_raises(self):
        with pytest.raises(ValueError):
            RateLimiterPlugin(model_calls_per_minute=60, on_limit="invalid")

    def test_no_limits_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            RateLimiterPlugin()

    def test_model_only(self):
        plugin = RateLimiterPlugin(model_calls_per_minute=60)
        assert plugin._model_limiter is not None
        assert plugin._tool_limiter is None

    def test_tool_only(self):
        plugin = RateLimiterPlugin(tool_calls_per_minute=100)
        assert plugin._model_limiter is None
        assert plugin._tool_limiter is not None

    def test_both_limits(self):
        plugin = RateLimiterPlugin(model_calls_per_minute=60, tool_calls_per_minute=100)
        assert plugin._model_limiter is not None
        assert plugin._tool_limiter is not None


class TestRateLimiterPluginHookRegistration:
    """Tests for hook registration."""

    def test_registers_model_hook_only(self):
        plugin = RateLimiterPlugin(model_calls_per_minute=60)
        agent = MagicMock()
        plugin.init_agent(agent)
        assert agent.add_hook.call_count == 1

    def test_registers_tool_hook_only(self):
        plugin = RateLimiterPlugin(tool_calls_per_minute=100)
        agent = MagicMock()
        plugin.init_agent(agent)
        assert agent.add_hook.call_count == 1

    def test_registers_both_hooks(self):
        plugin = RateLimiterPlugin(model_calls_per_minute=60, tool_calls_per_minute=100)
        agent = MagicMock()
        plugin.init_agent(agent)
        assert agent.add_hook.call_count == 2


class TestRateLimiterPluginModelLimit:
    """Tests for model call rate limiting."""

    def test_model_calls_within_limit(self):
        plugin = RateLimiterPlugin(model_calls_per_minute=5, on_limit=OnLimitAction.ERROR)
        agent = MagicMock()
        plugin.init_agent(agent)

        event = BeforeModelCallEvent(agent=agent)
        for _ in range(5):
            plugin._on_before_model_call(event)

    def test_model_calls_exceed_limit_error(self):
        plugin = RateLimiterPlugin(model_calls_per_minute=2, on_limit=OnLimitAction.ERROR)
        agent = MagicMock()
        plugin.init_agent(agent)

        event = BeforeModelCallEvent(agent=agent)
        plugin._on_before_model_call(event)
        plugin._on_before_model_call(event)

        with pytest.raises(RateLimitExceededException, match="model"):
            plugin._on_before_model_call(event)

    def test_model_calls_exceed_limit_wait(self):
        plugin = RateLimiterPlugin(model_calls_per_minute=1, on_limit=OnLimitAction.WAIT)
        plugin._model_limiter = _SlidingWindowCounter(max_calls=1, window_seconds=0.1)
        agent = MagicMock()
        plugin.init_agent(agent)

        event = BeforeModelCallEvent(agent=agent)
        plugin._on_before_model_call(event)

        start = time.monotonic()
        plugin._on_before_model_call(event)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.05


class TestRateLimiterPluginToolLimit:
    """Tests for tool call rate limiting."""

    def test_tool_calls_within_limit(self):
        plugin = RateLimiterPlugin(tool_calls_per_minute=5, on_limit=OnLimitAction.ERROR)
        agent = MagicMock()
        plugin.init_agent(agent)

        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "test", "name": "my_tool", "input": {}},
            invocation_state={},
        )
        for _ in range(5):
            plugin._on_before_tool_call(event)

    def test_tool_calls_exceed_limit_error(self):
        plugin = RateLimiterPlugin(tool_calls_per_minute=2, on_limit=OnLimitAction.ERROR)
        agent = MagicMock()
        plugin.init_agent(agent)

        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "test", "name": "my_tool", "input": {}},
            invocation_state={},
        )
        plugin._on_before_tool_call(event)
        plugin._on_before_tool_call(event)

        with pytest.raises(RateLimitExceededException, match="tool"):
            plugin._on_before_tool_call(event)


class TestRateLimitExceededException:
    """Tests for RateLimitExceededException."""

    def test_attributes(self):
        exc = RateLimitExceededException(RateLimitType.MODEL, 60)
        assert exc.limit_type is RateLimitType.MODEL
        assert exc.calls_per_minute == 60

    def test_message(self):
        exc = RateLimitExceededException(RateLimitType.TOOL, 100)
        assert "tool" in str(exc)
        assert "100" in str(exc)
