"""Tests for lazy import behavior and helpful error messages."""

import importlib


def test_plugins_module_imports_without_extras():
    """Importing the plugins module itself should not fail."""
    mod = importlib.import_module("strands_agents_extensions.plugins")
    assert hasattr(mod, "__all__")


def test_session_managers_module_imports_without_extras():
    """Importing the session_managers module itself should not fail."""
    mod = importlib.import_module("strands_agents_extensions.session_managers")
    assert hasattr(mod, "__all__")


def test_plugins_all_lists_expected_names():
    """The __all__ list should contain all public plugin symbols."""
    from strands_agents_extensions import plugins

    expected = {
        "BudgetPlugin", "BudgetExceededException", "BudgetLimitType", "OnExceedAction",
        "RateLimiterPlugin", "RateLimitExceededException", "RateLimitType", "OnLimitAction",
        "MessageRedactionPlugin",
        "ModelFallbackPlugin",
    }
    assert set(plugins.__all__) == expected


def test_session_managers_all_lists_expected_names():
    """The __all__ list should contain all public session manager symbols."""
    from strands_agents_extensions import session_managers

    assert "SQLiteSessionManager" in session_managers.__all__


def test_unknown_attribute_raises_attribute_error():
    """Accessing a nonexistent name should raise AttributeError, not ImportError."""
    import pytest

    from strands_agents_extensions import plugins

    with pytest.raises(AttributeError):
        _ = plugins.NonExistentThing
