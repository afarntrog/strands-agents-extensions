"""Plugins for the Strands Agents SDK."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .budget import BudgetExceededException, BudgetLimitType, BudgetPlugin, OnExceedAction
    from .message_redaction import MessageRedactionPlugin
    from .model_fallback import ModelFallbackPlugin
    from .rate_limiter import OnLimitAction, RateLimitExceededException, RateLimiterPlugin, RateLimitType

__all__ = [
    "BudgetExceededException",
    "BudgetLimitType",
    "BudgetPlugin",
    "MessageRedactionPlugin",
    "ModelFallbackPlugin",
    "OnExceedAction",
    "OnLimitAction",
    "RateLimitExceededException",
    "RateLimiterPlugin",
    "RateLimitType",
]

_BUDGET_NAMES = {"BudgetPlugin", "BudgetExceededException", "BudgetLimitType", "OnExceedAction"}
_RATE_LIMITER_NAMES = {"RateLimiterPlugin", "RateLimitExceededException", "RateLimitType", "OnLimitAction"}


def __getattr__(name):
    if name in _BUDGET_NAMES:
        try:
            from . import budget

            return getattr(budget, name)
        except ImportError:
            raise ImportError(
                f"{name} requires the 'budget' extra: "
                "pip install strands-agents-extensions[budget]"
            ) from None
    if name in _RATE_LIMITER_NAMES:
        from . import rate_limiter

        return getattr(rate_limiter, name)
    if name == "MessageRedactionPlugin":
        from .message_redaction import MessageRedactionPlugin

        return MessageRedactionPlugin
    if name == "ModelFallbackPlugin":
        from .model_fallback import ModelFallbackPlugin

        return ModelFallbackPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
