"""Session managers for the Strands Agents SDK."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sqlite import SQLiteSessionManager

__all__ = ["SQLiteSessionManager"]


def __getattr__(name):
    if name == "SQLiteSessionManager":
        from .sqlite import SQLiteSessionManager

        return SQLiteSessionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
