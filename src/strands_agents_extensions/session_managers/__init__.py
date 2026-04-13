"""Session managers for the Strands Agents SDK."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dynamodb import DynamoDBSessionManager
    from .sqlite import SQLiteSessionManager

__all__ = ["DynamoDBSessionManager", "SQLiteSessionManager"]


def __getattr__(name):
    if name == "SQLiteSessionManager":
        from .sqlite import SQLiteSessionManager

        return SQLiteSessionManager
    if name == "DynamoDBSessionManager":
        from .dynamodb import DynamoDBSessionManager

        return DynamoDBSessionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
