"""Shared fixtures and utilities for integration tests.

These tests call real LLMs via Bedrock and require AWS credentials.
"""

import functools
import logging
import time
from collections.abc import Callable

import pytest
from strands.models.bedrock import BedrockModel

logger = logging.getLogger(__name__)

# Model used across all integration tests
MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


def retry_on_flaky(
    reason: str,
    *,
    max_attempts: int = 3,
    wait_seconds: float = 2.0,
) -> Callable:
    """Retry flaky integration tests due to non-deterministic LLM responses or throttling.

    Args:
        reason: Explanation of why this test is flaky and needs retries.
        max_attempts: Maximum number of retry attempts.
        wait_seconds: Wait time between retries in seconds.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        logger.warning(
                            "attempt=<%d/%d>, reason=<%s>, error=<%s> | retrying",
                            attempt + 1,
                            max_attempts,
                            reason,
                            exc,
                        )
                        time.sleep(wait_seconds * (attempt + 1))
            raise last_exc

        return wrapper

    return decorator


@pytest.fixture
def model():
    """Create a Bedrock model for integration testing."""
    return BedrockModel(model_id=MODEL_ID)
