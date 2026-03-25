r"""Message redaction plugin for scrubbing sensitive data from agent conversations.

This module provides a plugin that scans and redacts sensitive data (PII, credentials,
etc.) from messages before they reach the model provider. It uses configurable regex
patterns to detect and replace sensitive content in user messages, tool inputs, and
tool results.

Example Usage:
    ```python
    from strands import Agent
    from strands_agents_extensions.plugins import MessageRedactionPlugin

    plugin = MessageRedactionPlugin(
        patterns={
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
        },
        replacement="[REDACTED:{name}]",
    )
    agent = Agent(plugins=[plugin])
    ```
"""

import logging
import re
from typing import TYPE_CHECKING, Any

from strands.hooks.events import AfterToolCallEvent, BeforeInvocationEvent, BeforeToolCallEvent
from strands.plugins import Plugin
from strands.types.content import ContentBlock, Messages

if TYPE_CHECKING:
    from strands.agent import Agent

logger = logging.getLogger(__name__)


class MessageRedactionPlugin(Plugin):
    r"""Plugin that redacts sensitive data from messages using regex patterns.

    Scans user messages, tool inputs, and tool results for sensitive data
    and replaces matches with a configurable redaction marker.

    Attributes:
        name: Plugin identifier ("message-redaction").

    Example:
        ```python
        from strands import Agent
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
                "aws_key": r"\bAKIA[0-9A-Z]{16}\b",
            },
            scan_tool_inputs=True,
            scan_tool_results=True,
        )
        agent = Agent(plugins=[plugin])
        ```
    """

    name: str = "message-redaction"

    def __init__(
        self,
        patterns: dict[str, str],
        replacement: str = "[REDACTED:{name}]",
        scan_tool_inputs: bool = True,
        scan_tool_results: bool = True,
        log_redactions: bool = True,
    ) -> None:
        """Initialize the message redaction plugin.

        Args:
            patterns: Mapping of pattern names to regex strings. Each pattern is compiled
                and used to scan text for sensitive data.
            replacement: Template for the replacement text. Use ``{name}`` to include
                the pattern name in the redaction marker.
            scan_tool_inputs: Whether to redact string values in tool call inputs.
            scan_tool_results: Whether to redact text content blocks in tool results.
            log_redactions: Whether to log when redactions occur.

        Raises:
            ValueError: If patterns is empty or contains an invalid regex.
        """
        if not patterns:
            raise ValueError("at least one pattern must be provided")

        self._replacement = replacement
        self._scan_tool_inputs = scan_tool_inputs
        self._scan_tool_results = scan_tool_results
        self._log_redactions = log_redactions
        self._redaction_count: int = 0

        self._compiled_patterns: dict[str, re.Pattern[str]] = {}
        for name, pattern in patterns.items():
            try:
                self._compiled_patterns[name] = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"invalid regex for pattern '{name}': {e}") from e

        super().__init__()

    def init_agent(self, agent: "Agent") -> None:
        """Register hooks for message redaction on the agent.

        Args:
            agent: The agent instance to attach redaction to.
        """
        agent.add_hook(self._on_before_invocation)

        if self._scan_tool_inputs:
            agent.add_hook(self._on_before_tool_call)

        if self._scan_tool_results:
            agent.add_hook(self._on_after_tool_call)

    @property
    def redaction_count(self) -> int:
        """Total number of redactions performed across all calls."""
        return self._redaction_count

    def redact_text(self, text: str) -> tuple[str, int]:
        """Apply all redaction patterns to a text string.

        Args:
            text: The text to scan and redact.

        Returns:
            A tuple of (redacted_text, number_of_redactions).
        """
        total_count = 0
        for name, pattern in self._compiled_patterns.items():
            replacement = self._replacement.format(name=name)
            text, count = pattern.subn(replacement, text)
            total_count += count

        self._redaction_count += total_count
        return text, total_count

    def redact_content_blocks(self, blocks: list[ContentBlock]) -> int:
        """Redact sensitive data in a list of content blocks.

        Scans ``text`` fields and ``toolResult`` content blocks in place.

        Args:
            blocks: The content blocks to scan and redact.

        Returns:
            The number of redactions performed.
        """
        total_count = 0
        for block in blocks:
            if "text" in block:
                redacted, count = self.redact_text(block["text"])
                if count > 0:
                    block["text"] = redacted
                    total_count += count

            if "toolResult" in block:
                tool_result = block["toolResult"]
                if "content" in tool_result:
                    total_count += self.redact_content_blocks(tool_result["content"])

        return total_count

    def redact_messages(self, messages: Messages) -> int:
        """Redact sensitive data across all messages.

        Args:
            messages: The messages to scan and redact.

        Returns:
            The number of redactions performed.
        """
        total_count = 0
        for message in messages:
            total_count += self.redact_content_blocks(message["content"])
        return total_count

    def _redact_dict_strings(self, data: Any) -> int:
        """Recursively redact string values in a dict or list structure.

        Args:
            data: The data structure to scan. Dicts and lists are traversed
                recursively; string values are redacted in place.

        Returns:
            The number of redactions performed.
        """
        total_count = 0
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], str):
                    redacted, count = self.redact_text(data[key])
                    if count > 0:
                        data[key] = redacted
                        total_count += count
                elif isinstance(data[key], (dict, list)):
                    total_count += self._redact_dict_strings(data[key])
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    redacted, count = self.redact_text(item)
                    if count > 0:
                        data[i] = redacted
                        total_count += count
                elif isinstance(item, (dict, list)):
                    total_count += self._redact_dict_strings(item)
        return total_count

    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Redact sensitive data in incoming messages before processing.

        Args:
            event: The before invocation event.
        """
        if event.messages is None:
            return

        count = self.redact_messages(event.messages)
        if count > 0 and self._log_redactions:
            logger.info("redaction_count=<%d> | redacted sensitive data from invocation messages", count)

    def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Redact sensitive data in tool call inputs.

        Args:
            event: The before tool call event.
        """
        if not self._scan_tool_inputs:
            return

        tool_input = event.tool_use.get("input")
        if tool_input is None:
            return

        count = self._redact_dict_strings(tool_input)
        if count > 0 and self._log_redactions:
            tool_name = event.tool_use.get("name", "unknown")
            logger.info(
                "tool_name=<%s>, redaction_count=<%d> | redacted sensitive data from tool input",
                tool_name,
                count,
            )

    def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Redact sensitive data in tool results.

        Args:
            event: The after tool call event.
        """
        if not self._scan_tool_results:
            return

        result_content = event.result.get("content")
        if not result_content:
            return

        count = self.redact_content_blocks(result_content)
        if count > 0 and self._log_redactions:
            tool_name = event.tool_use.get("name", "unknown")
            logger.info(
                "tool_name=<%s>, redaction_count=<%d> | redacted sensitive data from tool result",
                tool_name,
                count,
            )
