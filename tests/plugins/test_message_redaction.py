"""Tests for the MessageRedactionPlugin."""

from unittest.mock import MagicMock

import pytest
from strands.hooks.events import AfterToolCallEvent, BeforeInvocationEvent, BeforeToolCallEvent
from strands.types.content import ContentBlock, Message


class TestMessageRedactionPluginInit:
    """Tests for MessageRedactionPlugin initialization."""

    def test_name(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        assert plugin.name == "message-redaction"

    def test_empty_patterns_raises(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        with pytest.raises(ValueError, match="at least one"):
            MessageRedactionPlugin(patterns={})

    def test_invalid_regex_raises(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        with pytest.raises(ValueError, match="invalid regex"):
            MessageRedactionPlugin(patterns={"bad": r"[invalid"})

    def test_custom_replacement(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
            replacement="***{name}***",
        )
        assert plugin._replacement == "***{name}***"

    def test_default_replacement(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        assert plugin._replacement == "[REDACTED:{name}]"

    def test_patterns_compiled(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
            },
        )
        assert len(plugin._compiled_patterns) == 2
        assert "ssn" in plugin._compiled_patterns
        assert "email" in plugin._compiled_patterns


class TestMessageRedactionPluginHookRegistration:
    """Tests for hook registration."""

    def test_registers_all_hooks(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        agent = MagicMock()
        plugin.init_agent(agent)
        assert agent.add_hook.call_count == 3

    def test_registers_fewer_hooks_without_tool_scanning(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
            scan_tool_inputs=False,
            scan_tool_results=False,
        )
        agent = MagicMock()
        plugin.init_agent(agent)
        # Only before_invocation hook
        assert agent.add_hook.call_count == 1


class TestRedactText:
    """Tests for the core text redaction logic."""

    def test_ssn_redacted(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        result, count = plugin.redact_text("My SSN is 123-45-6789")
        assert result == "My SSN is [REDACTED:ssn]"
        assert count == 1

    def test_email_redacted(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"email": r"\b[\w.-]+@[\w.-]+\.\w+\b"})
        result, count = plugin.redact_text("Contact me at user@example.com please")
        assert result == "Contact me at [REDACTED:email] please"
        assert count == 1

    def test_credit_card_redacted(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"})
        result, count = plugin.redact_text("Card: 4111-1111-1111-1111")
        assert result == "Card: [REDACTED:credit_card]"
        assert count == 1

    def test_aws_key_redacted(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"aws_key": r"\bAKIA[0-9A-Z]{16}\b"})
        result, count = plugin.redact_text("Key: AKIAIOSFODNN7EXAMPLE")
        assert result == "Key: [REDACTED:aws_key]"
        assert count == 1

    def test_multiple_patterns_in_single_text(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
            },
        )
        result, count = plugin.redact_text("SSN: 123-45-6789, email: test@example.com")
        assert "[REDACTED:ssn]" in result
        assert "[REDACTED:email]" in result
        assert count == 2

    def test_multiple_matches_of_same_pattern(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        result, count = plugin.redact_text("SSNs: 123-45-6789 and 987-65-4321")
        assert result == "SSNs: [REDACTED:ssn] and [REDACTED:ssn]"
        assert count == 2

    def test_no_match_unchanged(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        result, count = plugin.redact_text("Nothing sensitive here")
        assert result == "Nothing sensitive here"
        assert count == 0

    def test_custom_replacement_template(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
            replacement="***{name}***",
        )
        result, count = plugin.redact_text("SSN: 123-45-6789")
        assert result == "SSN: ***ssn***"
        assert count == 1

    def test_empty_string(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        result, count = plugin.redact_text("")
        assert result == ""
        assert count == 0


class TestRedactionCount:
    """Tests for redaction count tracking."""

    def test_initial_count_zero(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        assert plugin.redaction_count == 0

    def test_count_increments(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        plugin.redact_text("SSN: 123-45-6789")
        assert plugin.redaction_count == 1
        plugin.redact_text("SSNs: 111-22-3333 and 444-55-6666")
        assert plugin.redaction_count == 3


class TestRedactContentBlocks:
    """Tests for content block redaction."""

    def test_text_block_redacted(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        blocks: list[ContentBlock] = [{"text": "SSN: 123-45-6789"}]
        count = plugin.redact_content_blocks(blocks)
        assert blocks[0]["text"] == "SSN: [REDACTED:ssn]"
        assert count == 1

    def test_non_text_block_untouched(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        blocks: list[ContentBlock] = [{"image": {"format": "png", "source": {"bytes": b"data"}}}]
        count = plugin.redact_content_blocks(blocks)
        assert count == 0
        assert "image" in blocks[0]

    def test_tool_result_content_redacted(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        blocks: list[ContentBlock] = [
            {
                "toolResult": {
                    "toolUseId": "test-id",
                    "content": [{"text": "SSN: 123-45-6789"}],
                    "status": "success",
                }
            }
        ]
        count = plugin.redact_content_blocks(blocks)
        assert blocks[0]["toolResult"]["content"][0]["text"] == "SSN: [REDACTED:ssn]"
        assert count == 1

    def test_mixed_blocks(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        blocks: list[ContentBlock] = [
            {"text": "SSN: 123-45-6789"},
            {"text": "No PII here"},
            {"image": {"format": "png", "source": {"bytes": b"data"}}},
        ]
        count = plugin.redact_content_blocks(blocks)
        assert blocks[0]["text"] == "SSN: [REDACTED:ssn]"
        assert blocks[1]["text"] == "No PII here"
        assert count == 1

    def test_empty_blocks(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        blocks: list[ContentBlock] = []
        count = plugin.redact_content_blocks(blocks)
        assert count == 0


class TestRedactMessages:
    """Tests for message-level redaction."""

    def test_user_messages_redacted(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "My SSN is 123-45-6789"}]},
        ]
        count = plugin.redact_messages(messages)
        assert messages[0]["content"][0]["text"] == "My SSN is [REDACTED:ssn]"
        assert count == 1

    def test_assistant_messages_redacted(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        messages: list[Message] = [
            {"role": "assistant", "content": [{"text": "Your SSN is 123-45-6789"}]},
        ]
        count = plugin.redact_messages(messages)
        assert messages[0]["content"][0]["text"] == "Your SSN is [REDACTED:ssn]"
        assert count == 1

    def test_multiple_messages(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        messages: list[Message] = [
            {"role": "user", "content": [{"text": "SSN: 111-22-3333"}]},
            {"role": "assistant", "content": [{"text": "Got it"}]},
            {"role": "user", "content": [{"text": "Also 444-55-6666"}]},
        ]
        count = plugin.redact_messages(messages)
        assert count == 2

    def test_empty_messages(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        messages: list[Message] = []
        count = plugin.redact_messages(messages)
        assert count == 0


class TestBeforeInvocationHook:
    """Tests for the before_invocation hook."""

    def test_redacts_invocation_messages(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        agent = MagicMock()

        messages = [
            {"role": "user", "content": [{"text": "My SSN is 123-45-6789"}]},
        ]
        event = BeforeInvocationEvent(agent=agent, messages=messages)
        plugin._on_before_invocation(event)

        assert messages[0]["content"][0]["text"] == "My SSN is [REDACTED:ssn]"

    def test_handles_none_messages(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"})
        agent = MagicMock()

        event = BeforeInvocationEvent(agent=agent, messages=None)
        # Should not raise
        plugin._on_before_invocation(event)
        assert plugin.redaction_count == 0


class TestBeforeToolCallHook:
    """Tests for the before_tool_call hook (tool input redaction)."""

    def test_redacts_string_tool_inputs(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
            scan_tool_inputs=True,
        )
        agent = MagicMock()

        tool_use = {
            "toolUseId": "test-id",
            "name": "my_tool",
            "input": {"query": "Look up SSN 123-45-6789", "count": 5},
        }
        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use=tool_use,
            invocation_state={},
        )
        plugin._on_before_tool_call(event)

        assert tool_use["input"]["query"] == "Look up SSN [REDACTED:ssn]"
        assert tool_use["input"]["count"] == 5  # Non-string values untouched

    def test_skips_when_disabled(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
            scan_tool_inputs=False,
        )
        agent = MagicMock()

        tool_use = {
            "toolUseId": "test-id",
            "name": "my_tool",
            "input": {"query": "SSN 123-45-6789"},
        }
        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use=tool_use,
            invocation_state={},
        )
        plugin._on_before_tool_call(event)
        assert tool_use["input"]["query"] == "SSN 123-45-6789"

    def test_nested_string_values(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
            scan_tool_inputs=True,
        )
        agent = MagicMock()

        tool_use = {
            "toolUseId": "test-id",
            "name": "my_tool",
            "input": {"data": {"nested": "SSN: 123-45-6789"}},
        }
        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use=tool_use,
            invocation_state={},
        )
        plugin._on_before_tool_call(event)
        assert tool_use["input"]["data"]["nested"] == "SSN: [REDACTED:ssn]"


class TestAfterToolCallHook:
    """Tests for the after_tool_call hook (tool result redaction)."""

    def test_redacts_tool_result_content(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
            scan_tool_results=True,
        )
        agent = MagicMock()

        result = {
            "toolUseId": "test-id",
            "content": [{"text": "Found SSN: 123-45-6789"}],
            "status": "success",
        }
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-id", "name": "my_tool", "input": {}},
            invocation_state={},
            result=result,
        )
        plugin._on_after_tool_call(event)

        assert result["content"][0]["text"] == "Found SSN: [REDACTED:ssn]"

    def test_skips_when_disabled(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
            scan_tool_results=False,
        )
        agent = MagicMock()

        result = {
            "toolUseId": "test-id",
            "content": [{"text": "SSN: 123-45-6789"}],
            "status": "success",
        }
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-id", "name": "my_tool", "input": {}},
            invocation_state={},
            result=result,
        )
        plugin._on_after_tool_call(event)
        assert result["content"][0]["text"] == "SSN: 123-45-6789"

    def test_handles_missing_content(self):
        from strands_agents_extensions.plugins import MessageRedactionPlugin

        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
            scan_tool_results=True,
        )
        agent = MagicMock()

        result = {
            "toolUseId": "test-id",
            "content": [],
            "status": "success",
        }
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "test-id", "name": "my_tool", "input": {}},
            invocation_state={},
            result=result,
        )
        # Should not raise
        plugin._on_after_tool_call(event)
