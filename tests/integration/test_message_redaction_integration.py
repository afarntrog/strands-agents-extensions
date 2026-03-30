"""Integration tests for MessageRedactionPlugin with a real Bedrock LLM."""

from strands import Agent

from strands_agents_extensions.plugins import MessageRedactionPlugin

from .conftest import retry_on_flaky


class TestMessageRedactionInvocation:
    """Verify redaction happens during invocations."""

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_user_message_redacted_before_model(self, model):
        """PII in user messages should be redacted before reaching the model."""
        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
        )
        agent = Agent(model=model, plugins=[plugin])

        agent("My SSN is 123-45-6789. What did I just tell you?")

        # Verify redaction happened
        assert plugin.redaction_count >= 1

        # Check the first user message in the conversation was redacted
        user_msg = agent.messages[0]
        user_text = user_msg["content"][0]["text"]
        assert "123-45-6789" not in user_text
        assert "[REDACTED:ssn]" in user_text

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_multiple_patterns_redacted(self, model):
        """Multiple PII types should all be redacted."""
        plugin = MessageRedactionPlugin(
            patterns={
                "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
                "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
            },
        )
        agent = Agent(model=model, plugins=[plugin])

        agent("SSN: 123-45-6789, email: alice@example.com")

        user_msg = agent.messages[0]
        user_text = user_msg["content"][0]["text"]
        assert "[REDACTED:ssn]" in user_text
        assert "[REDACTED:email]" in user_text
        assert "123-45-6789" not in user_text
        assert "alice@example.com" not in user_text

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_redaction_count_tracks_across_invocations(self, model):
        """Redaction count should accumulate across multiple agent calls."""
        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
        )
        agent = Agent(model=model, plugins=[plugin])

        agent("SSN: 123-45-6789")
        first_count = plugin.redaction_count
        assert first_count >= 1

        agent("Another SSN: 987-65-4321")
        assert plugin.redaction_count > first_count

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_no_redaction_when_no_pii(self, model):
        """Messages without PII should pass through without redaction."""
        plugin = MessageRedactionPlugin(
            patterns={"ssn": r"\b\d{3}-\d{2}-\d{4}\b"},
        )
        agent = Agent(model=model, plugins=[plugin])

        agent("Hello, how are you?")

        # No SSN patterns in the input, so no redactions from user message
        # (the model's response on the second invocation might contain the
        # original text in conversation history, but the initial user message
        # should have zero redactions)
        user_msg = agent.messages[0]
        user_text = user_msg["content"][0]["text"]
        assert user_text == "Hello, how are you?"
