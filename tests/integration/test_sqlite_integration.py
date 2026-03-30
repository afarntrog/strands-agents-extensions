"""Integration tests for SQLiteSessionManager with a real Bedrock LLM."""

from strands import Agent

from strands_agents_extensions.session_managers import SQLiteSessionManager

from .conftest import retry_on_flaky


class TestSQLiteSessionPersistence:
    """Verify session persistence through invocations."""

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_agent_with_session_manager_completes(self, model):
        """An agent with SQLiteSessionManager should complete a call."""
        manager = SQLiteSessionManager(session_id="test-session", db_path=":memory:")
        agent = Agent(model=model, session_manager=manager)

        result = agent("Say hello in one word.")
        assert result is not None

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_messages_persist_across_invocations(self, model):
        """Messages from multiple invocations should be persisted in the session."""
        manager = SQLiteSessionManager(session_id="test-persist", db_path=":memory:")
        agent = Agent(model=model, session_manager=manager)

        agent("Say hi.")
        agent("Say bye.")

        # The agent's messages should contain both conversation turns
        assert len(agent.messages) >= 4  # 2 user + 2 assistant messages

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_session_survives_new_agent_instance(self, model, tmp_path):
        """A new agent with the same session_id and db_path should see prior messages."""
        db_path = str(tmp_path / "test.db")

        manager1 = SQLiteSessionManager(session_id="resume-test", db_path=db_path)
        agent1 = Agent(model=model, session_manager=manager1)
        agent1("Hello!")

        message_count_after_first = len(agent1.messages)
        assert message_count_after_first >= 2  # At least user + assistant

        # Create a new agent with the same session
        manager2 = SQLiteSessionManager(session_id="resume-test", db_path=db_path)
        agent2 = Agent(model=model, session_manager=manager2)

        # The new agent should have loaded the prior conversation
        assert len(agent2.messages) == message_count_after_first

    @retry_on_flaky("LLM responses may vary or throttle")
    def test_different_sessions_are_isolated(self, model, tmp_path):
        """Different session IDs should have independent message histories."""
        db_path = str(tmp_path / "test.db")

        manager_a = SQLiteSessionManager(session_id="session-a", db_path=db_path)
        agent_a = Agent(model=model, session_manager=manager_a)
        agent_a("Hello from session A.")

        manager_b = SQLiteSessionManager(session_id="session-b", db_path=db_path)
        agent_b = Agent(model=model, session_manager=manager_b)

        # Session B should start empty
        assert len(agent_b.messages) == 0
