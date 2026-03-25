"""Basic usage of SQLiteSessionManager with a Strands agent."""

from strands import Agent

from strands_agents_extensions.session_managers import SQLiteSessionManager


def main():
    """Demonstrate basic chat with SQLite session persistence."""
    session_manager = SQLiteSessionManager(
        session_id="chat-session-001",
        db_path="./sessions.db",
    )

    agent = Agent(
        session_manager=session_manager,
        system_prompt="You are a helpful assistant.",
    )

    response = agent("Hello! My name is Alice.")
    print(f"Agent: {response}")

    # Later, resume the same session
    response = agent("What's my name?")
    print(f"Agent: {response}")


if __name__ == "__main__":
    main()
