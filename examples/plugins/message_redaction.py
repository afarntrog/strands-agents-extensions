"""Using MessageRedactionPlugin to scrub sensitive data."""

from strands import Agent

from strands_agents_extensions.plugins import MessageRedactionPlugin


def main():
    """Demonstrate message redaction."""
    plugin = MessageRedactionPlugin(
        patterns={
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
        },
        replacement="[REDACTED:{name}]",
    )

    agent = Agent(plugins=[plugin])
    agent("My SSN is 123-45-6789 and my email is alice@example.com")


if __name__ == "__main__":
    main()
