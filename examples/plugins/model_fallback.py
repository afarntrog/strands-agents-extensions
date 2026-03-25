"""Using ModelFallbackPlugin for automatic failover."""

from strands import Agent
from strands.models.bedrock import BedrockModel

from strands_agents_extensions.plugins import ModelFallbackPlugin


def main():
    """Demonstrate model fallback on failure."""
    primary = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")
    backup = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")

    plugin = ModelFallbackPlugin(
        fallback_models=[backup],
        retry_on=[Exception],
        cooldown_seconds=60.0,
    )

    agent = Agent(model=primary, plugins=[plugin])
    agent("Hello!")


if __name__ == "__main__":
    main()
