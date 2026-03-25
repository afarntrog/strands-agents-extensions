"""Using BudgetPlugin and RateLimiterPlugin together."""

from strands import Agent

from strands_agents_extensions.plugins import (
    BudgetPlugin,
    OnExceedAction,
    OnLimitAction,
    RateLimiterPlugin,
)


def main():
    """Demonstrate budget and rate limiting plugins."""
    agent = Agent(
        plugins=[
            BudgetPlugin(
                max_cost_per_session=5.00,
                on_exceed=OnExceedAction.STOP,
            ),
            RateLimiterPlugin(
                model_calls_per_minute=60,
                on_limit=OnLimitAction.WAIT,
            ),
        ]
    )

    agent("Summarize the top 10 news stories today")
    print(f"Session cost so far: ${agent.plugins[0].session_cost:.4f}")


if __name__ == "__main__":
    main()
