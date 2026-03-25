# strands-agents-extensions

Community extensions for the Strands Agents SDK — session managers and plugins in one package with optional extras.

## Installation

Install the base package:

```bash
pip install strands-agents-extensions
```

Install with specific extras:

```bash
# SQLite session manager
pip install strands-agents-extensions[sqlite]

# DynamoDB session manager
pip install strands-agents-extensions[dynamodb]

# All plugins
pip install strands-agents-extensions[plugins]

# All session managers
pip install strands-agents-extensions[session-managers]

# Everything
pip install strands-agents-extensions[all]
```

## Quick Start

### Session Managers

```python
from strands_agents_extensions.session_managers import SQLiteSessionManager

# Create a SQLite-backed session manager
session_manager = SQLiteSessionManager(db_path="sessions.db")

# Use with Strands Agent
agent = Agent(
    session_manager=session_manager,
    # ... other config
)
```

### Plugins

```python
from strands_agents_extensions.plugins import BudgetPlugin, RateLimiterPlugin

# Add budget control to your agent
budget_plugin = BudgetPlugin(max_budget_usd=10.0)

# Add rate limiting
rate_limiter = RateLimiterPlugin(max_requests_per_minute=60)

agent = Agent(
    plugins=[budget_plugin, rate_limiter],
    # ... other config
)
```

## Available Extras

| Extra | Description | Dependencies |
|-------|-------------|--------------|
| `sqlite` | SQLite session manager | None (stdlib) |
| `dynamodb` | DynamoDB session manager | boto3>=1.28 |
| `budget` | Budget control plugin | litellm<=1.82.6 |
| `rate-limiter` | Rate limiting plugin | None |
| `redaction` | PII redaction plugin | None |
| `fallback` | Model fallback plugin | None |
| `session-managers` | All session managers | sqlite + dynamodb |
| `plugins` | All plugins | budget + rate-limiter + redaction + fallback |
| `all` | Everything | session-managers + plugins |

## Available Components

### Session Managers

- **SQLiteSessionManager**: Local SQLite-backed session storage with full conversation history
- **DynamoDBSessionManager**: AWS DynamoDB-backed session storage for distributed systems

### Plugins

- **BudgetPlugin**: Track and enforce token budget limits
- **RateLimiterPlugin**: Control request rate to avoid API throttling
- **RedactionPlugin**: Automatically redact PII from messages
- **FallbackPlugin**: Implement model fallback strategies on failures

## License

Apache-2.0

## Contributing

Contributions welcome! This is a community-maintained package for extending the Strands Agents SDK.
