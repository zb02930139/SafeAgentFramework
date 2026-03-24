# SafeAgentFramework

A pluggable, security-first agent framework. Build AI agents that interact with
system resources (filesystem, shell, email, etc.) through modules gated by an
AWS IAM-style policy engine. The agent can attempt any tool call — your policies
decide in code whether to execute it.

## How It Works

Three concepts:

1. **Modules** — pluggable Python classes that provide tools (filesystem, shell,
   git, etc.). Each module describes its tools and what IAM action/resource they
   map to.
2. **Policies** — JSON documents that allow or deny actions on resources.
   Deny-by-default. Explicit deny always wins. Same evaluation logic as AWS IAM.
3. **Code gate** — every tool call from the LLM passes through policy evaluation
   in application code. No prompt-based guardrails. The LLM never sees the
   policies.

```
LLM tool call → look up descriptor → resolve conditions → policy evaluation → allow/deny
                                                                                  │
                                                                          denied: generic error
                                                                          allowed: module.execute()
```

See [`design/architecture.md`](design/architecture.md) for the full design.

## Policy Example

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowReadProjectFiles",
      "Effect": "Allow",
      "Action": ["filesystem:ReadFile", "filesystem:ListDirectory"],
      "Resource": ["/projects/*"]
    },
    {
      "Sid": "BlockSecrets",
      "Effect": "Deny",
      "Action": ["filesystem:*"],
      "Resource": ["*.env", "*.pem", "*.key"]
    }
  ]
}
```

## Current Status

Early stage — the design is documented in [`design/architecture.md`](design/architecture.md)
and the project skeleton exists, but implementation has not started yet. The
codebase is a clean slate ready for development.

## Quick Start

```bash
git clone https://github.com/zbrooks422/safeagentframework.git
cd SafeAgentFramework

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pre-commit install

pytest
```

> **Note:** The core package is lightweight (~5MB). If you need RAG capabilities
> with vector storage, install with `pip install -e ".[rag,dev]"` instead. This
> adds `chromadb` and `tiktoken` (~500MB+ with dependencies).

## Development

**Requires:** Python 3.13+

All tooling is configured in `pyproject.toml`:

- **Ruff** — linting and formatting
- **mypy** — strict type checking
- **pytest** — testing with coverage via pytest-cov
- **Bandit** — security scanning

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
pytest --cov
```

## License

[MIT](LICENSE)
