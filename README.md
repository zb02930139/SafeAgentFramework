# SafeAgentFramework

[![CI](https://github.com/zbrooks442/SafeAgentFramework/actions/workflows/ci.yml/badge.svg)](https://github.com/zbrooks442/SafeAgentFramework/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-85%25+-green.svg)](https://github.com/zbrooks442/SafeAgentFramework/actions)

A pluggable, security-first agent framework. Build AI agents that interact with
system resources (filesystem, shell, etc.) through modules gated by an
AWS IAM-style policy engine. The agent can attempt any tool call — your policies
decide in code whether to execute it.

## Current Status

**Active development — MVP complete.** SafeAgentFramework includes:

- **IAM Policy Engine** — JSON policies with Allow/Deny statements, condition operators, and AWS-style evaluation logic
- **Module Registry** — Pluggable modules with entry point discovery, tool descriptors, and dispatch
- **Code Gate** — Every LLM tool call passes through policy evaluation before execution
- **Filesystem Module** — Read/write operations with path-based policy controls
- **Shell Module** — Command execution with argument filtering and environment security
- **Session Management** — Per-session state with TTL eviction
- **Gateway & Agent Runtime** — High-level API for chat-driven agent workflows
- **Audit Logging** — Structured JSONL logs for every decision and action

**Stats:** 18 source files across 3 packages, 310 tests, 85%+ coverage.

## Installation

```bash
# Clone and install locally
git clone https://github.com/zbrooks442/SafeAgentFramework.git
cd SafeAgentFramework
pip install -e ".[dev]"
```

For development setup, see the Development section below.

## Quick Start

```python
from pathlib import Path
from safe_agent.core.agent import Agent
from safe_agent.core.llm import LLMClient
from safe_agent.modules.filesystem import FilesystemModule

# Create an agent with policies and modules
agent = Agent(
    policy_dir=Path("./policies"),
    llm_client=LLMClient(api_key="your-api-key"),
    modules=[FilesystemModule(root=Path("/projects"))]
)

# Chat with the agent (async)
response, session_id = await agent.chat("List the files in /projects/my-app")
print(response)

# Continue the conversation
response, session_id = await agent.chat("Read the README.md", session_id)
```

## How It Works

Three concepts:

1. **Modules** — pluggable Python classes that provide tools (filesystem, shell,
   etc.). Each module describes its tools and what IAM action/resource they
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

## Development

**Requires:** Python 3.13+

All tooling is configured in `pyproject.toml`:

- **Ruff** — linting and formatting
- **mypy** — strict type checking
- **pytest** — testing with coverage via pytest-cov
- **Bandit** — security scanning

```bash
# Clone and set up
git clone https://github.com/zbrooks442/SafeAgentFramework.git
cd SafeAgentFramework
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Run checks
ruff check src/ tests/
ruff format src/ tests/
mypy src/
pytest --cov
```

> **Note:** The core package is lightweight (~5MB). If you need RAG capabilities
> with vector storage, install with `pip install -e ".[rag,dev]"` instead. This
> adds `chromadb` and `tiktoken` (~500MB+ with dependencies).

## Roadmap

- **v0.1.0** — Core framework stable API
- **v0.2.0** — Git module with policy-controlled operations
- **v0.3.0** — Email module with sender/recipient controls
- **v1.0.0** — Stable API, full documentation, production-ready

## License

[MIT](LICENSE)
