# SafeAgentFramework

[![CI](https://github.com/zbrooks442/SafeAgentFramework/actions/workflows/ci.yml/badge.svg)](https://github.com/zbrooks442/SafeAgentFramework/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage](https://img.shields.io/badge/coverage-85%25+-green.svg)](https://github.com/zbrooks442/SafeAgentFramework/actions)

A pluggable, security-first agent framework. Build AI agents that interact with
system resources (filesystem, shell, databases, web APIs, etc.) through modules
gated by an access policy engine. The agent can attempt any tool call — your
policies decide in code whether to execute it.

## Current Status

**MVP complete (v0.1.0).** SafeAgentFramework is a security-first foundation
for AI agents. Use at your own risk — this is pre-1.0 software:

- **Access Policy Engine** — JSON policies with Allow/Deny statements, condition
  operators (StringEquals, StringLike, Bool, etc.), and deterministic evaluation
- **Module Registry** — Pluggable modules with entry point discovery, tool
  descriptors, and dispatch
- **Code Gate** — Every LLM tool call passes through policy evaluation before
  execution
- **Gateway & Agent Runtime** — High-level API for chat-driven agent workflows
- **Session Management** — Per-session state with TTL eviction and persistence
- **Audit Logging** — Comprehensive JSONL logs for every decision and action

**Stats:** 31 source files, 782 tests, 85%+ coverage.

## Built-in Modules

| Module | Actions | Description |
|--------|---------|-------------|
| **filesystem** | `ReadFile`, `WriteFile`, `ListDirectory`, `DeleteFile` | Read/write operations with path-based policy controls |
| **shell** | `ExecuteCommand`, `ReadOutput`, `SendInput` | Command execution with argument filtering and environment security |
| **web_api** | `Request` | HTTP/API calls to external services with SSRF protection |
| **web_browse** | `FetchPage`, `FetchAndExtract`, `Navigate` | Headless browser automation for web scraping |
| **remote_ssh** | `ExecuteCommand`, `TransferFile`, `ReadSshOutput` | SSH operations with host/port allowlisting |
| **audit** | `QueryLogs`, `GetMetrics`, `ExportLogs` | Audit log querying and metrics |
| **vault** | `GetSecret`, `ListSecrets` | Pluggable secret management (HashiCorp Vault, AWS Secrets Manager, etc.) |
| **calendar** | `CreateEvent`, `GetEvents`, `UpdateEvent`, `DeleteEvent` | Calendar integration (Google, Outlook, etc.) |
| **email** | `SendEmail`, `ReadEmails`, `ParseEmail` | Email sending and reading via pluggable backends |
| **database** | `Query`, `ExecuteStatement` | Database queries with parameterized SQL protection |
| **metrics** | `QueryMetrics` | Time-series metrics querying (Prometheus, Datadog, etc.) |
| **dashboard** | `GetPanel` | Dashboard panel retrieval (Grafana, etc.) |
| **alerting** | `ListAlerts`, `AcknowledgeAlert`, `EscalateAlert`, `SilenceAlert` | Alert management (PagerDuty, OpsGenie, etc.) |
| **error_tracking** | `QueryErrors`, `GetErrorDetails` | Error tracking integration (Sentry, Rollbar, etc.) |

## Quick Start

SafeAgent requires an `LLMClient` — any async class satisfying the `chat`
protocol that returns an `LLMResponse`. Here's a minimal example:

```python
import asyncio
from pathlib import Path

from safe_agent.core.agent import Agent
from safe_agent.core.llm import LLMClient, LLMResponse, ToolCall
from safe_agent.modules.filesystem import FilesystemModule


class MyLLMClient:
    """Minimal LLMClient implementation (satisfies the Protocol)."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        # Replace with your preferred LLM provider (OpenAI, Anthropic, etc.)
        ...


async def main() -> None:
    agent = Agent(
        policy_dir=Path("./policies"),
        llm_client=MyLLMClient(api_key="your-api-key"),
        modules=[FilesystemModule(root=Path("/projects"))],
    )

    # Start a conversation
    response, session_id = await agent.chat("List the files in /projects/my-app")
    print(response)

    # Continue the session
    response, session_id = await agent.chat(
        "Read the README.md",
        session_id=session_id
    )


asyncio.run(main())
```

## How It Works

SafeAgentFramework uses four core concepts:

1. **Modules** — Pluggable Python classes that provide tools. Each module
   describes its tools as `ToolDescriptor` objects mapping access actions to
   resources.

2. **Policies** — JSON documents that allow or deny actions on resources.
   **Deny-by-default.** Explicit deny always wins. Same evaluation logic as
   industry-standard policy engines (IAM-style).

3. **Code Gate** — Every tool call from the LLM passes through policy evaluation
   in application code. No prompt-based guardrails. The LLM never sees the
   policies.

4. **Dispatch** — The `ToolDispatcher` coordinates between LLM tool calls,
   policy evaluation, and module execution.

```
LLM tool call
    ↓
ToolDescriptor lookup (action + resource)
    ↓
Condition resolution (from tool arguments)
    ↓
Policy evaluation (allow/deny decision)
    ↓
    ├─ Denied → Return generic error (no policy leakage)
    └─ Allowed → module.execute() → Return result
```

See [`design/architecture.md`](design/architecture.md) for the full design
documentation.

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
      "Resource": ["*.env", "*.pem", "*.key", "*/secrets/*"]
    },
    {
      "Sid": "RestrictShellEnv",
      "Effect": "Deny",
      "Action": ["shell:ExecuteCommand"],
      "Resource": ["*"],
      "Condition": {
        "BoolEquals": {"shell:ClearEnvironment": false}
      }
    }
  ]
}
```

## Installation

```bash
# Clone and install locally
git clone https://github.com/zbrooks442/SafeAgentFramework.git
cd SafeAgentFramework
pip install -e ".[dev]"
```

For RAG capabilities with vector storage, use:

```bash
pip install -e ".[rag,dev]"  # Adds ChromaDB + tiktoken (~500MB+)
```

## Development

**Requires:** Python 3.13+

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

All tooling is configured in `pyproject.toml`:

- **Ruff** — linting and formatting
- **mypy** — strict type checking
- **pytest** — testing with coverage via pytest-cov
- **Bandit** — security scanning

## Roadmap

- **v0.1.x** — Core framework, stable API
- **v0.2.0** — Enhanced Git module with policy-controlled operations
- **v0.3.0** — RAG capabilities with vector search
- **v0.4.0** — Multi-agent orchestration
- **v1.0.0** — Production-ready release, complete documentation

## License

[Apache 2.0](LICENSE)
