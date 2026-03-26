# SafeAgentFramework — Design

A pluggable agent framework where **you** control exactly what the AI can do.

## Core Idea

The LLM is treated as untrusted input — like an HTTP request hitting a server.
It can attempt any tool call with any parameters. Your code decides whether to
execute it, based on policies you write. No prompt-based guardrails. No
self-governance. Code enforcement.

## Three Concepts

### 1. Modules

Modules are pluggable capabilities (filesystem, shell, email, etc.). Each module
is a Python class that:

- **Describes** its tools — name, parameters, and what IAM action/resource each
  tool maps to.
- **Resolves conditions** — derives context from tool parameters (e.g. file
  extension from a path) so policies can use them.
- **Executes** — performs the actual operation, only called after authorization.

```python
class BaseModule(ABC):

    @abstractmethod
    def describe(self) -> ModuleDescriptor:
        """Declare tools and their IAM metadata."""
        ...

    @abstractmethod
    async def resolve_conditions(self, tool_name: str, params: dict) -> dict[str, Any]:
        """Derive condition values from tool parameters."""
        ...

    @abstractmethod
    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        """Run the tool. Only called after authorization passes."""
        ...
```

A tool descriptor pairs the LLM-facing schema with the IAM-facing metadata.
All data models use Pydantic `BaseModel` for validation and serialization:

```python
class ToolDescriptor(BaseModel):
    name: str                        # Tool name exposed to the LLM
    description: str                 # Human-readable description
    parameters: dict                 # JSON Schema for parameters
    action: str                      # IAM action (e.g. "filesystem:ReadFile")
    resource_param: str | list[str]  # Which param(s) contain the IAM resource
    condition_keys: list[str]        # Condition keys this tool supports

class ModuleDescriptor(BaseModel):
    namespace: str                   # IAM namespace (e.g. "filesystem")
    description: str                 # Human-readable module description
    tools: list[ToolDescriptor]      # Tools provided by this module

class ToolResult(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Modules are discovered via Python entry points — install a package, restart,
and the new tools are available (subject to policies). No core runtime changes.

```toml
[project.entry-points."safe_agent.modules"]
filesystem = "my_package.filesystem:FilesystemModule"
```

### 2. Policies

AWS IAM-inspired, deny-by-default. A policy is a list of statements that
allow or deny specific actions on specific resources, optionally with
conditions.

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

**Evaluation rules** (same as AWS IAM):

1. Default: **deny**.
2. Collect statements whose Action and Resource match the request.
3. If any matching statement has `Effect: Deny` → **denied**. Explicit deny always wins.
4. If any matching statement has `Effect: Allow` (and conditions pass) → **allowed**.
5. Otherwise → **denied** (implicit deny, nothing matched).

The system fails closed. Missing or misconfigured policies result in denial.

**Conditions** add contextual constraints. Multiple conditions are ANDed;
multiple values within a condition are ORed.

```json
{
  "Condition": {
    "StringNotEquals": {
      "filesystem:FileExtension": [".env", ".pem"]
    }
  }
}
```

Supported operators: `StringEquals`, `StringNotEquals`, `StringLike`,
`StringNotLike`, `NumericEquals`, `NumericNotEquals`, `NumericLessThan`,
`NumericGreaterThan`, `NumericLessThanEquals`, `NumericGreaterThanEquals`.

Policies are loaded once at startup and frozen. No hot-reload, no runtime
modification. Changes require a restart.

### 3. The Code Gate

Every tool call passes through policy evaluation. There is exactly one path
from an LLM tool call to a module's `execute()`, and it goes through the
evaluator. No bypass, no override, no skip flag.

```
LLM requests tool call (untrusted)
        │
        ▼
Look up ToolDescriptor
        │
        ▼
Extract action + resource from descriptor
        │
        ▼
Module resolves conditions from params
        │
        ▼
Policy evaluator: allow or deny?
        │
   ┌────┴────┐
 DENY     ALLOW
   │         │
   ▼         ▼
Generic    module.execute()
error         │
              ▼
         Return result to LLM
```

What the LLM **never** sees: policy rules, denial reasons, which statement
matched, or whether denial was explicit or implicit. It gets tool definitions
(so it can construct calls) and a generic "Action denied" on failure. Policies
are invisible infrastructure.

## Design Principles

1. **Zero direct access.** The agent never touches system resources directly.
   Everything goes through a module.
2. **Deny by default.** No policy = no access.
3. **Code enforcement.** Authorization is a code gate, not a prompt instruction.
   The LLM and the enforcer are separate.
4. **Self-describing modules.** The runtime has no hardcoded knowledge of any
   module. Modules declare everything needed for tool registration and policy
   evaluation.
5. **Pluggable.** Add capabilities by installing a package. No runtime changes.

## Runtime Flow

```
User message → Gateway → Event Loop (per-session)
                              │
                              ├─ Assemble context
                              ├─ Call LLM
                              ├─ For each tool call:
                              │    ├─ Look up descriptor
                              │    ├─ Resolve conditions
                              │    ├─ Evaluate policy (CODE GATE)
                              │    ├─ If denied → generic error
                              │    └─ If allowed → execute, collect result
                              ├─ Append results to context
                              └─ Repeat until LLM returns text (or turn limit)
```

Sessions are isolated. Each session holds its own context and message history.
No cross-session state sharing.

## Security Model

- **Policies are not accessible to the agent.** The policy directory is outside
  every module's reachable namespace. There are no tools to list, read, or
  modify policies.
- **Policies are immutable after startup.** Loaded once, frozen, read-only.
- **Modules are discovered at startup only.** Mid-session package installs have
  no effect until restart.
- **Implementation-level sandboxing.** Modules enforce their own safety (e.g.
  filesystem path resolution prevents traversal) independently of policies.
  Sandboxing and authorization are separate layers.
- **Audit logging.** Every authorization decision is recorded (session, tool
  call, conditions, decision) to an append-only log outside the agent's reach.

## Adding a Module

A module is a self-contained capability. Adding one never requires changing
the core runtime. Here's the full process:

### 1. Implement the module class

Extend `BaseModule`. Set a namespace, describe your tools, resolve any
conditions policies might need, and implement the execution logic.

```python
class DatabaseModule(BaseModule):

    def describe(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            namespace="database",
            description="Query and write to databases",
            tools=[
                ToolDescriptor(
                    name="query",
                    description="Run a read-only SQL query",
                    parameters={
                        "type": "object",
                        "properties": {
                            "database": {"type": "string"},
                            "sql": {"type": "string"},
                        },
                        "required": ["database", "sql"],
                    },
                    action="database:Query",
                    resource_param="database",
                    condition_keys=["database:DatabaseName"],
                ),
            ],
        )

    async def resolve_conditions(self, tool_name: str, params: dict) -> dict:
        return {"database:DatabaseName": params.get("database", "")}

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        if tool_name == "query":
            # run the query against your backend
            return ToolResult(success=True, data={"rows": []})
        return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
```

What each method does:

- **`describe()`** — returns the module's namespace and tool definitions.
  The runtime uses this to register tools with the LLM and to know how to
  build authorization requests. The `resource_param` tells the runtime
  which parameter contains the IAM resource (here, `"database"` — so the
  value of the `database` parameter becomes the resource being authorized).
- **`resolve_conditions()`** — called before policy evaluation. Derives
  condition values from the tool parameters so policies can reference them.
  If a policy says `"database:DatabaseName": ["analytics"]`, this method
  is what provides the actual database name for comparison.
- **`execute()`** — the actual operation. Only called after the code gate
  allows it.

### 2. Write a policy

Without a policy, the module exists but the agent can't use it (deny by
default). Write a policy that grants exactly the access you want:

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowAnalyticsQueries",
      "Effect": "Allow",
      "Action": ["database:Query"],
      "Resource": ["analytics"]
    }
  ]
}
```

This allows `database:Query` only when the resource is `"analytics"`.
Any other database, or any action not listed, is implicitly denied.

### 3. Register via entry point

Add the module to your package's entry points so the runtime discovers it
at startup:

```toml
[project.entry-points."safe_agent.modules"]
database = "my_package.database:DatabaseModule"
```

Install the package, restart the runtime, and the agent has a new `query`
tool — gated by your policy. No core code was modified.

### Key constraints

- **Modules cannot access the policy system.** There is no import path from
  a module to the evaluator, policy store, or audit log. A module receives
  parameters and returns results — it has no way to check, modify, or
  influence its own authorization.
- **Modules own their sandboxing.** A filesystem module should resolve paths
  and prevent traversal. A shell module should enforce timeouts. This is
  independent of policies — sandboxing is a second layer of defense inside
  the module itself.
- **Namespace collisions are rejected.** If two modules declare the same
  namespace, the runtime refuses to start. An administrator resolves the
  conflict.

## What This Is Not

- **Not prompt-based enforcement.** We never tell the LLM "you can only read
  files in /projects." That's a suggestion, not a control.
- **Not pre-filtered tool lists.** The LLM sees all tools. Policies are
  resource-dependent — the same tool might be allowed for one path and denied
  for another. Filtering at the tool level can't express this.
- **Not MCP.** The module protocol borrows MCP's self-describing tools and JSON
  Schema parameters but adds authorization as a first-class concern. MCP has
  no policy evaluation — connected = trusted. We don't assume that.

## Pluggable Adapter Pattern

Many integration interfaces — email, messaging, monitoring, databases — have
multiple competing providers. Hardcoding a specific provider into the framework
would force unnecessary dependencies and limit adoption. Instead, the framework
uses a **pluggable adapter** pattern for these interfaces.

### How it works

The framework defines the **interface module**: namespace, tool descriptors,
IAM actions, and condition keys. This gives administrators a stable policy
surface — `email:SendEmail` means the same thing regardless of whether the
backend is SendGrid, Amazon SES, or a local SMTP relay.

A separate **adapter package** provides the concrete implementation by
supplying a backend that the interface module delegates to at execution time.

```
Framework ships:                Plugin provides:
┌──────────────────────┐       ┌───────────────────────┐
│  EmailModule         │       │  SendGridAdapter       │
│  ──────────────────  │       │  ───────────────────   │
│  describe()          │       │  Implements            │
│  resolve_conditions()│       │  EmailBackend protocol │
│  execute() ──────────┼──────►│  send(), read(), etc.  │
└──────────────────────┘       └───────────────────────┘
```

The module's `describe()` and `resolve_conditions()` are fixed by the
framework — they define the IAM surface that policies are written against.
Only `execute()` delegates to the injected backend. This means policies,
condition keys, and tool schemas are identical across all providers.

### Backend protocol

Each adapter interface defines a `typing.Protocol` for its backend. The
interface module accepts the backend at construction and delegates `execute()`
calls to it.

```python
class EmailBackend(Protocol):
    """Provider-specific email operations."""

    async def send(self, to: str, subject: str, body: str, **kwargs: Any) -> dict:
        ...

    async def read_inbox(self, folder: str, limit: int) -> list[dict]:
        ...


class EmailModule(BaseModule):
    """Framework-defined email interface. Backend injected at construction."""

    def __init__(self, backend: EmailBackend) -> None:
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        return ModuleDescriptor(namespace="email", ...)

    async def resolve_conditions(self, tool_name: str, params: dict) -> dict:
        return {"email:Recipient": params.get("to", "")}

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        if tool_name == "email:send_email":
            result = await self._backend.send(
                to=params["to"],
                subject=params["subject"],
                body=params["body"],
            )
            return ToolResult(success=True, data=result)
        ...
```

### When to use each model

| Model | When to use | Examples |
|-------|-------------|----------|
| **Built-in** | The operation is provider-agnostic or the framework must own the sandboxing | Filesystem, Shell, Remote SSH, Web Browse, Audit Trail |
| **Pluggable adapter** | Multiple providers exist and the choice is deployment-specific | Email, Messaging, Database, Monitoring, Vault |

### Registration

Adapter packages register via the same entry point mechanism. The adapter
package's entry point points to a factory that returns the interface module
initialized with the concrete backend:

```toml
# In the adapter package's pyproject.toml
[project.entry-points."safe_agent.modules"]
email = "my_email_adapter:create_email_module"
```

The core framework never imports or depends on any adapter package.

## Integration Interfaces

This section catalogs every integration interface the framework supports.
Each entry defines the IAM namespace, tools, actions, resource parameters,
and condition keys that policies can reference. Interfaces marked
**pluggable adapter** require a backend plugin; interfaces marked
**built-in** are implemented directly in the framework.

**Implementation status:** Filesystem and Shell are implemented. All other
interfaces are designed but not yet implemented.

### Communication

All communication interfaces use the pluggable adapter pattern. The
framework defines the IAM surface; a plugin provides the concrete provider
(SendGrid, Amazon SES, Slack, Microsoft Teams, Google Calendar, etc.).

#### `email` — Pluggable adapter

Send and receive emails, parse structured content from incoming messages.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `email:send_email` | `email:SendEmail` | `to` | Send an email to one or more recipients |
| `email:read_inbox` | `email:ReadInbox` | `folder` | Read messages from a mailbox folder |
| `email:parse_email` | `email:ParseEmail` | `message_id` | Extract structured data from an email body |

**Parameters:**

- `send_email`: `to` (string), `subject` (string), `body` (string),
  `cc` (string, optional), `attachments` (array, optional)
- `read_inbox`: `folder` (string, default `"inbox"`), `limit` (integer),
  `filter` (string, optional)
- `parse_email`: `message_id` (string), `extract_fields` (array, optional)

**Condition keys:** `email:Recipient`, `email:Sender`, `email:Subject`

#### `messaging` — Pluggable adapter

Send and read messages on team communication platforms.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `messaging:send_message` | `messaging:SendMessage` | `channel` | Send a message to a channel or user |
| `messaging:read_messages` | `messaging:ReadMessages` | `channel` | Read recent messages from a channel |

**Parameters:**

- `send_message`: `channel` (string), `text` (string),
  `thread_id` (string, optional)
- `read_messages`: `channel` (string), `limit` (integer),
  `since` (string, optional — ISO 8601 timestamp)

**Condition keys:** `messaging:Channel`, `messaging:Recipient`

#### `calendar` — Pluggable adapter

Manage maintenance windows, schedule events, and detect conflicts.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `calendar:create_event` | `calendar:CreateEvent` | `calendar_id` | Create a calendar event |
| `calendar:list_events` | `calendar:ListEvents` | `calendar_id` | List events in a time range |
| `calendar:check_conflicts` | `calendar:CheckConflicts` | `calendar_id` | Check for scheduling conflicts in a window |

**Parameters:**

- `create_event`: `calendar_id` (string), `title` (string),
  `start` (string — ISO 8601), `end` (string — ISO 8601),
  `description` (string, optional), `attendees` (array, optional)
- `list_events`: `calendar_id` (string), `start` (string — ISO 8601),
  `end` (string — ISO 8601)
- `check_conflicts`: `calendar_id` (string), `start` (string — ISO 8601),
  `end` (string — ISO 8601)

**Condition keys:** `calendar:CalendarId`

**Example policy — Communication:**

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowSendToOpsChannel",
      "Effect": "Allow",
      "Action": ["messaging:SendMessage"],
      "Resource": ["*"],
      "Condition": {
        "StringEquals": {
          "messaging:Channel": ["#network-ops", "#incidents"]
        }
      }
    },
    {
      "Sid": "AllowReadMaintenanceCalendar",
      "Effect": "Allow",
      "Action": ["calendar:ListEvents", "calendar:CheckConflicts"],
      "Resource": ["maintenance-windows"]
    },
    {
      "Sid": "DenyEmailToExternal",
      "Effect": "Deny",
      "Action": ["email:SendEmail"],
      "Resource": ["*"],
      "Condition": {
        "StringNotLike": {
          "email:Recipient": ["*@company.com"]
        }
      }
    }
  ]
}
```

---

### Web

#### `web_search` — Pluggable adapter

Search the web for documentation, CVEs, vendor advisories, and known issues.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `web_search:search` | `web_search:Search` | `query` | Execute a web search and return results |

**Parameters:**

- `search`: `query` (string), `max_results` (integer, optional),
  `domain_filter` (string, optional — restrict to a specific site)

**Condition keys:** `web_search:QueryDomain`

#### `web_browse` — Built-in

Fetch and interpret web page content. Supports tiered retrieval: raw HTML,
summarized text, or rendered content depending on the page and policy.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `web_browse:fetch_page` | `web_browse:FetchPage` | `url` | Fetch raw page content from a URL |
| `web_browse:summarize_page` | `web_browse:SummarizePage` | `url` | Fetch and summarize a page's content |

**Parameters:**

- `fetch_page`: `url` (string), `timeout` (integer, optional — seconds)
- `summarize_page`: `url` (string), `max_length` (integer, optional)

**Condition keys:** `web_browse:Domain`, `web_browse:ContentType`

#### `web_api` — Built-in

Make raw HTTP/API calls to external services. Primary use case is
interacting with REST and XML APIs on network appliances — switches,
routers, firewalls, load balancers. Cloud provider APIs (AWS, GCP, Azure)
would get their own dedicated adapter modules if needed.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `web_api:http_request` | `web_api:HttpRequest` | `url` | Make an HTTP request to an API endpoint |

**Parameters:**

- `http_request`: `url` (string), `method` (string — GET, POST, PUT,
  PATCH, DELETE), `headers` (object, optional), `body` (string, optional),
  `timeout` (integer, optional — seconds),
  `content_type` (string, optional — e.g. `application/json`,
  `application/xml`)

**Condition keys:** `web_api:Method`, `web_api:Domain`, `web_api:Path`

**Example policy — Web:**

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowBrowseVendorDocs",
      "Effect": "Allow",
      "Action": ["web_browse:FetchPage", "web_browse:SummarizePage"],
      "Resource": ["*"],
      "Condition": {
        "StringLike": {
          "web_browse:Domain": [
            "*.cisco.com", "*.arista.com", "*.juniper.net"
          ]
        }
      }
    },
    {
      "Sid": "AllowGetFromNetworkDevices",
      "Effect": "Allow",
      "Action": ["web_api:HttpRequest"],
      "Resource": ["https://10.0.0.0/8/*", "https://172.16.0.0/12/*"],
      "Condition": {
        "StringEquals": {
          "web_api:Method": ["GET"]
        }
      }
    },
    {
      "Sid": "DenyWriteToDevicesWithoutApproval",
      "Effect": "Deny",
      "Action": ["web_api:HttpRequest"],
      "Resource": ["*"],
      "Condition": {
        "StringEquals": {
          "web_api:Method": ["POST", "PUT", "PATCH", "DELETE"]
        }
      }
    }
  ]
}
```

---

### Monitoring

Most monitoring interfaces use the pluggable adapter pattern, since
monitoring stacks vary widely (Grafana, Datadog, Prometheus, Splunk,
PagerDuty, Sentry, etc.). The audit trail interface is built-in — it
queries the framework's own decision log.

#### `dashboard` — Pluggable adapter

Registry-based access to curated dashboard panels. Supports structured
queries and visual snapshots for trend analysis.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `dashboard:get_panel` | `dashboard:GetPanel` | `dashboard_id` | Retrieve a specific panel's data or snapshot |
| `dashboard:list_dashboards` | `dashboard:ListDashboards` | `"*"` | List available dashboards |

**Parameters:**

- `get_panel`: `dashboard_id` (string), `panel_id` (string),
  `time_range` (string, optional — e.g. `"7d"`, `"90d"`),
  `format` (string, optional — `"data"` or `"snapshot"`)
- `list_dashboards`: `filter` (string, optional), `tags` (array, optional)

**Condition keys:** `dashboard:DashboardId`

#### `metrics` — Pluggable adapter

Submit structured metric queries for short time range data.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `metrics:query_metrics` | `metrics:QueryMetrics` | `datasource` | Execute a metric query (PromQL, etc.) |

**Parameters:**

- `query_metrics`: `datasource` (string), `query` (string),
  `start` (string — ISO 8601), `end` (string — ISO 8601),
  `step` (string, optional — e.g. `"15s"`, `"1m"`)

**Condition keys:** `metrics:QueryLanguage`, `metrics:TimeRange`

#### `logging` — Pluggable adapter

Query and write to logging systems.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `logging:query_logs` | `logging:QueryLogs` | `source` | Search log entries |
| `logging:write_log` | `logging:WriteLog` | `source` | Write a log entry |

**Parameters:**

- `query_logs`: `source` (string), `query` (string),
  `start` (string — ISO 8601), `end` (string — ISO 8601),
  `limit` (integer, optional)
- `write_log`: `source` (string), `level` (string — e.g. `"info"`,
  `"warning"`, `"error"`), `message` (string),
  `metadata` (object, optional)

**Condition keys:** `logging:LogSource`, `logging:Severity`

#### `alerting` — Pluggable adapter

View, acknowledge, escalate, and silence alerts.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `alerting:list_alerts` | `alerting:ListAlerts` | `source` | List active alerts |
| `alerting:acknowledge_alert` | `alerting:AcknowledgeAlert` | `alert_id` | Acknowledge an alert |
| `alerting:escalate_alert` | `alerting:EscalateAlert` | `alert_id` | Escalate an alert to the next tier |
| `alerting:silence_alert` | `alerting:SilenceAlert` | `alert_id` | Silence an alert for a duration |

**Parameters:**

- `list_alerts`: `source` (string, optional — filter by source),
  `severity` (string, optional), `state` (string, optional — e.g.
  `"firing"`, `"acknowledged"`)
- `acknowledge_alert`: `alert_id` (string), `note` (string, optional)
- `escalate_alert`: `alert_id` (string), `target` (string, optional —
  team or individual), `note` (string, optional)
- `silence_alert`: `alert_id` (string),
  `duration` (string — e.g. `"1h"`, `"30m"`)

**Condition keys:** `alerting:Severity`, `alerting:AlertSource`

#### `error_tracking` — Pluggable adapter

Query application error tracking platforms.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `error_tracking:query_errors` | `error_tracking:QueryErrors` | `project` | Query recent errors for a project |

**Parameters:**

- `query_errors`: `project` (string), `query` (string, optional),
  `time_range` (string, optional — e.g. `"24h"`),
  `limit` (integer, optional)

**Condition keys:** `error_tracking:Project`, `error_tracking:ErrorType`

#### `audit` — Built-in

Query the framework's own audit trail — policy decisions and action
history. This interface reads the append-only audit log that the code gate
writes to. It is read-only by design.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `audit:query_audit_log` | `audit:QueryAuditLog` | `"*"` | Search the authorization decision log |

**Parameters:**

- `query_audit_log`: `start` (string — ISO 8601),
  `end` (string — ISO 8601), `session_id` (string, optional),
  `action_filter` (string, optional — glob pattern),
  `decision` (string, optional — `"allow"` or `"deny"`),
  `limit` (integer, optional)

**Condition keys:** `audit:TimeRange`, `audit:SessionId`

**Example policy — Monitoring:**

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowReadMetricsAndLogs",
      "Effect": "Allow",
      "Action": [
        "metrics:QueryMetrics",
        "logging:QueryLogs",
        "dashboard:GetPanel",
        "dashboard:ListDashboards"
      ],
      "Resource": ["*"]
    },
    {
      "Sid": "AllowAckAlerts",
      "Effect": "Allow",
      "Action": ["alerting:ListAlerts", "alerting:AcknowledgeAlert"],
      "Resource": ["*"]
    },
    {
      "Sid": "DenySilenceHighSeverity",
      "Effect": "Deny",
      "Action": ["alerting:SilenceAlert"],
      "Resource": ["*"],
      "Condition": {
        "StringEquals": {
          "alerting:Severity": ["critical", "high"]
        }
      }
    },
    {
      "Sid": "AllowAuditRead",
      "Effect": "Allow",
      "Action": ["audit:QueryAuditLog"],
      "Resource": ["*"]
    }
  ]
}
```

---

### Code & Compute

#### `shell` — Built-in (implemented)

Execute local commands and scripts. See the existing `ShellModule`
implementation for full details.

| Tool | Action | Resource param | Condition keys |
|------|--------|----------------|----------------|
| `shell:execute` | `shell:Execute` | `command` | `shell:CommandName`, `shell:WorkingDirectory` |

#### `remote_ssh` — Built-in

Connect to network devices and remote hosts, execute commands, and push
configuration changes. This module is built into the framework because SSH
session management, credential handling, and output sanitization require
tight sandboxing that the framework must own.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `remote_ssh:connect` | `remote_ssh:Connect` | `hostname` | Open an SSH session to a host |
| `remote_ssh:execute_command` | `remote_ssh:ExecuteCommand` | `hostname` | Run a command on a connected host |
| `remote_ssh:push_config` | `remote_ssh:PushConfig` | `hostname` | Push a configuration change to a device |

**Parameters:**

- `connect`: `hostname` (string), `username` (string),
  `port` (integer, optional — default 22)
- `execute_command`: `hostname` (string), `command` (string),
  `timeout` (integer, optional — seconds)
- `push_config`: `hostname` (string), `config` (string),
  `mode` (string, optional — e.g. `"merge"`, `"replace"`),
  `dry_run` (boolean, optional)

**Condition keys:** `remote_ssh:Hostname`, `remote_ssh:Username`,
`remote_ssh:CommandName`

**Example policy — Code & Compute:**

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowShowCommandsOnSwitches",
      "Effect": "Allow",
      "Action": ["remote_ssh:Connect", "remote_ssh:ExecuteCommand"],
      "Resource": ["10.0.1.*"],
      "Condition": {
        "StringLike": {
          "remote_ssh:CommandName": ["show *", "display *"]
        }
      }
    },
    {
      "Sid": "DenyConfigPushWithoutDryRun",
      "Effect": "Deny",
      "Action": ["remote_ssh:PushConfig"],
      "Resource": ["*"]
    }
  ]
}
```

---

### Data & Storage

#### `filesystem` — Built-in (implemented)

Read, write, list, delete, and move local files. See the existing
`FilesystemModule` implementation for full details.

| Tool | Action | Resource param | Condition keys |
|------|--------|----------------|----------------|
| `filesystem:read_file` | `filesystem:ReadFile` | `path` | `filesystem:FileExtension`, `filesystem:FileSize`, `filesystem:IsDirectory` |
| `filesystem:write_file` | `filesystem:WriteFile` | `path` | `filesystem:FileExtension` |
| `filesystem:list_directory` | `filesystem:ListDirectory` | `path` | `filesystem:IsDirectory` |
| `filesystem:delete_file` | `filesystem:DeleteFile` | `path` | `filesystem:FileExtension` |
| `filesystem:move_file` | `filesystem:MoveFile` | `source`, `destination` | `filesystem:FileExtension` |

#### `database` — Pluggable adapter

Query SQL and NoSQL databases. The adapter maps to a specific database
engine (PostgreSQL, MySQL, MongoDB, etc.) or an abstraction layer
like a CMDB.

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `database:query` | `database:Query` | `database` | Run a read-only query |
| `database:execute_statement` | `database:ExecuteStatement` | `database` | Run a write statement (INSERT, UPDATE, DELETE) |

**Parameters:**

- `query`: `database` (string), `sql` (string),
  `limit` (integer, optional)
- `execute_statement`: `database` (string), `sql` (string)

**Condition keys:** `database:DatabaseName`, `database:TableName`

**Example policy — Data & Storage:**

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowReadCMDB",
      "Effect": "Allow",
      "Action": ["database:Query"],
      "Resource": ["cmdb"]
    },
    {
      "Sid": "DenyAllWrites",
      "Effect": "Deny",
      "Action": ["database:ExecuteStatement"],
      "Resource": ["*"]
    }
  ]
}
```

---

### Identity & Secrets

#### `vault` — Pluggable adapter

Retrieve and rotate credentials, API keys, and other secrets. The adapter
maps to a specific secrets manager (HashiCorp Vault, AWS Secrets Manager,
Azure Key Vault, etc.).

| Tool | Action | Resource param | Description |
|------|--------|----------------|-------------|
| `vault:get_secret` | `vault:GetSecret` | `path` | Retrieve a secret by path |
| `vault:rotate_secret` | `vault:RotateSecret` | `path` | Trigger rotation of a secret |

**Parameters:**

- `get_secret`: `path` (string), `version` (integer, optional)
- `rotate_secret`: `path` (string)

**Condition keys:** `vault:SecretPath`, `vault:SecretEngine`

**Example policy — Identity & Secrets:**

```json
{
  "Version": "2025-01",
  "Statement": [
    {
      "Sid": "AllowReadNetworkCredentials",
      "Effect": "Allow",
      "Action": ["vault:GetSecret"],
      "Resource": ["network/devices/*"]
    },
    {
      "Sid": "DenyRotation",
      "Effect": "Deny",
      "Action": ["vault:RotateSecret"],
      "Resource": ["*"]
    }
  ]
}
```
