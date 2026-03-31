# Copyright 2026 Zachary Brooks
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SafeAgent modules package — public re-exports."""

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)
from safe_agent.modules.coding.git import GitModule
from safe_agent.modules.coding.scm import (
    Branch,
    Comment,
    GitHubSCM,
    GitLabSCM,
    Issue,
    PullRequest,
    RateLimitError,
    Repository,
    SCMError,
    SCMModule,
    SCMProvider,
    SCMRegistry,
    User,
    Webhook,
)
from safe_agent.modules.coding.shell import ShellModule
from safe_agent.modules.communication import (
    CalendarBackend,
    CalendarModule,
    EmailBackend,
    EmailModule,
    MessagingBackend,
    MessagingModule,
)
from safe_agent.modules.filesystem.database import DatabaseBackend, DatabaseModule
from safe_agent.modules.observability import (
    AlertingBackend,
    AlertingModule,
    AuditModule,
    DashboardBackend,
    DashboardModule,
    ErrorTrackingBackend,
    ErrorTrackingModule,
    LoggingBackend,
    LoggingModule,
    MetricsBackend,
    MetricsModule,
)
from safe_agent.modules.registry import ModuleRegistry
from safe_agent.modules.security.remote_ssh import RemoteSSHModule, SSHCredential
from safe_agent.modules.security.vault import VaultBackend, VaultModule
from safe_agent.modules.web import (
    DEFAULT_MAX_RESPONSE_SIZE,
    DEFAULT_TIMEOUT,
    MAX_SUMMARY_TEXT_LENGTH,
    MAX_TIMEOUT,
    WebApiModule,
    WebBrowseModule,
    WebSearchBackend,
    WebSearchModule,
)

__all__ = [
    "DEFAULT_MAX_RESPONSE_SIZE",
    "DEFAULT_TIMEOUT",
    "MAX_SUMMARY_TEXT_LENGTH",
    "MAX_TIMEOUT",
    "AlertingBackend",
    "AlertingModule",
    "AuditModule",
    "BaseModule",
    "Branch",
    "CalendarBackend",
    "CalendarModule",
    "Comment",
    "DashboardBackend",
    "DashboardModule",
    "DatabaseBackend",
    "DatabaseModule",
    "EmailBackend",
    "EmailModule",
    "ErrorTrackingBackend",
    "ErrorTrackingModule",
    "GitHubSCM",
    "GitLabSCM",
    "GitModule",
    "Issue",
    "LoggingBackend",
    "LoggingModule",
    "MessagingBackend",
    "MessagingModule",
    "MetricsBackend",
    "MetricsModule",
    "ModuleDescriptor",
    "ModuleRegistry",
    "PullRequest",
    "RateLimitError",
    "RemoteSSHModule",
    "Repository",
    "SCMError",
    "SCMModule",
    "SCMProvider",
    "SCMRegistry",
    "SSHCredential",
    "ShellModule",
    "ToolDescriptor",
    "ToolResult",
    "User",
    "VaultBackend",
    "VaultModule",
    "WebApiModule",
    "WebBrowseModule",
    "WebSearchBackend",
    "WebSearchModule",
    "Webhook",
]
