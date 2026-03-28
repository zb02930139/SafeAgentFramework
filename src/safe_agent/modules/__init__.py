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

from safe_agent.modules.alerting import AlertingBackend, AlertingModule
from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)
from safe_agent.modules.database import DatabaseBackend, DatabaseModule
from safe_agent.modules.error_tracking import (
    ErrorTrackingBackend,
    ErrorTrackingModule,
)
from safe_agent.modules.git import GitModule
from safe_agent.modules.messaging import MessagingBackend, MessagingModule
from safe_agent.modules.registry import ModuleRegistry
from safe_agent.modules.vault import VaultBackend, VaultModule

__all__ = [
    "AlertingBackend",
    "AlertingModule",
    "BaseModule",
    "DatabaseBackend",
    "DatabaseModule",
    "ErrorTrackingBackend",
    "ErrorTrackingModule",
    "GitModule",
    "MessagingBackend",
    "MessagingModule",
    "ModuleDescriptor",
    "ModuleRegistry",
    "ToolDescriptor",
    "ToolResult",
    "VaultBackend",
    "VaultModule",
]
