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

"""Observability modules for SafeAgent.

This sub-package provides monitoring, logging, and observability capabilities:
- alerting: Alert management through pluggable backends
- audit: Read-only audit log query interface
- dashboard: Dashboard panel access for trend analysis
- error_tracking: Error tracking platform integration
- logging: Pluggable logging interface
- metrics: Metrics query interface
"""

from safe_agent.modules.observability.alerting import (
    AlertingBackend,
    AlertingModule,
)
from safe_agent.modules.observability.audit import (
    AuditModule,
)
from safe_agent.modules.observability.dashboard import (
    DashboardBackend,
    DashboardModule,
)
from safe_agent.modules.observability.error_tracking import (
    ErrorTrackingBackend,
    ErrorTrackingModule,
)
from safe_agent.modules.observability.logging import (
    LoggingBackend,
    LoggingModule,
)
from safe_agent.modules.observability.metrics import (
    MetricsBackend,
    MetricsModule,
)

__all__ = [
    "AlertingBackend",
    "AlertingModule",
    "AuditModule",
    "DashboardBackend",
    "DashboardModule",
    "ErrorTrackingBackend",
    "ErrorTrackingModule",
    "LoggingBackend",
    "LoggingModule",
    "MetricsBackend",
    "MetricsModule",
]
