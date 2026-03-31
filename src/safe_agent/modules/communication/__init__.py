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

"""Communication modules package — email, messaging, calendar adapters."""

from safe_agent.modules.communication.calendar import (
    CalendarBackend,
    CalendarModule,
)
from safe_agent.modules.communication.email import (
    EmailBackend,
    EmailModule,
)
from safe_agent.modules.communication.messaging import (
    MessagingBackend,
    MessagingModule,
)

__all__ = [
    "CalendarBackend",
    "CalendarModule",
    "EmailBackend",
    "EmailModule",
    "MessagingBackend",
    "MessagingModule",
]
