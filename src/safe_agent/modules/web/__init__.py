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

"""Web modules sub-package for SafeAgent.

This sub-package contains web-related modules:
- api: Raw HTTP/API calls to external services
- browse: Web page fetching and summarization
- search: Pluggable web search interface
"""

from safe_agent.modules.web.api import WebApiModule
from safe_agent.modules.web.browse import (
    DEFAULT_MAX_RESPONSE_SIZE,
    DEFAULT_TIMEOUT,
    MAX_SUMMARY_TEXT_LENGTH,
    MAX_TIMEOUT,
    WebBrowseModule,
)
from safe_agent.modules.web.search import WebSearchBackend, WebSearchModule

__all__ = [
    "DEFAULT_MAX_RESPONSE_SIZE",
    "DEFAULT_TIMEOUT",
    "MAX_SUMMARY_TEXT_LENGTH",
    "MAX_TIMEOUT",
    "WebApiModule",
    "WebBrowseModule",
    "WebSearchBackend",
    "WebSearchModule",
]
