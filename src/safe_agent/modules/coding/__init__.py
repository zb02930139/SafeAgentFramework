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

"""Coding modules for SafeAgent.

This sub-package contains modules for coding-related operations:
- GitModule: Version control operations
- ShellModule: Controlled subprocess execution
- SCMModule: Source Control Management (GitHub, GitLab)
"""

from safe_agent.modules.coding.git import GitModule
from safe_agent.modules.coding.scm import (
    Branch,
    Comment,
    GitHubSCM,
    GitLabSCM,
    Issue,
    PullRequest,
    Repository,
    SCMError,
    SCMModule,
    SCMProvider,
    SCMRegistry,
    User,
    Webhook,
)

__all__ = [
    "Branch",
    "Comment",
    "GitHubSCM",
    "GitLabSCM",
    "GitModule",
    "Issue",
    "PullRequest",
    "Repository",
    "SCMError",
    "SCMModule",
    "SCMProvider",
    "SCMRegistry",
    "ShellModule",
    "User",
    "Webhook",
]

# Import ShellModule after defining __all__ to avoid circular import issues
from safe_agent.modules.coding.shell import ShellModule
