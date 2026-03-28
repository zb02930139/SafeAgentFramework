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

"""Source Control Management (SCM) module for SafeAgent.

Provides a pluggable SCM provider abstraction for GitHub and GitLab
operations with a unified interface.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class User(BaseModel):
    """Represents a user in an SCM system."""

    id: int | str
    username: str
    name: str | None = None
    email: str | None = None
    avatar_url: str | None = None
    html_url: str | None = None


class Repository(BaseModel):
    """Represents a repository in an SCM system."""

    id: int | str
    name: str
    full_name: str
    owner: str
    description: str | None = None
    html_url: str
    clone_url: str | None = None
    ssh_url: str | None = None
    default_branch: str | None = None
    private: bool = False
    fork: bool = False
    parent: str | None = None
    created_at: str
    updated_at: str | None = None


class Branch(BaseModel):
    """Represents a branch in a repository."""

    name: str
    commit_sha: str
    protected: bool = False
    default: bool = False


class PullRequest(BaseModel):
    """Represents a pull/merge request."""

    id: int | str
    number: int
    title: str
    body: str | None = None
    state: str
    head: str
    head_repo: str | None = None
    base: str
    base_repo: str
    author: User | None = None
    assignees: list[User] = Field(default_factory=list)
    reviewers: list[User] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    draft: bool = False
    mergeable: bool | None = None
    merged: bool = False
    merged_at: str | None = None
    html_url: str
    created_at: str
    updated_at: str | None = None


class Issue(BaseModel):
    """Represents an issue in a repository."""

    id: int | str
    number: int
    title: str
    body: str | None = None
    state: str
    author: User | None = None
    assignees: list[User] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    milestone: str | None = None
    html_url: str
    created_at: str
    updated_at: str | None = None
    closed_at: str | None = None


class Comment(BaseModel):
    """Represents a comment on an issue or pull request."""

    id: int | str
    body: str
    author: User | None = None
    html_url: str
    created_at: str
    updated_at: str | None = None


class Webhook(BaseModel):
    """Represents a webhook configuration."""

    id: int | str
    url: str
    events: list[str] = Field(default_factory=list)
    active: bool = True
    secret: str | None = None
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class SCMError(Exception):
    """Base exception for SCM-related errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        """Initialize SCMError."""
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
        """Return string representation."""
        parts = [f"[{self.provider}] {self.message}"]
        if self.status_code:
            parts.append(f" (status={self.status_code})")
        return "".join(parts)


class RateLimitError(SCMError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        provider: str,
        reset_at: int | None = None,
        remaining: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize RateLimitError."""
        super().__init__(message="Rate limit exceeded", provider=provider, **kwargs)
        self.reset_at = reset_at
        self.remaining = remaining


# ---------------------------------------------------------------------------
# Abstract Base Provider
# ---------------------------------------------------------------------------


class SCMProvider(ABC):
    """Abstract base class for SCM provider implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'github', 'gitlab')."""

    @abstractmethod
    async def list_repos(self, owner: str, **filters: Any) -> list[Repository]:
        """List repositories for an owner/user/organization."""

    @abstractmethod
    async def get_repo(self, owner: str, name: str) -> Repository:
        """Get a specific repository."""

    @abstractmethod
    async def create_repo(self, name: str, **opts: Any) -> Repository:
        """Create a new repository."""

    @abstractmethod
    async def create_fork(self, owner: str, repo: str, **opts: Any) -> Repository:
        """Fork a repository."""

    @abstractmethod
    async def list_branches(
        self, owner: str, repo: str, **filters: Any
    ) -> list[Branch]:
        """List branches in a repository."""

    @abstractmethod
    async def get_branch(self, owner: str, repo: str, branch: str) -> Branch:
        """Get a specific branch."""

    @abstractmethod
    async def create_pull_request(
        self, owner: str, repo: str, title: str, head: str, base: str, **opts: Any
    ) -> PullRequest:
        """Create a pull request."""

    @abstractmethod
    async def get_pull_request(self, owner: str, repo: str, number: int) -> PullRequest:
        """Get a specific pull request."""

    @abstractmethod
    async def list_pull_requests(
        self, owner: str, repo: str, **filters: Any
    ) -> list[PullRequest]:
        """List pull requests in a repository."""

    @abstractmethod
    async def approve_pull_request(
        self, owner: str, repo: str, number: int, **opts: Any
    ) -> PullRequest:
        """Approve a pull request."""

    @abstractmethod
    async def comment_on_pull_request(
        self, owner: str, repo: str, number: int, body: str
    ) -> Comment:
        """Add a comment to a pull request."""

    @abstractmethod
    async def create_issue(
        self, owner: str, repo: str, title: str, **opts: Any
    ) -> Issue:
        """Create an issue."""

    @abstractmethod
    async def list_issues(self, owner: str, repo: str, **filters: Any) -> list[Issue]:
        """List issues in a repository."""

    @abstractmethod
    async def create_webhook(
        self, owner: str, repo: str, url: str, events: list[str], **opts: Any
    ) -> Webhook:
        """Create a webhook."""

    @abstractmethod
    async def list_webhooks(self, owner: str, repo: str) -> list[Webhook]:
        """List webhooks for a repository."""

    @abstractmethod
    async def close(self) -> None:
        """Close the provider and release resources."""


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


async def _backoff_with_delay(
    attempt: int, base_delay: float, max_delay: float
) -> None:
    """Calculate and sleep for exponential backoff with jitter."""
    delay = min(base_delay * (2**attempt), max_delay) * (0.5 + random.random() * 0.5)  # noqa: S311
    await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# GitHub Implementation
# ---------------------------------------------------------------------------


class GitHubSCM(SCMProvider):
    """GitHub SCM provider implementation using the REST API v3."""

    DEFAULT_API_URL = "https://api.github.com"

    def __init__(
        self,
        token: str,
        api_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ) -> None:
        """Initialize GitHub SCM provider."""
        self._token = token
        self._api_url = (api_url or self.DEFAULT_API_URL).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "github"

    def __repr__(self) -> str:
        """Return string representation with masked token."""
        masked_token = (
            f"{self._token[:4]}...{self._token[-4:]}" if len(self._token) > 8 else "***"
        )
        return f"GitHubSCM(api_url={self._api_url!r}, token={masked_token!r})"

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._api_url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any] | list[Any]:
        """Make an HTTP request with retry logic."""
        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await client.request(method, path, **kwargs)

                if response.status_code >= 500:
                    last_error = SCMError(
                        message=f"Server error: {response.status_code}",
                        provider=self.name,
                        status_code=response.status_code,
                        response_body=response.text,
                    )
                    if attempt < self._max_retries:
                        await _backoff_with_delay(
                            attempt, self._base_delay, self._max_delay
                        )
                        continue
                    raise last_error

                # Check rate limit BEFORE other error handling
                remaining = response.headers.get("x-ratelimit-remaining")
                if response.status_code == 429 and remaining == "0":
                    reset_at = int(response.headers.get("x-ratelimit-reset", 0))
                    raise RateLimitError(
                        provider=self.name,
                        reset_at=reset_at,
                        remaining=0,
                        status_code=429,
                    )

                if response.status_code >= 400:
                    error_data = response.json() if response.content else {}
                    raise SCMError(
                        message=error_data.get(
                            "message", f"HTTP {response.status_code}"
                        ),
                        provider=self.name,
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                return response.json() if response.content else {}

            except httpx.TimeoutException as exc:
                last_error = SCMError(
                    message=f"Request timeout: {exc}", provider=self.name
                )
                if attempt < self._max_retries:
                    await _backoff_with_delay(
                        attempt, self._base_delay, self._max_delay
                    )
                    continue

            except RateLimitError:
                raise

            except httpx.HTTPError as exc:
                last_error = SCMError(message=f"HTTP error: {exc}", provider=self.name)
                if attempt < self._max_retries:
                    await _backoff_with_delay(
                        attempt, self._base_delay, self._max_delay
                    )
                    continue

        raise last_error or SCMError(
            message="Unknown error after retries", provider=self.name
        )

    def _parse_user(self, data: dict[str, Any]) -> User:
        """Parse GitHub user data."""
        return User(
            id=data.get("id", 0),
            username=data.get("login", ""),
            name=data.get("name"),
            email=data.get("email"),
            avatar_url=data.get("avatar_url"),
            html_url=data.get("html_url"),
        )

    def _parse_repository(self, data: dict[str, Any]) -> Repository:
        """Parse GitHub repository data."""
        owner_data = data.get("owner", {})
        return Repository(
            id=data.get("id", 0),
            name=data.get("name", ""),
            full_name=data.get("full_name", ""),
            owner=owner_data.get("login", "") if owner_data else "",
            description=data.get("description"),
            html_url=data.get("html_url", ""),
            clone_url=data.get("clone_url"),
            ssh_url=data.get("ssh_url"),
            default_branch=data.get("default_branch"),
            private=data.get("private", False),
            fork=data.get("fork", False),
            parent=(
                data.get("parent", {}).get("full_name") if data.get("parent") else None
            ),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
        )

    def _parse_branch(self, data: dict[str, Any]) -> Branch:
        """Parse GitHub branch data."""
        commit_data = data.get("commit", {})
        return Branch(
            name=data.get("name", ""),
            commit_sha=commit_data.get("sha", "") if commit_data else "",
            protected=data.get("protected", False),
        )

    def _parse_pull_request(self, data: dict[str, Any]) -> PullRequest:
        """Parse GitHub pull request data."""
        head_data = data.get("head", {})
        base_data = data.get("base", {})
        user_data = data.get("user")

        return PullRequest(
            id=data.get("id", 0),
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=data.get("body"),
            state=data.get("state", ""),
            head=head_data.get("ref", "") if head_data else "",
            head_repo=(
                head_data.get("repo", {}).get("full_name")
                if head_data.get("repo")
                else None
            ),
            base=base_data.get("ref", "") if base_data else "",
            base_repo=(
                base_data.get("repo", {}).get("full_name")
                if base_data.get("repo")
                else ""
            ),
            author=self._parse_user(user_data) if user_data else None,
            draft=data.get("draft", False),
            mergeable=data.get("mergeable"),
            merged=data.get("merged", False),
            merged_at=data.get("merged_at"),
            html_url=data.get("html_url", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
        )

    def _parse_issue(self, data: dict[str, Any]) -> Issue:
        """Parse GitHub issue data."""
        user_data = data.get("user")

        return Issue(
            id=data.get("id", 0),
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=data.get("body"),
            state=data.get("state", ""),
            author=self._parse_user(user_data) if user_data else None,
            labels=[label.get("name", "") for label in data.get("labels", [])],
            milestone=(
                data.get("milestone", {}).get("title")
                if data.get("milestone")
                else None
            ),
            html_url=data.get("html_url", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
            closed_at=data.get("closed_at"),
        )

    def _parse_comment(self, data: dict[str, Any]) -> Comment:
        """Parse GitHub comment data."""
        user_data = data.get("user")

        return Comment(
            id=data.get("id", 0),
            body=data.get("body", ""),
            author=self._parse_user(user_data) if user_data else None,
            html_url=data.get("html_url", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
        )

    def _parse_webhook(self, data: dict[str, Any]) -> Webhook:
        """Parse GitHub webhook data."""
        return Webhook(
            id=data.get("id", 0),
            url=data.get("config", {}).get("url", ""),
            events=data.get("events", []),
            active=data.get("active", True),
            created_at=data.get("created_at"),
        )

    async def list_repos(self, owner: str, **filters: Any) -> list[Repository]:
        """List repositories for an owner."""
        params: dict[str, Any] = {"per_page": 100}
        params.update(filters)
        data = await self._request("GET", f"/users/{owner}/repos", params=params)
        return [self._parse_repository(repo) for repo in data]

    async def get_repo(self, owner: str, name: str) -> Repository:
        """Get a specific repository."""
        data = await self._request("GET", f"/repos/{owner}/{name}")
        return self._parse_repository(data)

    async def create_repo(self, name: str, **opts: Any) -> Repository:
        """Create a new repository."""
        body: dict[str, Any] = {"name": name}
        body.update(opts)
        data = await self._request("POST", "/user/repos", json=body)
        return self._parse_repository(data)

    async def create_fork(self, owner: str, repo: str, **opts: Any) -> Repository:
        """Fork a repository."""
        data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/forks",
            json=opts if opts else None,
        )
        return self._parse_repository(data)

    async def list_branches(
        self, owner: str, repo: str, **filters: Any
    ) -> list[Branch]:
        """List branches in a repository."""
        params: dict[str, Any] = {"per_page": 100}
        params.update(filters)
        data = await self._request(
            "GET", f"/repos/{owner}/{repo}/branches", params=params
        )
        return [self._parse_branch(branch) for branch in data]

    async def get_branch(self, owner: str, repo: str, branch: str) -> Branch:
        """Get a specific branch."""
        data = await self._request("GET", f"/repos/{owner}/{repo}/branches/{branch}")
        return self._parse_branch(data)

    async def create_pull_request(
        self, owner: str, repo: str, title: str, head: str, base: str, **opts: Any
    ) -> PullRequest:
        """Create a pull request."""
        body: dict[str, Any] = {"title": title, "head": head, "base": base}
        body.update(opts)
        data = await self._request("POST", f"/repos/{owner}/{repo}/pulls", json=body)
        return self._parse_pull_request(data)

    async def get_pull_request(self, owner: str, repo: str, number: int) -> PullRequest:
        """Get a specific pull request."""
        data = await self._request("GET", f"/repos/{owner}/{repo}/pulls/{number}")
        return self._parse_pull_request(data)

    async def list_pull_requests(
        self, owner: str, repo: str, **filters: Any
    ) -> list[PullRequest]:
        """List pull requests in a repository."""
        params: dict[str, Any] = {"per_page": 100}
        params.update(filters)
        data = await self._request("GET", f"/repos/{owner}/{repo}/pulls", params=params)
        return [self._parse_pull_request(pr) for pr in data]

    async def approve_pull_request(
        self, owner: str, repo: str, number: int, **opts: Any
    ) -> PullRequest:
        """Approve a pull request."""
        body: dict[str, Any] = {"event": "APPROVE"}
        body.update(opts)
        await self._request(
            "POST", f"/repos/{owner}/{repo}/pulls/{number}/reviews", json=body
        )
        return await self.get_pull_request(owner, repo, number)

    async def comment_on_pull_request(
        self, owner: str, repo: str, number: int, body: str
    ) -> Comment:
        """Add a comment to a pull request."""
        data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{number}/comments",
            json={"body": body},
        )
        return self._parse_comment(data)

    async def create_issue(
        self, owner: str, repo: str, title: str, **opts: Any
    ) -> Issue:
        """Create an issue."""
        body: dict[str, Any] = {"title": title}
        body.update(opts)
        data = await self._request("POST", f"/repos/{owner}/{repo}/issues", json=body)
        return self._parse_issue(data)

    async def list_issues(self, owner: str, repo: str, **filters: Any) -> list[Issue]:
        """List issues in a repository."""
        params: dict[str, Any] = {"per_page": 100}
        params.update(filters)
        data = await self._request(
            "GET", f"/repos/{owner}/{repo}/issues", params=params
        )
        return [self._parse_issue(issue) for issue in data]

    async def create_webhook(
        self, owner: str, repo: str, url: str, events: list[str], **opts: Any
    ) -> Webhook:
        """Create a webhook."""
        config: dict[str, Any] = {"url": url}
        if "secret" in opts:
            config["secret"] = opts.pop("secret")
        body: dict[str, Any] = {"config": config, "events": events}
        body.update(opts)
        data = await self._request("POST", f"/repos/{owner}/{repo}/hooks", json=body)
        return self._parse_webhook(data)

    async def list_webhooks(self, owner: str, repo: str) -> list[Webhook]:
        """List webhooks for a repository."""
        data = await self._request("GET", f"/repos/{owner}/{repo}/hooks")
        return [self._parse_webhook(hook) for hook in data]


# ---------------------------------------------------------------------------
# GitLab Implementation
# ---------------------------------------------------------------------------


class GitLabSCM(SCMProvider):
    """GitLab SCM provider implementation using API v4."""

    DEFAULT_API_URL = "https://gitlab.com/api/v4"

    def __init__(
        self,
        token: str,
        api_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
    ) -> None:
        """Initialize GitLab SCM provider."""
        self._token = token
        self._api_url = (api_url or self.DEFAULT_API_URL).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "gitlab"

    def __repr__(self) -> str:
        """Return string representation with masked token."""
        masked_token = (
            f"{self._token[:4]}...{self._token[-4:]}" if len(self._token) > 8 else "***"
        )
        return f"GitLabSCM(api_url={self._api_url!r}, token={masked_token!r})"

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._api_url,
                headers={"PRIVATE-TOKEN": self._token},
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any] | list[Any]:
        """Make an HTTP request with retry logic."""
        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await client.request(method, path, **kwargs)

                if response.status_code >= 500:
                    last_error = SCMError(
                        message=f"Server error: {response.status_code}",
                        provider=self.name,
                        status_code=response.status_code,
                        response_body=response.text,
                    )
                    if attempt < self._max_retries:
                        await _backoff_with_delay(
                            attempt, self._base_delay, self._max_delay
                        )
                        continue
                    raise last_error

                # Check rate limit BEFORE other error handling
                remaining = response.headers.get("ratelimit-remaining")
                if response.status_code == 429 and remaining == "0":
                    reset_at = int(response.headers.get("ratelimit-reset", 0))
                    raise RateLimitError(
                        provider=self.name,
                        reset_at=reset_at,
                        remaining=0,
                        status_code=429,
                    )

                if response.status_code >= 400:
                    error_data = response.json() if response.content else {}
                    message = error_data.get(
                        "message",
                        error_data.get("error", f"HTTP {response.status_code}"),
                    )
                    raise SCMError(
                        message=str(message),
                        provider=self.name,
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                return response.json() if response.content else {}

            except httpx.TimeoutException as exc:
                last_error = SCMError(
                    message=f"Request timeout: {exc}", provider=self.name
                )
                if attempt < self._max_retries:
                    await _backoff_with_delay(
                        attempt, self._base_delay, self._max_delay
                    )
                    continue

            except RateLimitError:
                raise

            except httpx.HTTPError as exc:
                last_error = SCMError(message=f"HTTP error: {exc}", provider=self.name)
                if attempt < self._max_retries:
                    await _backoff_with_delay(
                        attempt, self._base_delay, self._max_delay
                    )
                    continue

        raise last_error or SCMError(
            message="Unknown error after retries", provider=self.name
        )

    @staticmethod
    def _encode_path(path: str) -> str:
        """URL-encode a path for GitLab API."""
        return quote(path, safe="")

    def _parse_user(self, data: dict[str, Any]) -> User:
        """Parse GitLab user data."""
        return User(
            id=data.get("id", 0),
            username=data.get("username", ""),
            name=data.get("name"),
            email=data.get("email"),
            avatar_url=data.get("avatar_url"),
            html_url=data.get("web_url"),
        )

    def _parse_repository(self, data: dict[str, Any]) -> Repository:
        """Parse GitLab project data."""
        return Repository(
            id=data.get("id", 0),
            name=data.get("name", ""),
            full_name=data.get("path_with_namespace", ""),
            owner=(
                data.get("namespace", {}).get("name", "")
                if data.get("namespace")
                else ""
            ),
            description=data.get("description"),
            html_url=data.get("web_url", ""),
            clone_url=data.get("http_url_to_repo"),
            ssh_url=data.get("ssh_url_to_repo"),
            default_branch=data.get("default_branch"),
            private=data.get("visibility", "public") != "public",
            fork=data.get("forked_from_project") is not None,
            parent=(
                data.get("forked_from_project", {}).get("path_with_namespace")
                if data.get("forked_from_project")
                else None
            ),
            created_at=data.get("created_at", ""),
            updated_at=data.get("last_activity_at"),
        )

    def _parse_branch(self, data: dict[str, Any]) -> Branch:
        """Parse GitLab branch data."""
        commit_data = data.get("commit", {})
        return Branch(
            name=data.get("name", ""),
            commit_sha=commit_data.get("id", "") if commit_data else "",
            protected=data.get("protected", False),
            default=data.get("default", False),
        )

    def _parse_pull_request(self, data: dict[str, Any]) -> PullRequest:
        """Parse GitLab merge request data."""
        author_data = data.get("author")
        return PullRequest(
            id=data.get("id", 0),
            number=data.get("iid", 0),
            title=data.get("title", ""),
            body=data.get("description"),
            state=data.get("state", ""),
            head=data.get("source_branch", ""),
            head_repo=(
                data.get("source_project", {}).get("path_with_namespace")
                if data.get("source_project")
                else None
            ),
            base=data.get("target_branch", ""),
            base_repo=(
                data.get("target_project", {}).get("path_with_namespace")
                if data.get("target_project")
                else ""
            ),
            author=self._parse_user(author_data) if author_data else None,
            draft=data.get("draft", False),
            mergeable=data.get("merge_status") == "can_be_merged",
            merged=data.get("state") == "merged",
            merged_at=data.get("merged_at"),
            html_url=data.get("web_url", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
        )

    def _parse_issue(self, data: dict[str, Any]) -> Issue:
        """Parse GitLab issue data."""
        author_data = data.get("author")
        return Issue(
            id=data.get("id", 0),
            number=data.get("iid", 0),
            title=data.get("title", ""),
            body=data.get("description"),
            state=data.get("state", ""),
            author=self._parse_user(author_data) if author_data else None,
            labels=data.get("labels", []),
            milestone=(
                data.get("milestone", {}).get("title")
                if data.get("milestone")
                else None
            ),
            html_url=data.get("web_url", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
            closed_at=data.get("closed_at"),
        )

    def _parse_comment(self, data: dict[str, Any]) -> Comment:
        """Parse GitLab comment/note data."""
        author_data = data.get("author")
        return Comment(
            id=data.get("id", 0),
            body=data.get("body", "") or data.get("note", ""),
            author=self._parse_user(author_data) if author_data else None,
            html_url=(
                data.get("web_url", "") or data.get("noteable", {}).get("web_url", "")
            ),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
        )

    def _parse_webhook(self, data: dict[str, Any]) -> Webhook:
        """Parse GitLab webhook data."""
        # Reconstruct events list from boolean event flags
        events: list[str] = []
        if data.get("push_events"):
            events.append("push")
        if data.get("issues_events"):
            events.append("issues")
        if data.get("merge_requests_events"):
            events.append("merge_requests")
        if data.get("wiki_page_events"):
            events.append("wiki")
        if data.get("releases_events"):
            events.append("releases")
        if data.get("tag_push_events"):
            events.append("tag_push")
        if data.get("note_events"):
            events.append("note")
        if data.get("job_events"):
            events.append("job")
        if data.get("pipeline_events"):
            events.append("pipeline")

        return Webhook(
            id=data.get("id", 0),
            url=data.get("url", ""),
            events=events,
            active=data.get("active", True),
            created_at=data.get("created_at"),
        )

    async def list_repos(self, owner: str, **filters: Any) -> list[Repository]:
        """List repositories for an owner."""
        params: dict[str, Any] = {"per_page": 100}
        params.update(filters)
        data = await self._request(
            "GET", f"/groups/{self._encode_path(owner)}/projects", params=params
        )
        return [self._parse_repository(repo) for repo in data]

    async def get_repo(self, owner: str, name: str) -> Repository:
        """Get a specific repository."""
        path = self._encode_path(f"{owner}/{name}")
        data = await self._request("GET", f"/projects/{path}")
        return self._parse_repository(data)

    async def create_repo(self, name: str, **opts: Any) -> Repository:
        """Create a new repository."""
        body: dict[str, Any] = {"name": name}
        body.update(opts)
        data = await self._request("POST", "/projects", json=body)
        return self._parse_repository(data)

    async def create_fork(self, owner: str, repo: str, **opts: Any) -> Repository:
        """Fork a repository."""
        path = self._encode_path(f"{owner}/{repo}")
        body: dict[str, Any] = {}
        if "namespace" in opts:
            body["namespace_path"] = opts["namespace"]
        data = await self._request(
            "POST", f"/projects/{path}/fork", json=body if body else None
        )
        return self._parse_repository(data)

    async def list_branches(
        self, owner: str, repo: str, **filters: Any
    ) -> list[Branch]:
        """List branches in a repository."""
        path = self._encode_path(f"{owner}/{repo}")
        params: dict[str, Any] = {"per_page": 100}
        params.update(filters)
        data = await self._request(
            "GET", f"/projects/{path}/repository/branches", params=params
        )
        return [self._parse_branch(branch) for branch in data]

    async def get_branch(self, owner: str, repo: str, branch: str) -> Branch:
        """Get a specific branch."""
        path = self._encode_path(f"{owner}/{repo}")
        data = await self._request(
            "GET",
            f"/projects/{path}/repository/branches/{self._encode_path(branch)}",
        )
        return self._parse_branch(data)

    async def create_pull_request(
        self, owner: str, repo: str, title: str, head: str, base: str, **opts: Any
    ) -> PullRequest:
        """Create a pull request."""
        path = self._encode_path(f"{owner}/{repo}")
        body: dict[str, Any] = {
            "title": title,
            "source_branch": head,
            "target_branch": base,
        }
        body.update(opts)
        data = await self._request(
            "POST", f"/projects/{path}/merge_requests", json=body
        )
        return self._parse_pull_request(data)

    async def get_pull_request(self, owner: str, repo: str, number: int) -> PullRequest:
        """Get a specific pull request."""
        path = self._encode_path(f"{owner}/{repo}")
        data = await self._request("GET", f"/projects/{path}/merge_requests/{number}")
        return self._parse_pull_request(data)

    async def list_pull_requests(
        self, owner: str, repo: str, **filters: Any
    ) -> list[PullRequest]:
        """List pull requests in a repository."""
        path = self._encode_path(f"{owner}/{repo}")
        params: dict[str, Any] = {"per_page": 100}
        params.update(filters)
        data = await self._request(
            "GET", f"/projects/{path}/merge_requests", params=params
        )
        return [self._parse_pull_request(pr) for pr in data]

    async def approve_pull_request(
        self, owner: str, repo: str, number: int, **opts: Any
    ) -> PullRequest:
        """Approve a pull request."""
        path = self._encode_path(f"{owner}/{repo}")
        body: dict[str, Any] = {}
        if opts.get("body"):
            body["note"] = opts["body"]
        await self._request(
            "POST",
            f"/projects/{path}/merge_requests/{number}/approve",
            json=body if body else None,
        )
        return await self.get_pull_request(owner, repo, number)

    async def comment_on_pull_request(
        self, owner: str, repo: str, number: int, body: str
    ) -> Comment:
        """Add a comment to a pull request."""
        path = self._encode_path(f"{owner}/{repo}")
        data = await self._request(
            "POST",
            f"/projects/{path}/merge_requests/{number}/notes",
            json={"body": body},
        )
        return self._parse_comment(data)

    async def create_issue(
        self, owner: str, repo: str, title: str, **opts: Any
    ) -> Issue:
        """Create an issue."""
        path = self._encode_path(f"{owner}/{repo}")
        body: dict[str, Any] = {"title": title}
        body.update(opts)
        data = await self._request("POST", f"/projects/{path}/issues", json=body)
        return self._parse_issue(data)

    async def list_issues(self, owner: str, repo: str, **filters: Any) -> list[Issue]:
        """List issues in a repository."""
        path = self._encode_path(f"{owner}/{repo}")
        params: dict[str, Any] = {"per_page": 100}
        params.update(filters)
        data = await self._request("GET", f"/projects/{path}/issues", params=params)
        return [self._parse_issue(issue) for issue in data]

    async def create_webhook(
        self, owner: str, repo: str, url: str, events: list[str], **opts: Any
    ) -> Webhook:
        """Create a webhook."""
        path = self._encode_path(f"{owner}/{repo}")
        body: dict[str, Any] = {"url": url}
        for event in events:
            if event == "push":
                body["push_events"] = True
            elif event == "issues":
                body["issues_events"] = True
            elif event == "merge_requests":
                body["merge_requests_events"] = True
            elif event == "wiki":
                body["wiki_page_events"] = True
            elif event == "releases":
                body["releases_events"] = True
        if "secret" in opts:
            body["token"] = opts.pop("secret")
        body["active"] = opts.get("active", True)
        data = await self._request("POST", f"/projects/{path}/hooks", json=body)
        return self._parse_webhook(data)

    async def list_webhooks(self, owner: str, repo: str) -> list[Webhook]:
        """List webhooks for a repository."""
        path = self._encode_path(f"{owner}/{repo}")
        data = await self._request("GET", f"/projects/{path}/hooks")
        return [self._parse_webhook(hook) for hook in data]


# ---------------------------------------------------------------------------
# SCM Registry
# ---------------------------------------------------------------------------


class SCMRegistry:
    """Registry for SCM providers."""

    def __init__(self) -> None:
        """Initialize SCMRegistry."""
        self._providers: dict[str, SCMProvider] = {}

    def register(self, name: str, provider: SCMProvider) -> None:
        """Register a provider."""
        self._providers[name] = provider

    def get(self, name: str) -> SCMProvider:
        """Get a provider by name."""
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        return self._providers[name]

    def list_providers(self) -> list[str]:
        """List all registered providers."""
        return list(self._providers.keys())

    async def close_all(self) -> None:
        """Close all providers."""
        await asyncio.gather(
            *[provider.close() for provider in self._providers.values()],
            return_exceptions=True,
        )


# ---------------------------------------------------------------------------
# SafeAgent Module
# ---------------------------------------------------------------------------


class SCMModule(BaseModule):
    """SafeAgent module exposing SCM tools."""

    def __init__(self, registry: SCMRegistry | None = None) -> None:
        """Initialize SCMModule."""
        self._registry = registry or SCMRegistry()

    def describe(self) -> ModuleDescriptor:
        """Return module descriptor."""
        return ModuleDescriptor(
            namespace="scm",
            description="Source Control Management operations (GitHub, GitLab).",
            tools=[
                ToolDescriptor(
                    name="scm:CreatePullRequest",
                    description="Create a pull/merge request.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name (github or gitlab).",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Repository owner.",
                            },
                            "repo": {
                                "type": "string",
                                "description": "Repository name.",
                            },
                            "title": {"type": "string", "description": "PR title."},
                            "head": {
                                "type": "string",
                                "description": "Source branch.",
                            },
                            "base": {
                                "type": "string",
                                "description": "Target branch.",
                            },
                            "body": {
                                "type": "string",
                                "description": "PR description.",
                            },
                            "draft": {
                                "type": "boolean",
                                "description": "Create as draft PR.",
                            },
                        },
                        "required": [
                            "provider",
                            "owner",
                            "repo",
                            "title",
                            "head",
                            "base",
                        ],
                        "additionalProperties": False,
                    },
                    action="scm:CreatePullRequest",
                    resource_param=["owner", "repo"],
                    condition_keys=["scm:Repository"],
                ),
                ToolDescriptor(
                    name="scm:CreateIssue",
                    description="Create an issue.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name.",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Repository owner.",
                            },
                            "repo": {
                                "type": "string",
                                "description": "Repository name.",
                            },
                            "title": {
                                "type": "string",
                                "description": "Issue title.",
                            },
                            "body": {
                                "type": "string",
                                "description": "Issue description.",
                            },
                            "labels": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Labels to apply.",
                            },
                        },
                        "required": ["provider", "owner", "repo", "title"],
                        "additionalProperties": False,
                    },
                    action="scm:CreateIssue",
                    resource_param=["owner", "repo"],
                    condition_keys=["scm:Repository"],
                ),
                ToolDescriptor(
                    name="scm:CreateRepo",
                    description="Create a new repository.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name.",
                            },
                            "name": {
                                "type": "string",
                                "description": "Repository name.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Repository description.",
                            },
                            "private": {
                                "type": "boolean",
                                "description": "Make repository private.",
                            },
                        },
                        "required": ["provider", "name"],
                        "additionalProperties": False,
                    },
                    action="scm:CreateRepo",
                    resource_param=["name"],
                    condition_keys=["scm:Repository"],
                ),
                ToolDescriptor(
                    name="scm:CreateFork",
                    description="Fork a repository.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name.",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Repository owner.",
                            },
                            "repo": {
                                "type": "string",
                                "description": "Repository name.",
                            },
                            "namespace": {
                                "type": "string",
                                "description": "Target namespace for fork.",
                            },
                        },
                        "required": ["provider", "owner", "repo"],
                        "additionalProperties": False,
                    },
                    action="scm:CreateFork",
                    resource_param=["owner", "repo"],
                    condition_keys=["scm:Repository"],
                ),
                ToolDescriptor(
                    name="scm:ListRepos",
                    description="List repositories for an owner.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name.",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Owner username or organization.",
                            },
                        },
                        "required": ["provider", "owner"],
                        "additionalProperties": False,
                    },
                    action="scm:ListRepos",
                    resource_param=["owner"],
                    condition_keys=["scm:Owner"],
                ),
                ToolDescriptor(
                    name="scm:ListBranches",
                    description="List branches in a repository.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name.",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Repository owner.",
                            },
                            "repo": {
                                "type": "string",
                                "description": "Repository name.",
                            },
                        },
                        "required": ["provider", "owner", "repo"],
                        "additionalProperties": False,
                    },
                    action="scm:ListBranches",
                    resource_param=["owner", "repo"],
                    condition_keys=["scm:Repository"],
                ),
                ToolDescriptor(
                    name="scm:CreateWebhook",
                    description="Create a webhook.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name.",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Repository owner.",
                            },
                            "repo": {
                                "type": "string",
                                "description": "Repository name.",
                            },
                            "url": {
                                "type": "string",
                                "description": "Webhook URL.",
                            },
                            "events": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Event types.",
                            },
                            "secret": {
                                "type": "string",
                                "description": "Webhook secret.",
                            },
                        },
                        "required": ["provider", "owner", "repo", "url", "events"],
                        "additionalProperties": False,
                    },
                    action="scm:CreateWebhook",
                    resource_param=["owner", "repo"],
                    condition_keys=["scm:Repository"],
                ),
                ToolDescriptor(
                    name="scm:ApprovePullRequest",
                    description="Approve a pull request.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name.",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Repository owner.",
                            },
                            "repo": {
                                "type": "string",
                                "description": "Repository name.",
                            },
                            "number": {
                                "type": "integer",
                                "description": "PR number.",
                            },
                            "body": {
                                "type": "string",
                                "description": "Review comment.",
                            },
                        },
                        "required": ["provider", "owner", "repo", "number"],
                        "additionalProperties": False,
                    },
                    action="scm:ApprovePullRequest",
                    resource_param=["owner", "repo"],
                    condition_keys=["scm:Repository"],
                ),
                ToolDescriptor(
                    name="scm:CommentOnPullRequest",
                    description="Add a comment to a pull request.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "description": "Provider name.",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Repository owner.",
                            },
                            "repo": {
                                "type": "string",
                                "description": "Repository name.",
                            },
                            "number": {
                                "type": "integer",
                                "description": "PR number.",
                            },
                            "body": {
                                "type": "string",
                                "description": "Comment body.",
                            },
                        },
                        "required": [
                            "provider",
                            "owner",
                            "repo",
                            "number",
                            "body",
                        ],
                        "additionalProperties": False,
                    },
                    action="scm:CommentOnPullRequest",
                    resource_param=["owner", "repo"],
                    condition_keys=["scm:Repository"],
                ),
            ],
        )

    async def resolve_conditions(
        self, tool_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve condition keys for policy evaluation."""
        conditions: dict[str, Any] = {}
        if "owner" in params:
            conditions["scm:Owner"] = params["owner"]
        if "repo" in params:
            conditions["scm:Repository"] = f"{params.get('owner', '')}/{params['repo']}"
        return conditions

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult[Any]:
        """Execute a tool."""
        try:
            provider_name = params.get("provider")
            if not provider_name:
                return ToolResult(success=False, error="provider is required")

            provider = self._registry.get(provider_name)
            normalized_tool = tool_name.removeprefix("scm:")

            if normalized_tool == "CreatePullRequest":
                pr = await provider.create_pull_request(
                    owner=params["owner"],
                    repo=params["repo"],
                    title=params["title"],
                    head=params["head"],
                    base=params["base"],
                    body=params.get("body"),
                    draft=params.get("draft", False),
                )
                return ToolResult(success=True, data=pr.model_dump())

            if normalized_tool == "CreateIssue":
                issue = await provider.create_issue(
                    owner=params["owner"],
                    repo=params["repo"],
                    title=params["title"],
                    body=params.get("body"),
                    labels=params.get("labels", []),
                )
                return ToolResult(success=True, data=issue.model_dump())

            if normalized_tool == "CreateRepo":
                repo = await provider.create_repo(
                    name=params["name"],
                    description=params.get("description"),
                    private=params.get("private", False),
                )
                return ToolResult(success=True, data=repo.model_dump())

            if normalized_tool == "CreateFork":
                namespace = params.get("namespace")
                opts = {"namespace": namespace} if namespace else {}
                fork = await provider.create_fork(
                    owner=params["owner"],
                    repo=params["repo"],
                    **opts,
                )
                return ToolResult(success=True, data=fork.model_dump())

            if normalized_tool == "ListRepos":
                repos = await provider.list_repos(owner=params["owner"])
                return ToolResult(success=True, data=[r.model_dump() for r in repos])

            if normalized_tool == "ListBranches":
                branches = await provider.list_branches(
                    owner=params["owner"], repo=params["repo"]
                )
                return ToolResult(success=True, data=[b.model_dump() for b in branches])

            if normalized_tool == "CreateWebhook":
                secret = params.get("secret")
                opts = {"secret": secret} if secret else {}
                webhook = await provider.create_webhook(
                    owner=params["owner"],
                    repo=params["repo"],
                    url=params["url"],
                    events=params["events"],
                    **opts,
                )
                return ToolResult(success=True, data=webhook.model_dump())

            if normalized_tool == "ApprovePullRequest":
                pr = await provider.approve_pull_request(
                    owner=params["owner"],
                    repo=params["repo"],
                    number=params["number"],
                    body=params.get("body"),
                )
                return ToolResult(success=True, data=pr.model_dump())

            if normalized_tool == "CommentOnPullRequest":
                comment = await provider.comment_on_pull_request(
                    owner=params["owner"],
                    repo=params["repo"],
                    number=params["number"],
                    body=params["body"],
                )
                return ToolResult(success=True, data=comment.model_dump())

            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        except SCMError as exc:
            return ToolResult(success=False, error=str(exc))
        except KeyError as exc:
            return ToolResult(success=False, error=f"Provider not found: {exc}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Unexpected error: {exc}")

    @property
    def registry(self) -> SCMRegistry:
        """Return the registry."""
        return self._registry
