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

"""Tests for the SCM module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from safe_agent.modules.scm import (
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
    SCMRegistry,
    User,
    Webhook,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def github_provider() -> GitHubSCM:
    """Create a GitHub SCM provider instance."""
    return GitHubSCM(token="test-token")


@pytest.fixture
def gitlab_provider() -> GitLabSCM:
    """Create a GitLab SCM provider instance."""
    return GitLabSCM(token="test-token")


@pytest.fixture
def scm_registry() -> SCMRegistry:
    """Create an SCM registry with mock providers."""
    registry = SCMRegistry()
    github = GitHubSCM(token="test-token")
    gitlab = GitLabSCM(token="test-token")
    registry.register("github", github)
    registry.register("gitlab", gitlab)
    return registry


@pytest.fixture
def scm_module(scm_registry: SCMRegistry) -> SCMModule:
    """Create an SCMModule instance."""
    return SCMModule(registry=scm_registry)


# ---------------------------------------------------------------------------
# Data Model Tests
# ---------------------------------------------------------------------------


class TestUser:
    """Tests for User model."""

    def test_user_creation(self) -> None:
        """Test creating a User instance."""
        user = User(id=1, username="testuser", name="Test User")
        assert user.id == 1
        assert user.username == "testuser"
        assert user.name == "Test User"

    def test_user_optional_fields(self) -> None:
        """Test User with optional fields."""
        user = User(
            id=1,
            username="testuser",
            email="test@example.com",
            avatar_url="https://example.com/avatar.png",
            html_url="https://example.com/user",
        )
        assert user.email == "test@example.com"


class TestRepository:
    """Tests for Repository model."""

    def test_repository_creation(self) -> None:
        """Test creating a Repository instance."""
        repo = Repository(
            id=1,
            name="test-repo",
            full_name="owner/test-repo",
            owner="owner",
            html_url="https://github.com/owner/test-repo",
            created_at="2024-01-01T00:00:00Z",
        )
        assert repo.name == "test-repo"
        assert repo.private is False


class TestBranch:
    """Tests for Branch model."""

    def test_branch_creation(self) -> None:
        """Test creating a Branch instance."""
        branch = Branch(name="main", commit_sha="abc123")
        assert branch.name == "main"
        assert branch.commit_sha == "abc123"
        assert branch.protected is False


class TestPullRequest:
    """Tests for PullRequest model."""

    def test_pull_request_creation(self) -> None:
        """Test creating a PullRequest instance."""
        pr = PullRequest(
            id=1,
            number=42,
            title="Test PR",
            state="open",
            head="feature-branch",
            base="main",
            base_repo="owner/repo",
            html_url="https://github.com/owner/repo/pull/42",
            created_at="2024-01-01T00:00:00Z",
        )
        assert pr.number == 42
        assert pr.draft is False


class TestIssue:
    """Tests for Issue model."""

    def test_issue_creation(self) -> None:
        """Test creating an Issue instance."""
        issue = Issue(
            id=1,
            number=42,
            title="Test Issue",
            state="open",
            html_url="https://github.com/owner/repo/issues/42",
            created_at="2024-01-01T00:00:00Z",
        )
        assert issue.number == 42
        assert issue.labels == []


class TestComment:
    """Tests for Comment model."""

    def test_comment_creation(self) -> None:
        """Test creating a Comment instance."""
        comment = Comment(
            id=1,
            body="Test comment",
            html_url="https://github.com/owner/repo/issues/1#issuecomment-1",
            created_at="2024-01-01T00:00:00Z",
        )
        assert comment.body == "Test comment"


class TestWebhook:
    """Tests for Webhook model."""

    def test_webhook_creation(self) -> None:
        """Test creating a Webhook instance."""
        webhook = Webhook(id=1, url="https://example.com/webhook", events=["push"])
        assert webhook.url == "https://example.com/webhook"
        assert webhook.active is True


# ---------------------------------------------------------------------------
# Error Tests
# ---------------------------------------------------------------------------


class TestSCMError:
    """Tests for SCMError exception."""

    def test_error_creation(self) -> None:
        """Test creating an SCMError."""
        error = SCMError(message="Not found", provider="github", status_code=404)
        assert str(error) == "[github] Not found (status=404)"

    def test_error_without_status(self) -> None:
        """Test SCMError without status code."""
        error = SCMError(message="Error", provider="github")
        assert str(error) == "[github] Error"


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_rate_limit_error(self) -> None:
        """Test creating a RateLimitError."""
        error = RateLimitError(
            provider="github",
            reset_at=1234567890,
            remaining=0,
            status_code=403,
        )
        assert error.reset_at == 1234567890
        assert "Rate limit exceeded" in str(error)


# ---------------------------------------------------------------------------
# SCM Registry Tests
# ---------------------------------------------------------------------------


class TestSCMRegistry:
    """Tests for SCMRegistry."""

    def test_register_provider(self) -> None:
        """Test registering a provider."""
        registry = SCMRegistry()
        provider = GitHubSCM(token="test")
        registry.register("github", provider)
        assert "github" in registry.list_providers()

    def test_get_provider(self) -> None:
        """Test getting a provider."""
        registry = SCMRegistry()
        provider = GitHubSCM(token="test")
        registry.register("github", provider)
        retrieved = registry.get("github")
        assert retrieved is provider

    def test_get_nonexistent_provider(self) -> None:
        """Test getting a nonexistent provider raises KeyError."""
        registry = SCMRegistry()
        with pytest.raises(KeyError, match="Provider 'nonexistent' not registered"):
            registry.get("nonexistent")

    def test_list_providers(self) -> None:
        """Test listing providers."""
        registry = SCMRegistry()
        registry.register("github", GitHubSCM(token="test"))
        registry.register("gitlab", GitLabSCM(token="test"))
        providers = registry.list_providers()
        assert "github" in providers
        assert "gitlab" in providers


# ---------------------------------------------------------------------------
# GitHub SCM Tests
# ---------------------------------------------------------------------------


class TestGitHubSCM:
    """Tests for GitHubSCM provider."""

    def test_provider_name(self, github_provider: GitHubSCM) -> None:
        """Test provider name."""
        assert github_provider.name == "github"

    def test_custom_api_url(self) -> None:
        """Test custom API URL for GitHub Enterprise."""
        provider = GitHubSCM(token="test", api_url="https://github.example.com/api")
        assert provider._api_url == "https://github.example.com/api"

    @pytest.mark.asyncio
    async def test_close_client(self, github_provider: GitHubSCM) -> None:
        """Test closing the HTTP client."""
        await github_provider.close()
        assert github_provider._client is None

    def test_parse_user(self, github_provider: GitHubSCM) -> None:
        """Test parsing GitHub user data."""
        user_data = {
            "id": 1,
            "login": "testuser",
            "name": "Test User",
            "email": "test@example.com",
        }
        user = github_provider._parse_user(user_data)
        assert user.id == 1
        assert user.username == "testuser"

    def test_parse_repository(self, github_provider: GitHubSCM) -> None:
        """Test parsing GitHub repository data."""
        repo_data = {
            "id": 1,
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "owner": {"login": "owner"},
            "html_url": "https://github.com/owner/test-repo",
            "private": True,
            "created_at": "2024-01-01T00:00:00Z",
        }
        repo = github_provider._parse_repository(repo_data)
        assert repo.name == "test-repo"
        assert repo.private is True

    def test_parse_branch(self, github_provider: GitHubSCM) -> None:
        """Test parsing GitHub branch data."""
        branch_data = {
            "name": "main",
            "commit": {"sha": "abc123"},
            "protected": True,
        }
        branch = github_provider._parse_branch(branch_data)
        assert branch.name == "main"
        assert branch.commit_sha == "abc123"
        assert branch.protected is True

    def test_parse_pull_request(self, github_provider: GitHubSCM) -> None:
        """Test parsing GitHub pull request data."""
        pr_data = {
            "id": 1,
            "number": 42,
            "title": "Test PR",
            "state": "open",
            "head": {"ref": "feature", "repo": {"full_name": "owner/repo"}},
            "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
            "user": {"id": 1, "login": "testuser"},
            "html_url": "https://github.com/owner/repo/pull/42",
            "created_at": "2024-01-01T00:00:00Z",
        }
        pr = github_provider._parse_pull_request(pr_data)
        assert pr.number == 42
        assert pr.title == "Test PR"

    def test_parse_issue(self, github_provider: GitHubSCM) -> None:
        """Test parsing GitHub issue data."""
        issue_data = {
            "id": 1,
            "number": 42,
            "title": "Test Issue",
            "state": "open",
            "user": {"id": 1, "login": "testuser"},
            "labels": [{"name": "bug"}, {"name": "enhancement"}],
            "html_url": "https://github.com/owner/repo/issues/42",
            "created_at": "2024-01-01T00:00:00Z",
        }
        issue = github_provider._parse_issue(issue_data)
        assert issue.number == 42
        assert issue.labels == ["bug", "enhancement"]

    def test_parse_comment(self, github_provider: GitHubSCM) -> None:
        """Test parsing GitHub comment data."""
        comment_data = {
            "id": 1,
            "body": "Test comment",
            "user": {"id": 1, "login": "testuser"},
            "html_url": "https://github.com/owner/repo/issues/1#issuecomment-1",
            "created_at": "2024-01-01T00:00:00Z",
        }
        comment = github_provider._parse_comment(comment_data)
        assert comment.body == "Test comment"

    def test_parse_webhook(self, github_provider: GitHubSCM) -> None:
        """Test parsing GitHub webhook data."""
        webhook_data = {
            "id": 1,
            "config": {"url": "https://example.com/webhook"},
            "events": ["push", "pull_request"],
            "active": True,
        }
        webhook = github_provider._parse_webhook(webhook_data)
        assert webhook.url == "https://example.com/webhook"
        assert webhook.events == ["push", "pull_request"]


# ---------------------------------------------------------------------------
# GitLab SCM Tests
# ---------------------------------------------------------------------------


class TestGitLabSCM:
    """Tests for GitLabSCM provider."""

    def test_provider_name(self, gitlab_provider: GitLabSCM) -> None:
        """Test provider name."""
        assert gitlab_provider.name == "gitlab"

    def test_custom_api_url(self) -> None:
        """Test custom API URL for self-hosted GitLab."""
        provider = GitLabSCM(token="test", api_url="https://gitlab.example.com/api/v4")
        assert provider._api_url == "https://gitlab.example.com/api/v4"

    def test_encode_path(self, gitlab_provider: GitLabSCM) -> None:
        """Test URL encoding for GitLab paths."""
        encoded = gitlab_provider._encode_path("group/subgroup/project")
        assert "/" not in encoded
        assert encoded == "group%2Fsubgroup%2Fproject"

    def test_parse_user(self, gitlab_provider: GitLabSCM) -> None:
        """Test parsing GitLab user data."""
        user_data = {
            "id": 1,
            "username": "testuser",
            "name": "Test User",
            "web_url": "https://gitlab.com/testuser",
        }
        user = gitlab_provider._parse_user(user_data)
        assert user.username == "testuser"
        assert user.html_url == "https://gitlab.com/testuser"

    def test_parse_repository(self, gitlab_provider: GitLabSCM) -> None:
        """Test parsing GitLab project data."""
        project_data = {
            "id": 1,
            "name": "test-project",
            "path_with_namespace": "group/test-project",
            "namespace": {"name": "group"},
            "web_url": "https://gitlab.com/group/test-project",
            "visibility": "private",
            "created_at": "2024-01-01T00:00:00Z",
        }
        repo = gitlab_provider._parse_repository(project_data)
        assert repo.full_name == "group/test-project"
        assert repo.private is True

    def test_parse_branch(self, gitlab_provider: GitLabSCM) -> None:
        """Test parsing GitLab branch data."""
        branch_data = {
            "name": "main",
            "commit": {"id": "abc123"},
            "protected": True,
            "default": True,
        }
        branch = gitlab_provider._parse_branch(branch_data)
        assert branch.name == "main"
        assert branch.default is True

    def test_parse_pull_request(self, gitlab_provider: GitLabSCM) -> None:
        """Test parsing GitLab merge request data."""
        mr_data = {
            "id": 1,
            "iid": 42,
            "title": "Test MR",
            "description": "Test description",
            "state": "opened",
            "source_branch": "feature",
            "target_branch": "main",
            "author": {"id": 1, "username": "testuser"},
            "web_url": "https://gitlab.com/group/project/-/merge_requests/42",
            "created_at": "2024-01-01T00:00:00Z",
        }
        pr = gitlab_provider._parse_pull_request(mr_data)
        assert pr.number == 42
        assert pr.head == "feature"
        assert pr.base == "main"

    def test_parse_issue(self, gitlab_provider: GitLabSCM) -> None:
        """Test parsing GitLab issue data."""
        issue_data = {
            "id": 1,
            "iid": 42,
            "title": "Test Issue",
            "description": "Test description",
            "state": "opened",
            "labels": ["bug", "enhancement"],
            "web_url": "https://gitlab.com/group/project/-/issues/42",
            "created_at": "2024-01-01T00:00:00Z",
        }
        issue = gitlab_provider._parse_issue(issue_data)
        assert issue.number == 42
        assert issue.labels == ["bug", "enhancement"]


# ---------------------------------------------------------------------------
# SCM Module Tests
# ---------------------------------------------------------------------------


class TestSCMModule:
    """Tests for SCMModule."""

    def test_describe(self, scm_module: SCMModule) -> None:
        """Test module descriptor."""
        descriptor = scm_module.describe()
        assert descriptor.namespace == "scm"
        assert len(descriptor.tools) == 9

    def test_tool_names(self, scm_module: SCMModule) -> None:
        """Test tool names in descriptor."""
        descriptor = scm_module.describe()
        tool_names = [t.name for t in descriptor.tools]
        assert "scm:CreatePullRequest" in tool_names
        assert "scm:CreateIssue" in tool_names
        assert "scm:CreateRepo" in tool_names
        assert "scm:CreateFork" in tool_names
        assert "scm:ListRepos" in tool_names
        assert "scm:ListBranches" in tool_names
        assert "scm:CreateWebhook" in tool_names
        assert "scm:ApprovePullRequest" in tool_names
        assert "scm:CommentOnPullRequest" in tool_names

    @pytest.mark.asyncio
    async def test_resolve_conditions_with_owner(self, scm_module: SCMModule) -> None:
        """Test resolving conditions with owner."""
        conditions = await scm_module.resolve_conditions(
            "scm:ListRepos",
            {"provider": "github", "owner": "testorg"},
        )
        assert conditions["scm:Owner"] == "testorg"

    @pytest.mark.asyncio
    async def test_resolve_conditions_with_repo(self, scm_module: SCMModule) -> None:
        """Test resolving conditions with owner and repo."""
        conditions = await scm_module.resolve_conditions(
            "scm:CreateIssue",
            {"provider": "github", "owner": "testorg", "repo": "testrepo"},
        )
        assert conditions["scm:Owner"] == "testorg"
        assert conditions["scm:Repository"] == "testorg/testrepo"

    @pytest.mark.asyncio
    async def test_execute_missing_provider(self, scm_module: SCMModule) -> None:
        """Test execute without provider."""
        result = await scm_module.execute("scm:CreateRepo", {"name": "test"})
        assert result.success is False
        assert "provider is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_provider(self, scm_module: SCMModule) -> None:
        """Test execute with unknown provider."""
        result = await scm_module.execute(
            "scm:CreateRepo",
            {"provider": "unknown", "name": "test"},
        )
        assert result.success is False
        assert "Provider not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, scm_module: SCMModule) -> None:
        """Test execute with unknown tool."""
        result = await scm_module.execute(
            "scm:Unknown",
            {"provider": "github"},
        )
        assert result.success is False
        assert "Unknown tool" in result.error


# ---------------------------------------------------------------------------
# Integration Tests (Mocked HTTP)
# ---------------------------------------------------------------------------


class TestGitHubSCMIntegration:
    """Integration tests for GitHubSCM with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_list_repos(self) -> None:
        """Test listing repositories."""
        provider = GitHubSCM(token="test")

        mock_response = [
            {
                "id": 1,
                "name": "repo1",
                "full_name": "owner/repo1",
                "owner": {"login": "owner"},
                "html_url": "https://github.com/owner/repo1",
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]

        with patch.object(
            provider, "_request", new_callable=AsyncMock, return_value=mock_response
        ):
            repos = await provider.list_repos("owner")
            assert len(repos) == 1
            assert repos[0].name == "repo1"

    @pytest.mark.asyncio
    async def test_create_pull_request(self) -> None:
        """Test creating a pull request."""
        provider = GitHubSCM(token="test")

        mock_response = {
            "id": 1,
            "number": 42,
            "title": "Test PR",
            "state": "open",
            "head": {"ref": "feature"},
            "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
            "html_url": "https://github.com/owner/repo/pull/42",
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch.object(
            provider, "_request", new_callable=AsyncMock, return_value=mock_response
        ):
            pr = await provider.create_pull_request(
                owner="owner",
                repo="repo",
                title="Test PR",
                head="feature",
                base="main",
            )
            assert pr.number == 42
            assert pr.title == "Test PR"

    @pytest.mark.asyncio
    async def test_create_issue(self) -> None:
        """Test creating an issue."""
        provider = GitHubSCM(token="test")

        mock_response = {
            "id": 1,
            "number": 42,
            "title": "Test Issue",
            "state": "open",
            "html_url": "https://github.com/owner/repo/issues/42",
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch.object(
            provider, "_request", new_callable=AsyncMock, return_value=mock_response
        ):
            issue = await provider.create_issue(
                owner="owner",
                repo="repo",
                title="Test Issue",
            )
            assert issue.number == 42


class TestGitLabSCMIntegration:
    """Integration tests for GitLabSCM with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_list_repos(self) -> None:
        """Test listing projects."""
        provider = GitLabSCM(token="test")

        mock_response = [
            {
                "id": 1,
                "name": "project1",
                "path_with_namespace": "group/project1",
                "namespace": {"name": "group"},
                "web_url": "https://gitlab.com/group/project1",
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]

        with patch.object(
            provider, "_request", new_callable=AsyncMock, return_value=mock_response
        ):
            repos = await provider.list_repos("group")
            assert len(repos) == 1
            assert repos[0].name == "project1"

    @pytest.mark.asyncio
    async def test_create_pull_request(self) -> None:
        """Test creating a merge request."""
        provider = GitLabSCM(token="test")

        mock_response = {
            "id": 1,
            "iid": 42,
            "title": "Test MR",
            "state": "opened",
            "source_branch": "feature",
            "target_branch": "main",
            "web_url": "https://gitlab.com/group/project/-/merge_requests/42",
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch.object(
            provider, "_request", new_callable=AsyncMock, return_value=mock_response
        ):
            pr = await provider.create_pull_request(
                owner="group",
                repo="project",
                title="Test MR",
                head="feature",
                base="main",
            )
            assert pr.number == 42


# ---------------------------------------------------------------------------
# Rate Limiting Tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limit handling."""

    @pytest.mark.asyncio
    async def test_github_rate_limit_error(self) -> None:
        """Test GitHub rate limit error is raised properly."""
        provider = GitHubSCM(token="test")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "1234567890",
        }
        mock_response.content = b""
        mock_response.json.return_value = []
        mock_client.request = AsyncMock(return_value=mock_response)

        provider._client = mock_client

        with pytest.raises(RateLimitError) as exc_info:
            await provider.list_repos("owner")

        assert exc_info.value.provider == "github"

    @pytest.mark.asyncio
    async def test_gitlab_rate_limit_error(self) -> None:
        """Test GitLab rate limit error is raised properly."""
        provider = GitLabSCM(token="test")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {
            "ratelimit-remaining": "0",
            "ratelimit-reset": "1234567890",
        }
        mock_response.content = b""
        mock_response.json.return_value = []
        mock_client.request = AsyncMock(return_value=mock_response)

        provider._client = mock_client

        with pytest.raises(RateLimitError) as exc_info:
            await provider.list_repos("group")

        assert exc_info.value.provider == "gitlab"


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_github_404_error(self) -> None:
        """Test GitHub 404 error handling."""
        provider = GitHubSCM(token="test")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.headers = {"x-ratelimit-remaining": "100"}
        mock_response.status_code = 404
        mock_response.content = b'{"message": "Not Found"}'
        mock_response.json.return_value = {"message": "Not Found"}
        mock_response.text = '{"message": "Not Found"}'
        mock_client.request = AsyncMock(return_value=mock_response)

        provider._client = mock_client

        with pytest.raises(SCMError) as exc_info:
            await provider.get_repo("owner", "nonexistent")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_github_500_error_with_retry(self) -> None:
        """Test GitHub 500 error with retries."""
        provider = GitHubSCM(token="test", max_retries=2)

        mock_client = MagicMock()

        # First two calls return 500, third returns success
        mock_response_500 = MagicMock()
        mock_response_500.headers = {"x-ratelimit-remaining": "100"}
        mock_response_500.status_code = 500
        mock_response_500.text = "Internal Server Error"

        mock_response_ok = MagicMock()
        mock_response_ok.headers = {"x-ratelimit-remaining": "100"}
        mock_response_ok.status_code = 200
        mock_response_ok.content = (
            b'{"id": 1, "name": "repo", "full_name": "owner/repo", '
            b'"owner": {"login": "owner"}, "html_url": "https://github.com/owner/repo",'
            b' "created_at": "2024-01-01T00:00:00Z"}'
        )
        mock_response_ok.json.return_value = {
            "id": 1,
            "name": "repo",
            "full_name": "owner/repo",
            "owner": {"login": "owner"},
            "html_url": "https://github.com/owner/repo",
            "created_at": "2024-01-01T00:00:00Z",
        }

        mock_client.request = AsyncMock(
            side_effect=[mock_response_500, mock_response_500, mock_response_ok]
        )

        provider._client = mock_client

        repo = await provider.get_repo("owner", "repo")
        assert repo.name == "repo"
        assert mock_client.request.call_count == 3
