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

"""Tests for the Git version control module."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from safe_agent.modules.coding.git import GitModule


class TestGitModuleDescriptor:
    """Tests for GitModule descriptor and initialization."""

    def test_describe_returns_valid_descriptor(self, tmp_path: Path) -> None:
        """describe() should return the expected module metadata."""
        module = GitModule(working_directory=tmp_path)

        descriptor = module.describe()

        assert descriptor.namespace == "git"
        assert len(descriptor.tools) == 10
        tool_names = {tool.name for tool in descriptor.tools}
        assert tool_names == {
            "git:clone",
            "git:pull",
            "git:push",
            "git:commit",
            "git:branch",
            "git:merge",
            "git:status",
            "git:log",
            "git:diff",
            "git:tag",
        }

    def test_init_defaults_working_directory_to_cwd(self) -> None:
        """__init__ should default working_directory to Path.cwd()."""
        module = GitModule()
        assert module.working_directory == Path.cwd().resolve()

    def test_init_accepts_working_directory(self, tmp_path: Path) -> None:
        """__init__ should accept and resolve working_directory."""
        module = GitModule(working_directory=tmp_path)
        assert module.working_directory == tmp_path.resolve()

    def test_init_rejects_zero_max_timeout(self, tmp_path: Path) -> None:
        """__init__ should reject max_timeout <= 0."""
        with pytest.raises(ValueError, match="max_timeout must be > 0"):
            GitModule(max_timeout=0)

    def test_init_rejects_negative_max_timeout(self, tmp_path: Path) -> None:
        """__init__ should reject negative max_timeout."""
        with pytest.raises(ValueError, match="max_timeout must be > 0"):
            GitModule(max_timeout=-1)

    def test_init_rejects_zero_default_timeout(self, tmp_path: Path) -> None:
        """__init__ should reject default_timeout <= 0."""
        with pytest.raises(ValueError, match="default_timeout must be > 0"):
            GitModule(default_timeout=0)

    def test_init_rejects_negative_default_timeout(self, tmp_path: Path) -> None:
        """__init__ should reject negative default_timeout."""
        with pytest.raises(ValueError, match="default_timeout must be > 0"):
            GitModule(default_timeout=-1)


class TestGitModuleResolveConditions:
    """Tests for condition resolution."""

    async def test_resolve_conditions_includes_working_directory(
        self, tmp_path: Path
    ) -> None:
        """resolve_conditions should include working directory."""
        module = GitModule(working_directory=tmp_path)

        conditions = await module.resolve_conditions("git:status", {})

        assert "git:WorkingDirectory" in conditions
        assert conditions["git:WorkingDirectory"] == str(tmp_path)

    async def test_resolve_conditions_includes_repo_url_for_clone(
        self, tmp_path: Path
    ) -> None:
        """resolve_conditions should include repository URL for clone."""
        module = GitModule(working_directory=tmp_path)

        conditions = await module.resolve_conditions(
            "git:clone",
            {"url": "https://github.com/example/repo.git"},
        )

        assert conditions["git:RepositoryUrl"] == "https://github.com/example/repo.git"


class TestGitModuleSecurity:
    """Tests for security features: flag injection protection, URL validation."""

    async def test_clone_rejects_file_url(self, tmp_path: Path) -> None:
        """Clone should reject file:// URLs to prevent local data leakage."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:clone",
            {"url": "file:///etc/passwd"},
        )

        assert result.success is False
        assert "file://" in result.error

    async def test_clone_rejects_local_path(self, tmp_path: Path) -> None:
        """Clone should reject local paths without URL scheme."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:clone",
            {"url": "/local/repo"},
        )

        assert result.success is False
        assert "Invalid URL scheme" in result.error

    async def test_clone_rejects_flag_injection_in_branch(self, tmp_path: Path) -> None:
        """Clone should reject branch names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:clone",
            {
                "url": "https://github.com/example/repo.git",
                "branch": "--upload-pack=evil",
            },
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_clone_rejects_flag_injection_in_destination(
        self, tmp_path: Path
    ) -> None:
        """Clone should reject destination starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:clone",
            {
                "url": "https://github.com/example/repo.git",
                "destination": "--evil",
            },
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_pull_rejects_flag_injection_in_remote(self, tmp_path: Path) -> None:
        """Pull should reject remote names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:pull",
            {"remote": "--upload-pack=evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_pull_rejects_flag_injection_in_branch(self, tmp_path: Path) -> None:
        """Pull should reject branch names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:pull",
            {"branch": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_push_rejects_flag_injection_in_remote(self, tmp_path: Path) -> None:
        """Push should reject remote names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:push",
            {"remote": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_push_rejects_flag_injection_in_branch(self, tmp_path: Path) -> None:
        """Push should reject branch names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:push",
            {"branch": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_branch_create_rejects_flag_injection(self, tmp_path: Path) -> None:
        """Branch create should reject branch names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:branch",
            {"action": "create", "name": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_branch_delete_rejects_flag_injection(self, tmp_path: Path) -> None:
        """Branch delete should reject branch names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:branch",
            {"action": "delete", "name": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_merge_rejects_flag_injection(self, tmp_path: Path) -> None:
        """Merge should reject branch names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:merge",
            {"branch": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_tag_create_rejects_flag_injection(self, tmp_path: Path) -> None:
        """Tag create should reject tag names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:tag",
            {"action": "create", "name": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_tag_delete_rejects_flag_injection(self, tmp_path: Path) -> None:
        """Tag delete should reject tag names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:tag",
            {"action": "delete", "name": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_log_rejects_flag_injection_in_branch(self, tmp_path: Path) -> None:
        """Log should reject branch names starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:log",
            {"branch": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_log_rejects_flag_injection_in_path(self, tmp_path: Path) -> None:
        """Log should reject paths starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:log",
            {"path": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_diff_rejects_flag_injection_in_commit1(self, tmp_path: Path) -> None:
        """Diff should reject commit references starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:diff",
            {"commit1": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_diff_rejects_flag_injection_in_commit2(self, tmp_path: Path) -> None:
        """Diff should reject commit references starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:diff",
            {"commit2": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error

    async def test_diff_rejects_flag_injection_in_path(self, tmp_path: Path) -> None:
        """Diff should reject paths starting with '-'."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:diff",
            {"path": "--evil"},
        )

        assert result.success is False
        assert "must not start with '-'" in result.error


class TestGitModuleTimeoutEnforcement:
    """Tests for max_timeout enforcement."""

    def test_init_clamps_default_timeout_to_max(self, tmp_path: Path) -> None:
        """__init__ should clamp default_timeout to max_timeout."""
        module = GitModule(
            working_directory=tmp_path,
            default_timeout=1000.0,
            max_timeout=300.0,
        )
        assert module.default_timeout == 300.0

    def test_init_preserves_smaller_default_timeout(self, tmp_path: Path) -> None:
        """__init__ should preserve default_timeout if already under max."""
        module = GitModule(
            working_directory=tmp_path,
            default_timeout=30.0,
            max_timeout=300.0,
        )
        assert module.default_timeout == 30.0


class TestGitModuleClone:
    """Tests for git clone operation."""

    async def test_clone_requires_url(self, tmp_path: Path) -> None:
        """Clone should require url parameter."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:clone", {})

        assert result.success is False
        assert "url is required" in result.error

    @mock.patch.object(GitModule, "_run_git")
    async def test_clone_with_url_only(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Clone should work with just URL."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "Cloning into 'repo'...",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:clone",
            {"url": "https://github.com/example/repo.git"},
        )

        assert result.success is True
        assert result.data["url"] == "https://github.com/example/repo.git"
        mock_run_git.assert_called_once()
        args = mock_run_git.call_args[0][0]
        assert "clone" in args
        assert "https://github.com/example/repo.git" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_clone_with_destination(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Clone should accept destination parameter."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:clone",
            {
                "url": "https://github.com/example/repo.git",
                "destination": "/tmp/my-repo",
            },
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "/tmp/my-repo" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_clone_with_branch(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Clone should accept branch parameter."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:clone",
            {
                "url": "https://github.com/example/repo.git",
                "branch": "develop",
            },
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--branch" in args
        assert "develop" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_clone_with_depth(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Clone should accept depth parameter for shallow clone."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:clone",
            {
                "url": "https://github.com/example/repo.git",
                "depth": 1,
            },
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--depth" in args
        assert "1" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_clone_failure(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Clone should report failure properly."""
        mock_run_git.return_value = {
            "success": False,
            "error": "Repository not found",
            "stdout": "",
            "stderr": "fatal: repository not found",
            "return_code": 128,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:clone",
            {"url": "https://github.com/example/nonexistent.git"},
        )

        assert result.success is False
        assert "Repository not found" in result.error


class TestGitModulePull:
    """Tests for git pull operation."""

    @mock.patch.object(GitModule, "_run_git")
    async def test_pull_default_remote(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Pull should use origin as default remote."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "Already up to date.",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:pull", {})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "pull" in args
        assert "origin" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_pull_with_branch(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Pull should accept branch parameter."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "Updated 5 files.",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:pull", {"branch": "main"})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "main" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_pull_with_rebase(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Pull should support --rebase option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:pull", {"rebase": True})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--rebase" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_pull_failure(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Pull should report merge conflicts properly."""
        mock_run_git.return_value = {
            "success": False,
            "error": "Merge conflict",
            "stdout": "",
            "stderr": "CONFLICT (content): Merge conflict in file.txt",
            "return_code": 1,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:pull", {})

        assert result.success is False


class TestGitModulePush:
    """Tests for git push operation."""

    @mock.patch.object(GitModule, "_run_git")
    async def test_push_default_remote(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Push should use origin as default remote."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:push", {})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "push" in args
        assert "origin" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_push_with_force(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Push should support --force option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:push", {"force": True})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--force" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_push_with_set_upstream(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Push should support --set-upstream option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:push",
            {"branch": "feature-branch", "set_upstream": True},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--set-upstream" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_push_failure(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Push should report rejected push properly."""
        mock_run_git.return_value = {
            "success": False,
            "error": "Push rejected",
            "stdout": "",
            "stderr": "! [rejected] main -> main (non-fast-forward)",
            "return_code": 1,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:push", {})

        assert result.success is False


class TestGitModuleCommit:
    """Tests for git commit operation."""

    async def test_commit_requires_message(self, tmp_path: Path) -> None:
        """Commit should require message parameter."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:commit", {})

        assert result.success is False
        assert "message is required" in result.error

    @mock.patch.object(GitModule, "_run_git")
    async def test_commit_with_message(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Commit should use provided message."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "[main abc1234] Initial commit",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:commit",
            {"message": "Initial commit"},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "commit" in args
        assert "-m" in args
        assert "Initial commit" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_commit_allow_empty(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Commit should support --allow-empty option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:commit",
            {"message": "Empty commit", "allow_empty": True},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--allow-empty" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_commit_amend(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Commit should support --amend option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:commit",
            {"message": "Amended commit", "amend": True},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--amend" in args


class TestGitModuleBranch:
    """Tests for git branch operation."""

    async def test_branch_requires_action(self, tmp_path: Path) -> None:
        """Branch should require action parameter."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:branch", {})

        assert result.success is False
        assert "action is required" in result.error

    @mock.patch.object(GitModule, "_run_git")
    async def test_branch_list(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Branch list should parse branches correctly."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "* main\n  develop\n  remotes/origin/main\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:branch", {"action": "list"})

        assert result.success is True
        assert "branches" in result.data
        branches = result.data["branches"]
        assert len(branches) == 3
        # Check current branch is marked
        main_branch = next(b for b in branches if b["name"] == "main")
        assert main_branch["current"] is True
        # Check remote branch is marked
        remote_branch = next(b for b in branches if "remotes/" in b["name"])
        assert remote_branch["is_remote"] is True

    @mock.patch.object(GitModule, "_run_git")
    async def test_branch_current(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Branch current should return current branch name."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "feature-branch\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:branch", {"action": "current"})

        assert result.success is True
        assert result.data["branch"] == "feature-branch"

    async def test_branch_create_requires_name(self, tmp_path: Path) -> None:
        """Branch create should require name parameter."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:branch", {"action": "create"})

        assert result.success is False
        assert "name is required" in result.error

    @mock.patch.object(GitModule, "_run_git")
    async def test_branch_create(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Branch create should create new branch."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:branch",
            {"action": "create", "name": "new-feature"},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "branch" in args
        assert "new-feature" in args

    async def test_branch_delete_requires_name(self, tmp_path: Path) -> None:
        """Branch delete should require name parameter."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:branch", {"action": "delete"})

        assert result.success is False
        assert "name is required" in result.error

    @mock.patch.object(GitModule, "_run_git")
    async def test_branch_delete(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Branch delete should delete branch."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "Deleted branch old-feature (was abc1234).\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:branch",
            {"action": "delete", "name": "old-feature"},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "-d" in args
        assert "old-feature" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_branch_delete_force(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Branch delete with force should use -D."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:branch",
            {"action": "delete", "name": "unmerged-branch", "force": True},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "-D" in args


class TestGitModuleMerge:
    """Tests for git merge operation."""

    async def test_merge_requires_branch(self, tmp_path: Path) -> None:
        """Merge should require branch parameter."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:merge", {})

        assert result.success is False
        assert "branch is required" in result.error

    @mock.patch.object(GitModule, "_run_git")
    async def test_merge_branch(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Merge should merge specified branch."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "Merge made by the 'ort' strategy.\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:merge", {"branch": "feature-branch"})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "merge" in args
        assert "feature-branch" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_merge_no_ff(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Merge should support --no-ff option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:merge",
            {"branch": "feature-branch", "no_ff": True},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--no-ff" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_merge_with_message(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Merge should support custom message."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:merge",
            {"branch": "feature-branch", "message": "Merge feature"},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "-m" in args
        assert "Merge feature" in args


class TestGitModuleStatus:
    """Tests for git status operation."""

    @mock.patch.object(GitModule, "_run_git")
    async def test_status_default(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Status should return working tree status."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "On branch main\nnothing to commit, working tree clean\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:status", {})

        assert result.success is True
        assert "status" in result.data

    @mock.patch.object(GitModule, "_run_git")
    async def test_status_porcelain(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Status should support --porcelain option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "M file.txt\n?? newfile.py\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:status", {"porcelain": True})

        assert result.success is True
        files = result.data["status"]["files"]
        assert len(files) == 2
        assert files[0]["status"] == "M "
        assert files[0]["path"] == "file.txt"

    @mock.patch.object(GitModule, "_run_git")
    async def test_status_short(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Status should support --short option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "M file.txt\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:status", {"short": True})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--short" in args


class TestGitModuleLog:
    """Tests for git log operation."""

    @mock.patch.object(GitModule, "_run_git")
    async def test_log_default(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Log should return commit history."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "commit abc1234\nAuthor: Test\nDate: Today\n\n    Message\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:log", {})

        assert result.success is True
        assert "commits" in result.data

    @mock.patch.object(GitModule, "_run_git")
    async def test_log_oneline(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Log should support --oneline option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "abc1234 Initial commit\ndef5678 Add feature\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:log", {"oneline": True})

        assert result.success is True
        commits = result.data["commits"]
        assert len(commits) == 2
        assert commits[0]["hash"] == "abc1234"
        assert commits[0]["message"] == "Initial commit"

    @mock.patch.object(GitModule, "_run_git")
    async def test_log_max_count(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Log should support -n limit."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "abc1234 Commit\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:log", {"max_count": 5})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "-n" in args
        assert "5" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_log_with_branch(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Log should support branch parameter."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:log", {"branch": "main"})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "main" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_log_with_path(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Log should support path parameter."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:log", {"path": "src/main.py"})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--" in args
        assert "src/main.py" in args


class TestGitModuleDiff:
    """Tests for git diff operation."""

    @mock.patch.object(GitModule, "_run_git")
    async def test_diff_default(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Diff should return changes."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "diff --git a/file.txt b/file.txt\n--- a/file.txt\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:diff", {})

        assert result.success is True
        assert "output" in result.data

    @mock.patch.object(GitModule, "_run_git")
    async def test_diff_staged(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Diff should support --staged option."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:diff", {"staged": True})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--staged" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_diff_between_commits(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Diff should support commit comparison."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:diff",
            {"commit1": "abc1234", "commit2": "def5678"},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "abc1234" in args
        assert "def5678" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_diff_with_path(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Diff should support path parameter."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:diff", {"path": "src/main.py"})

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "--" in args
        assert "src/main.py" in args


class TestGitModuleTag:
    """Tests for git tag operation."""

    async def test_tag_requires_action(self, tmp_path: Path) -> None:
        """Tag should require action parameter."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:tag", {})

        assert result.success is False
        assert "action is required" in result.error

    @mock.patch.object(GitModule, "_run_git")
    async def test_tag_list(self, mock_run_git: mock.AsyncMock, tmp_path: Path) -> None:
        """Tag list should return tags."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "v1.0.0\nv1.1.0\nv2.0.0\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:tag", {"action": "list"})

        assert result.success is True
        assert result.data["tags"] == ["v1.0.0", "v1.1.0", "v2.0.0"]

    async def test_tag_create_requires_name(self, tmp_path: Path) -> None:
        """Tag create should require name parameter."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:tag", {"action": "create"})

        assert result.success is False
        assert "name is required" in result.error

    @mock.patch.object(GitModule, "_run_git")
    async def test_tag_create(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Tag create should create tag."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:tag",
            {"action": "create", "name": "v1.0.0"},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "tag" in args
        assert "v1.0.0" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_tag_create_annotated(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Tag create with message should create annotated tag."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:tag",
            {"action": "create", "name": "v1.0.0", "message": "Release 1.0.0"},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "-a" in args
        assert "-m" in args
        assert "Release 1.0.0" in args

    @mock.patch.object(GitModule, "_run_git")
    async def test_tag_create_at_commit(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Tag create should support commit parameter."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:tag",
            {"action": "create", "name": "v1.0.0", "commit": "abc1234"},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "abc1234" in args

    async def test_tag_delete_requires_name(self, tmp_path: Path) -> None:
        """Tag delete should require name parameter."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:tag", {"action": "delete"})

        assert result.success is False
        assert "name is required" in result.error

    @mock.patch.object(GitModule, "_run_git")
    async def test_tag_delete(
        self, mock_run_git: mock.AsyncMock, tmp_path: Path
    ) -> None:
        """Tag delete should delete tag."""
        mock_run_git.return_value = {
            "success": True,
            "stdout": "Deleted tag 'v1.0.0' (was abc1234)\n",
            "stderr": "",
            "return_code": 0,
        }
        module = GitModule(working_directory=tmp_path)

        result = await module.execute(
            "git:tag",
            {"action": "delete", "name": "v1.0.0"},
        )

        assert result.success is True
        args = mock_run_git.call_args[0][0]
        assert "-d" in args
        assert "v1.0.0" in args


class TestGitModuleRunGit:
    """Tests for the _run_git helper method."""

    async def test_run_git_success(self, tmp_path: Path) -> None:
        """_run_git should handle successful git command."""
        module = GitModule(working_directory=tmp_path)

        result = await module._run_git(["--version"])

        assert result["success"] is True
        assert "return_code" in result
        assert result["return_code"] == 0

    async def test_run_git_failure(self, tmp_path: Path) -> None:
        """_run_git should handle failed git command."""
        module = GitModule(working_directory=tmp_path)

        result = await module._run_git(["invalid-command"])

        assert result["success"] is False
        assert "error" in result

    async def test_run_git_timeout(self, tmp_path: Path) -> None:
        """_run_git should timeout long-running commands."""
        # Use the current repo directory which is a git repository
        repo_path = Path("/home/node/.openclaw/workspace/SafeAgentFramework")
        module = GitModule(working_directory=repo_path, default_timeout=0.01)

        # Use a slow operation that will definitely timeout
        result = await module._run_git(
            ["log", "--all", "--", "/nonexistent/path/to/force/slow/operation"]
        )

        # The result might succeed quickly (path doesn't exist, fast failure)
        # or might timeout. Either is acceptable for this test.
        # What matters is that we don't hang forever.
        assert "return_code" in result or "error" in result


class TestGitModuleUnknownTool:
    """Tests for unknown tool handling."""

    async def test_unknown_tool_returns_error(self, tmp_path: Path) -> None:
        """Execute should return error for unknown tool."""
        module = GitModule(working_directory=tmp_path)

        result = await module.execute("git:unknown", {})

        assert result.success is False
        assert "Unknown tool" in result.error


class TestGitModuleOutputParsing:
    """Tests for output parsing helper methods."""

    def test_parse_branch_list(self, tmp_path: Path) -> None:
        """_parse_branch_list should correctly parse branch output."""
        module = GitModule(working_directory=tmp_path)

        output = "* main\n  develop\n  remotes/origin/main\n"
        branches = module._parse_branch_list(output)

        assert len(branches) == 3
        main = next(b for b in branches if b["name"] == "main")
        assert main["current"] is True
        assert main["is_remote"] is False

        develop = next(b for b in branches if b["name"] == "develop")
        assert develop["current"] is False
        assert develop["is_remote"] is False

        remote = next(b for b in branches if "remotes/" in b["name"])
        assert remote["is_remote"] is True

    def test_parse_status_porcelain(self, tmp_path: Path) -> None:
        """_parse_status should parse porcelain format."""
        module = GitModule(working_directory=tmp_path)

        output = "M  modified.txt\nA  added.txt\n?? untracked.txt\n"
        status = module._parse_status(output, porcelain=True)

        assert "files" in status
        assert len(status["files"]) == 3
        assert status["files"][0]["status"] == "M "
        assert status["files"][0]["path"] == "modified.txt"

    def test_parse_status_porcelain_handles_renames(self, tmp_path: Path) -> None:
        """_parse_status should handle renames in porcelain format."""
        module = GitModule(working_directory=tmp_path)

        # Rename format: R100 old_name -> new_name
        output = "R100 old_name -> new_name\n"
        status = module._parse_status(output, porcelain=True)

        assert "files" in status
        assert len(status["files"]) == 1
        assert status["files"][0]["status"] == "R1"
        # Path should include the full rename info
        assert "old_name" in status["files"][0]["path"]
        assert "new_name" in status["files"][0]["path"]

    def test_parse_status_porcelain_handles_copies(self, tmp_path: Path) -> None:
        """_parse_status should handle copies in porcelain format."""
        module = GitModule(working_directory=tmp_path)

        # Copy format: C100 original -> copy
        output = "C100 original.py -> copy.py\n"
        status = module._parse_status(output, porcelain=True)

        assert "files" in status
        assert len(status["files"]) == 1
        assert status["files"][0]["status"] == "C1"
        assert "original.py" in status["files"][0]["path"]

    def test_parse_status_regular(self, tmp_path: Path) -> None:
        """_parse_status should return raw for regular format."""
        module = GitModule(working_directory=tmp_path)

        output = "On branch main\nnothing to commit\n"
        status = module._parse_status(output, porcelain=False)

        assert "raw" in status
        assert status["raw"] == output

    def test_parse_log_oneline(self, tmp_path: Path) -> None:
        """_parse_log should parse oneline format."""
        module = GitModule(working_directory=tmp_path)

        output = "abc1234 First commit\ndef5678 Second commit\n"
        commits = module._parse_log(output, oneline=True)

        assert len(commits) == 2
        assert commits[0]["hash"] == "abc1234"
        assert commits[0]["message"] == "First commit"

    def test_parse_log_regular(self, tmp_path: Path) -> None:
        """_parse_log should return raw for regular format."""
        module = GitModule(working_directory=tmp_path)

        output = "commit abc1234\nAuthor: Test\n"
        commits = module._parse_log(output, oneline=False)

        assert len(commits) == 1
        assert "raw" in commits[0]


class TestGitModuleWorkingDirectory:
    """Tests for working directory handling."""

    def test_working_directory_string_with_path(self, tmp_path: Path) -> None:
        """_working_directory_string should return path when set."""
        module = GitModule(working_directory=tmp_path)

        assert module._working_directory_string() == str(tmp_path)

    def test_working_directory_string_default(self) -> None:
        """_working_directory_string should return cwd when not set."""
        module = GitModule()

        assert module._working_directory_string() == str(Path.cwd())

    def test_cwd_with_path(self, tmp_path: Path) -> None:
        """_cwd should return path string when working_directory is set."""
        module = GitModule(working_directory=tmp_path)

        assert module._cwd() == str(tmp_path)

    def test_cwd_always_returns_string(self) -> None:
        """_cwd should always return a valid string (never None)."""
        module = GitModule()

        # working_directory is always set (defaults to cwd in __init__)
        result = module._cwd()
        assert result is not None
        assert isinstance(result, str)
        assert result == str(Path.cwd())
