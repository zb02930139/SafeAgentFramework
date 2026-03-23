"""Tests for the sandboxed filesystem module."""

from pathlib import Path

from safe_agent.modules.filesystem import FilesystemModule


class TestFilesystemModule:
    """Tests for FilesystemModule operations and descriptors."""

    def test_describe_returns_valid_descriptor(self, tmp_path: Path) -> None:
        """describe() should return the expected module metadata."""
        module = FilesystemModule(tmp_path)

        descriptor = module.describe()

        assert descriptor.namespace == "filesystem"
        assert len(descriptor.tools) == 5
        tool_names = {tool.name for tool in descriptor.tools}
        assert tool_names == {
            "filesystem:read_file",
            "filesystem:write_file",
            "filesystem:list_directory",
            "filesystem:delete_file",
            "filesystem:move_file",
        }

    def test_move_file_declares_multi_resource_param(self, tmp_path: Path) -> None:
        """move_file should declare both source and destination resources."""
        module = FilesystemModule(tmp_path)

        descriptor = module.describe()
        move_tool = next(
            tool for tool in descriptor.tools if tool.name == "filesystem:move_file"
        )

        assert move_tool.resource_param == ["source", "destination"]

    async def test_write_and_read_file(self, tmp_path: Path) -> None:
        """write_file and read_file should round-trip content correctly."""
        module = FilesystemModule(tmp_path)

        write_result = await module.execute(
            "filesystem:write_file",
            {"path": "notes/hello.txt", "content": "hello world"},
        )
        read_result = await module.execute(
            "filesystem:read_file",
            {"path": "notes/hello.txt"},
        )

        assert write_result.success is True
        assert read_result.success is True
        assert read_result.data == {
            "path": "notes/hello.txt",
            "content": "hello world",
        }

    async def test_list_directory_non_recursive_and_recursive(
        self,
        tmp_path: Path,
    ) -> None:
        """list_directory should support direct and recursive listing."""
        module = FilesystemModule(tmp_path)
        (tmp_path / "top.txt").write_text("top", encoding="utf-8")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "child.py").write_text("print('x')", encoding="utf-8")

        direct = await module.execute("filesystem:list_directory", {"path": "."})
        recursive = await module.execute(
            "filesystem:list_directory",
            {"path": ".", "recursive": True, "pattern": "*.py"},
        )

        assert direct.success is True
        assert direct.data == {
            "entries": [
                {"path": "nested", "is_directory": True},
                {"path": "top.txt", "is_directory": False},
            ]
        }
        assert recursive.success is True
        assert recursive.data == {
            "entries": [
                {"path": "nested/child.py", "is_directory": False},
            ]
        }

    async def test_delete_file(self, tmp_path: Path) -> None:
        """delete_file should remove an existing file."""
        module = FilesystemModule(tmp_path)
        target = tmp_path / "trash.txt"
        target.write_text("delete me", encoding="utf-8")

        result = await module.execute("filesystem:delete_file", {"path": "trash.txt"})

        assert result.success is True
        assert target.exists() is False

    async def test_move_file(self, tmp_path: Path) -> None:
        """move_file should rename an existing file within the sandbox."""
        module = FilesystemModule(tmp_path)
        source = tmp_path / "from.txt"
        source.write_text("content", encoding="utf-8")

        result = await module.execute(
            "filesystem:move_file",
            {"source": "from.txt", "destination": "to/renamed.txt"},
        )

        assert result.success is True
        assert source.exists() is False
        assert (tmp_path / "to" / "renamed.txt").read_text(
            encoding="utf-8"
        ) == "content"

    async def test_path_traversal_is_blocked_for_read(self, tmp_path: Path) -> None:
        """Traversal attempts should fail with a sandbox error."""
        module = FilesystemModule(tmp_path)

        result = await module.execute(
            "filesystem:read_file",
            {"path": "../../etc/passwd"},
        )

        assert result.success is False
        assert result.error == "Path outside allowed directory"

    async def test_path_traversal_is_blocked_for_move_destination(
        self,
        tmp_path: Path,
    ) -> None:
        """Traversal attempts in move_file should also be blocked."""
        module = FilesystemModule(tmp_path)
        (tmp_path / "safe.txt").write_text("ok", encoding="utf-8")

        result = await module.execute(
            "filesystem:move_file",
            {"source": "safe.txt", "destination": "../../etc/passwd"},
        )

        assert result.success is False
        assert result.error == "Path outside allowed directory"

    async def test_resolve_conditions_for_existing_file(self, tmp_path: Path) -> None:
        """resolve_conditions should derive extension, size, and file flag."""
        module = FilesystemModule(tmp_path)
        target = tmp_path / "config.json"
        target.write_text('{"ok": true}', encoding="utf-8")

        conditions = await module.resolve_conditions(
            "filesystem:read_file",
            {"path": "config.json"},
        )

        assert conditions == {
            "filesystem:FileExtension": ".json",
            "filesystem:FileSize": target.stat().st_size,
            "filesystem:IsDirectory": "false",
        }

    async def test_move_file_conditions_use_destination_path(
        self,
        tmp_path: Path,
    ) -> None:
        """move_file conditions should reflect the destination resource."""
        module = FilesystemModule(tmp_path)

        conditions = await module.resolve_conditions(
            "filesystem:move_file",
            {"source": "notes.txt", "destination": "config.secret"},
        )

        assert conditions == {
            "filesystem:FileExtension": ".secret",
            "filesystem:IsDirectory": "false",
        }

    async def test_resolve_conditions_for_directory(self, tmp_path: Path) -> None:
        """resolve_conditions should mark directories and omit file size."""
        module = FilesystemModule(tmp_path)
        (tmp_path / "docs").mkdir()

        conditions = await module.resolve_conditions(
            "filesystem:list_directory",
            {"path": "docs"},
        )

        assert conditions == {
            "filesystem:FileExtension": "",
            "filesystem:IsDirectory": "true",
        }

    async def test_read_missing_file_is_graceful(self, tmp_path: Path) -> None:
        """Reading a missing file should return a clean tool error."""
        module = FilesystemModule(tmp_path)

        result = await module.execute("filesystem:read_file", {"path": "missing.txt"})

        assert result.success is False
        assert result.error == "File not found"

    async def test_read_directory_is_graceful(self, tmp_path: Path) -> None:
        """Reading a directory should return a clean tool error."""
        module = FilesystemModule(tmp_path)
        (tmp_path / "folder").mkdir()

        result = await module.execute("filesystem:read_file", {"path": "folder"})

        assert result.success is False
        assert result.error == "Path is a directory"

    async def test_list_file_path_is_graceful(self, tmp_path: Path) -> None:
        """Listing a file path should return a directory-specific error."""
        module = FilesystemModule(tmp_path)
        (tmp_path / "file.txt").write_text("x", encoding="utf-8")

        result = await module.execute("filesystem:list_directory", {"path": "file.txt"})

        assert result.success is False
        assert result.error == "Path is not a directory"

    async def test_list_directory_rejects_parent_glob_pattern(
        self,
        tmp_path: Path,
    ) -> None:
        """Glob patterns containing parent components should be rejected."""
        module = FilesystemModule(tmp_path)
        (tmp_path / "nested").mkdir()

        result = await module.execute(
            "filesystem:list_directory",
            {"path": ".", "pattern": "../../etc/*"},
        )

        assert result.success is False
        assert result.error == "Pattern must not contain '..' components"

    async def test_move_file_rejects_existing_destination(self, tmp_path: Path) -> None:
        """move_file should not silently overwrite an existing file."""
        module = FilesystemModule(tmp_path)
        (tmp_path / "source.txt").write_text("source", encoding="utf-8")
        destination = tmp_path / "destination.txt"
        destination.write_text("existing", encoding="utf-8")

        result = await module.execute(
            "filesystem:move_file",
            {"source": "source.txt", "destination": "destination.txt"},
        )

        assert result.success is False
        assert result.error == "Destination file already exists"
        assert destination.read_text(encoding="utf-8") == "existing"

    async def test_write_file_rejects_over_limit(self, tmp_path: Path) -> None:
        """write_file should reject content exceeding max_write_size."""
        small_limit = 100
        module = FilesystemModule(tmp_path, max_write_size=small_limit)
        large_content = "x" * 200

        result = await module.execute(
            "filesystem:write_file",
            {"path": "large.txt", "content": large_content},
        )

        assert result.success is False
        assert "exceeds maximum" in result.error
        assert f"{small_limit} bytes" in result.error

    async def test_write_file_accepts_under_limit(self, tmp_path: Path) -> None:
        """write_file should accept content under max_write_size."""
        limit = 1000
        module = FilesystemModule(tmp_path, max_write_size=limit)
        content = "x" * 500

        result = await module.execute(
            "filesystem:write_file",
            {"path": "within_limit.txt", "content": content},
        )

        assert result.success is True
        assert result.data == {"path": "within_limit.txt", "bytes_written": 500}
