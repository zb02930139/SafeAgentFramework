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

"""Tests for the pluggable web search module."""

from typing import Any

from safe_agent.modules.web.search import WebSearchBackend, WebSearchModule


class MockWebSearchBackend:
    """Mock backend for testing WebSearchModule."""

    def __init__(self) -> None:
        """Initialize mock backend with storage for search results."""
        self.search_results: list[dict[str, Any]] = []
        # Track kwargs passed to search
        self.last_search_kwargs: dict[str, Any] = {}

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Mock search that returns stored results."""
        self.last_search_kwargs = kwargs  # Track what kwargs were passed
        return self.search_results


class TestWebSearchModule:
    """Tests for WebSearchModule operations and descriptors."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        backend = MockWebSearchBackend()
        module = WebSearchModule(backend)

        descriptor = module.describe()

        assert descriptor.namespace == "web_search"
        assert len(descriptor.tools) == 1
        tool_names = {tool.name for tool in descriptor.tools}
        assert tool_names == {"web_search:search"}

    def test_describe_tools_have_correct_actions_and_resource_params(
        self,
    ) -> None:
        """Tools should have correct action names and resource params."""
        backend = MockWebSearchBackend()
        module = WebSearchModule(backend)

        descriptor = module.describe()

        search_tool = next(
            tool for tool in descriptor.tools if tool.name == "web_search:search"
        )

        assert search_tool.action == "web_search:Search"
        assert search_tool.resource_param == ["query"]
        assert search_tool.condition_keys == ["web_search:QueryDomain"]

    async def test_search_delegates_to_backend(self) -> None:
        """Search should delegate to the backend and return success."""
        backend = MockWebSearchBackend()
        backend.search_results = [
            {
                "title": "Result 1",
                "url": "https://example.com/1",
                "snippet": "First",
            },
            {
                "title": "Result 2",
                "url": "https://example.com/2",
                "snippet": "Second",
            },
        ]
        module = WebSearchModule(backend)

        result = await module.execute(
            "web_search:search",
            {"query": "test query"},
        )

        assert result.success is True
        assert result.data == {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "snippet": "First",
                },
                {
                    "title": "Result 2",
                    "url": "https://example.com/2",
                    "snippet": "Second",
                },
            ]
        }

    async def test_search_with_max_results_and_domain_filter(self) -> None:
        """Search should pass through optional max_results and domain_filter."""
        backend = MockWebSearchBackend()
        backend.search_results = [
            {
                "title": "Python Docs",
                "url": "https://docs.python.org/3",
                "snippet": "Python",
            },
        ]
        module = WebSearchModule(backend)

        result = await module.execute(
            "web_search:search",
            {
                "query": "python async",
                "max_results": 5,
                "domain_filter": "docs.python.org",
            },
        )

        assert result.success is True
        assert result.data == {
            "results": [
                {
                    "title": "Python Docs",
                    "url": "https://docs.python.org/3",
                    "snippet": "Python",
                },
            ]
        }
        # Verify the kwargs were actually forwarded to the backend
        assert backend.last_search_kwargs.get("max_results") == 5
        assert backend.last_search_kwargs.get("domain_filter") == "docs.python.org"

    async def test_execute_returns_error_for_unknown_tool(self) -> None:
        """Execute should return error for unknown tool names."""
        backend = MockWebSearchBackend()
        module = WebSearchModule(backend)

        result = await module.execute(
            "web_search:unknown_tool",
            {"query": "test"},
        )

        assert result.success is False
        assert result.error == "Unknown tool: web_search:unknown_tool"

    async def test_backend_error_propagates_as_tool_result_error(self) -> None:
        """Backend errors should be caught and returned as ToolResult errors."""

        class FailingBackend(MockWebSearchBackend):
            async def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
                raise RuntimeError("API rate limit exceeded")

        backend = FailingBackend()
        module = WebSearchModule(backend)

        result = await module.execute(
            "web_search:search",
            {"query": "test"},
        )

        assert result.success is False
        assert result.error == "API rate limit exceeded"

    async def test_search_missing_query_returns_error(self) -> None:
        """Search without required query param should return clear error."""
        backend = MockWebSearchBackend()
        module = WebSearchModule(backend)

        result = await module.execute(
            "web_search:search",
            {},  # Missing query
        )

        assert result.success is False
        assert result.error == "Missing required parameter: query"

    async def test_resolve_conditions_derives_query_domain(self) -> None:
        """Resolve web_search:QueryDomain from domain_filter."""
        backend = MockWebSearchBackend()
        module = WebSearchModule(backend)

        conditions = await module.resolve_conditions(
            "web_search:search",
            {"query": "python async", "domain_filter": "docs.python.org"},
        )

        assert conditions == {
            "web_search:QueryDomain": "docs.python.org",
        }

    async def test_resolve_conditions_returns_empty_if_no_domain_filter(
        self,
    ) -> None:
        """Return empty dict when domain_filter not present."""
        backend = MockWebSearchBackend()
        module = WebSearchModule(backend)

        conditions = await module.resolve_conditions(
            "web_search:search",
            {"query": "python async"},
        )

        assert conditions == {}

    def test_backend_protocol_check(self) -> None:
        """MockWebSearchBackend should satisfy the WebSearchBackend protocol."""
        backend = MockWebSearchBackend()
        assert isinstance(backend, WebSearchBackend)
