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

"""Pluggable web search module for SafeAgent.

This module provides a web search interface that delegates to an injected backend
provider. The framework defines the interface; a plugin provides the concrete
implementation (Brave Search, Google Custom Search, Bing, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)


@runtime_checkable
class WebSearchBackend(Protocol):
    """Protocol for web search backend implementations.

    Implementations provide concrete integrations with search providers
    such as Brave Search, Google Custom Search, Bing, etc.

    Methods:
        search: Execute a web search and return results.
    """

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute a web search and return results.

        Args:
            query: The search query string.
            **kwargs: Backend-specific options (e.g., max_results, domain_filter).

        Returns:
            List of result dicts with title, url, snippet, etc.
        """
        ...


class WebSearchModule(BaseModule):
    """Web search module using the pluggable adapter pattern.

    This module exposes web search tools to the SafeAgent framework while
    delegating actual search operations to an injected backend provider.

    Attributes:
        _backend: The web search backend implementation (private).
    """

    def __init__(self, backend: WebSearchBackend) -> None:
        """Initialize the web search module with a backend provider.

        Args:
            backend: An implementation of WebSearchBackend that handles
                actual search operations for a specific provider.
        """
        self._backend = backend

    def describe(self) -> ModuleDescriptor:
        """Return the web search module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="web_search",
            description=(
                "Search the web for documentation, CVEs, vendor "
                "advisories, and known issues."
            ),
            tools=[
                ToolDescriptor(
                    name="web_search:search",
                    description="Execute a web search and return results.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string.",
                            },
                            "max_results": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "Maximum number of results to return.",
                            },
                            "domain_filter": {
                                "type": "string",
                                "description": (
                                    "Optional domain to restrict results "
                                    "(e.g., 'docs.python.org')."
                                ),
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                    action="web_search:Search",
                    resource_param=["query"],
                    condition_keys=[
                        "web_search:QueryDomain",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve web search condition values from the domain_filter parameter.

        Derives condition keys for policy evaluation:
            - web_search:QueryDomain: The domain filter if specified.

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the invocation.

        Returns:
            Dict mapping condition keys to resolved values.
        """
        domain_filter = params.get("domain_filter")
        if domain_filter is None:
            return {}

        domain_str = str(domain_filter)
        conditions: dict[str, Any] = {
            "web_search:QueryDomain": domain_str,
        }

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[Any]:
        """Execute a web search tool invocation by delegating to the backend.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure, plus any data.
        """
        try:
            normalized_tool = tool_name.removeprefix("web_search:")
            if normalized_tool == "search":
                return await self._search(params)
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    async def _search(self, params: dict[str, Any]) -> ToolResult[Any]:
        """Delegate search to the backend."""
        if "query" not in params:
            return ToolResult(
                success=False,
                error="Missing required parameter: query",
            )

        query = str(params["query"])
        kwargs: dict[str, Any] = {}
        if "max_results" in params:
            kwargs["max_results"] = int(params["max_results"])
        if "domain_filter" in params:
            kwargs["domain_filter"] = params["domain_filter"]

        results = await self._backend.search(query, **kwargs)
        return ToolResult(success=True, data={"results": results})
