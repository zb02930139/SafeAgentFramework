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

"""Web browsing module for fetching and summarizing web pages."""

from __future__ import annotations

import ipaddress
import logging
import re
from html import unescape as html_unescape
from typing import Any
from urllib.parse import urlparse

import httpx

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Default response size limit (5MB)
DEFAULT_MAX_RESPONSE_SIZE = 5 * 1024 * 1024

# Default timeout in seconds
DEFAULT_TIMEOUT = 30.0

# Maximum allowed timeout (prevent indefinitely long requests)
MAX_TIMEOUT = 300.0

# Maximum text length for summarize_page output
MAX_SUMMARY_TEXT_LENGTH = 100000  # ~100KB of text


def _extract_domain(url: str) -> str:
    """Extract domain from URL, handling IPs, ports, and IDN.

    Args:
        url: The URL string to extract domain from.

    Returns:
        The domain (with port if present), or empty string if parsing fails.
    """
    try:
        parsed = urlparse(url)
        if not parsed.hostname:
            return ""

        # Handle IPv6 addresses - urlparse returns them without brackets
        # but we want to preserve the bracket notation for clarity
        hostname = parsed.hostname
        try:
            # Try to parse as an IPv6 address (handles various formats)
            ipaddress.IPv6Address(hostname)
            # It's a valid IPv6 address - add brackets if not present
            if not hostname.startswith("["):
                hostname = f"[{hostname}]"
        except (ipaddress.AddressValueError, ValueError):
            # Not an IPv6 address, keep hostname as-is
            pass

        # Handle port
        if parsed.port:
            # For IPv6, port comes after the closing bracket
            if hostname.startswith("["):
                domain = f"{hostname}:{parsed.port}"
            else:
                domain = f"{parsed.hostname}:{parsed.port}"
        else:
            domain = hostname

        return domain
    except Exception:
        return ""


def _extract_content_type(content_type_header: str | None) -> str:
    """Extract content type from header, stripping charset and other parameters.

    Args:
        content_type_header: The Content-Type header value.

    Returns:
        The content type (e.g., "text/html"), or empty string if not found.
    """
    if not content_type_header:
        return ""

    # Split on semicolon and take first part
    content_type = content_type_header.split(";")[0].strip().lower()
    return content_type


def _html_to_text(html: str) -> str:
    """Extract text content from HTML, removing tags and scripts.

    This is a simple extraction that:
    - Removes script and style tags with their content
    - Removes all HTML tags
    - Decodes common HTML entities
    - Normalizes whitespace

    Args:
        html: The HTML content to extract text from.

    Returns:
        Extracted text content.
    """
    # Remove script and style tags with content
    text = re.sub(
        r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Replace common block elements with newlines
    text = re.sub(r"<br[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</div>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</td>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</th>", " ", text, flags=re.IGNORECASE)

    # Remove remaining tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities using standard library
    text = html_unescape(text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = text.strip()

    return text


class WebBrowseModule(BaseModule):
    """Provide web page fetching and summarization tools.

    This module exposes tools for:
    - fetch_page: Retrieve raw HTML content from a URL
    - summarize_page: Extract and truncate text content from a URL
    """

    def __init__(
        self,
        default_timeout: float = DEFAULT_TIMEOUT,
        max_response_size: int = DEFAULT_MAX_RESPONSE_SIZE,
        proxy_url: str | None = None,
    ) -> None:
        """Initialize web browsing settings.

        Args:
            default_timeout: Timeout used when a tool call does not override.
                Must be > 0 and <= MAX_TIMEOUT.
            max_response_size: Maximum bytes to read from response (default 5MB).
                Prevents memory exhaustion from large responses.
            proxy_url: Optional forward proxy URL for environments where
                outbound HTTP must route through a corporate proxy.

        Raises:
            ValueError: If default_timeout or max_response_size is invalid.
        """
        if default_timeout <= 0:
            raise ValueError("default_timeout must be > 0")
        if default_timeout > MAX_TIMEOUT:
            raise ValueError(f"default_timeout must be <= {MAX_TIMEOUT} seconds")
        self.default_timeout = default_timeout

        if max_response_size <= 0:
            raise ValueError("max_response_size must be > 0")
        self.max_response_size = max_response_size

        self.proxy_url = proxy_url

        # Create httpx client with timeout
        base_timeout = httpx.Timeout(
            MAX_TIMEOUT, connect=5.0, read=MAX_TIMEOUT, write=MAX_TIMEOUT
        )
        if proxy_url:
            self._client = httpx.AsyncClient(
                proxy=proxy_url,
                follow_redirects=True,
                timeout=base_timeout,
            )
        else:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=base_timeout,
            )

    def describe(self) -> ModuleDescriptor:
        """Return the web_browse module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="web_browse",
            description=(
                "Fetch and summarize web pages with controlled resource limits."
            ),
            tools=[
                ToolDescriptor(
                    name="web_browse:fetch_page",
                    description="Fetch a web page and return its raw HTML content.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to fetch (http:// or https://).",
                            },
                            "timeout": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "description": "Optional timeout in seconds.",
                            },
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                    action="web_browse:FetchPage",
                    resource_param=["url"],
                    condition_keys=[
                        "web_browse:Domain",
                        "web_browse:ContentType",
                    ],
                ),
                ToolDescriptor(
                    name="web_browse:summarize_page",
                    description=(
                        "Fetch a web page and return extracted text content, "
                        "truncated to a reasonable length."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to fetch (http:// or https://).",
                            },
                            "timeout": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "description": "Optional timeout in seconds.",
                            },
                            "max_length": {
                                "type": "integer",
                                "minimum": 1,
                                "description": (
                                    "Maximum characters to return. "
                                    f"Default: {MAX_SUMMARY_TEXT_LENGTH}."
                                ),
                            },
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                    action="web_browse:SummarizePage",
                    resource_param=["url"],
                    condition_keys=[
                        "web_browse:Domain",
                        "web_browse:ContentType",
                    ],
                ),
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve web_browse condition values from URL and response headers.

        Derives condition keys for policy evaluation:
            - web_browse:Domain: Target hostname (with port if present).
            - web_browse:ContentType: Content-Type from response headers.

        Note: ContentType is resolved at execution time when the response
        is received. This method extracts domain from URL params only.

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the invocation.

        Returns:
            Dict mapping condition keys to resolved values.
        """
        normalized_tool = tool_name.removeprefix("web_browse:")
        if normalized_tool not in ("fetch_page", "summarize_page"):
            return {}

        url = str(params.get("url", ""))
        domain = _extract_domain(url)

        return {
            "web_browse:Domain": domain,
            "web_browse:ContentType": "",  # Resolved at execution time
        }

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Execute a web browsing tool invocation.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure with response data.
        """
        normalized_tool = tool_name.removeprefix("web_browse:")

        if normalized_tool == "fetch_page":
            return await self._fetch_page(params)
        if normalized_tool == "summarize_page":
            return await self._summarize_page(params)

        return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

    async def _fetch_page(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Fetch a web page and return raw HTML."""
        url = params.get("url")
        if not url or not isinstance(url, str):
            return ToolResult(
                success=False, error="url is required and must be a string"
            )

        # Validate URL scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                success=False,
                error=(
                    f"Invalid URL scheme: {parsed.scheme!r}. Only http/https allowed."
                ),
            )
        if not parsed.hostname:
            return ToolResult(success=False, error="URL must include a hostname")

        # Get timeout
        timeout = self._get_timeout(params)
        if timeout is None:
            return ToolResult(
                success=False,
                error="timeout must be a number > 0",
            )

        try:
            request = self._client.build_request("GET", url, timeout=timeout)
            response = await self._client.send(request, stream=True)

            try:
                body_bytes, truncated = await self._read_response_with_limit(response)

                # Get content type from response headers
                content_type = _extract_content_type(
                    response.headers.get("content-type")
                )
                domain = _extract_domain(url)

                # Try to decode as text
                try:
                    body_text = body_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    # Try other common encodings
                    try:
                        body_text = body_bytes.decode("latin-1")
                    except UnicodeDecodeError:
                        body_text = body_bytes.decode("utf-8", errors="replace")

                result: dict[str, Any] = {
                    "url": str(response.url),
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "domain": domain,
                    "html": body_text,
                    "bytes_read": len(body_bytes),
                }

                if truncated:
                    result["truncated"] = True
                    result["max_response_size"] = self.max_response_size

                return ToolResult(success=True, data=result)

            finally:
                await response.aclose()

        except httpx.TimeoutException:
            return ToolResult(success=False, error="Request timed out")
        except httpx.ConnectError as exc:
            return ToolResult(success=False, error=f"Connection failed: {exc}")
        except Exception as exc:
            logger.error("Fetch page failed: %s", exc)
            return ToolResult(success=False, error=str(exc))

    async def _summarize_page(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Fetch a web page and return extracted text content."""
        url = params.get("url")
        if not url or not isinstance(url, str):
            return ToolResult(
                success=False, error="url is required and must be a string"
            )

        # Validate URL scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult(
                success=False,
                error=(
                    f"Invalid URL scheme: {parsed.scheme!r}. Only http/https allowed."
                ),
            )
        if not parsed.hostname:
            return ToolResult(success=False, error="URL must include a hostname")

        # Get timeout and max_length
        timeout = self._get_timeout(params)
        if timeout is None:
            return ToolResult(
                success=False,
                error="timeout must be a number > 0",
            )
        max_length = params.get("max_length", MAX_SUMMARY_TEXT_LENGTH)
        if not isinstance(max_length, int) or max_length < 1:
            max_length = MAX_SUMMARY_TEXT_LENGTH

        try:
            request = self._client.build_request("GET", url, timeout=timeout)
            response = await self._client.send(request, stream=True)

            try:
                body_bytes, truncated = await self._read_response_with_limit(response)

                # Get content type from response headers
                content_type = _extract_content_type(
                    response.headers.get("content-type")
                )
                domain = _extract_domain(url)

                # Decode and extract text
                try:
                    html = body_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        html = body_bytes.decode("latin-1")
                    except UnicodeDecodeError:
                        html = body_bytes.decode("utf-8", errors="replace")

                text = _html_to_text(html)

                # Truncate text if needed
                text_truncated = False
                if len(text) > max_length:
                    text = text[:max_length]
                    text_truncated = True

                result: dict[str, Any] = {
                    "url": str(response.url),
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "domain": domain,
                    "text": text,
                    "bytes_read": len(body_bytes),
                    "text_length": len(text),
                }

                if truncated:
                    result["response_truncated"] = True
                    result["max_response_size"] = self.max_response_size

                if text_truncated:
                    result["text_truncated"] = True
                    result["max_text_length"] = max_length

                return ToolResult(success=True, data=result)

            finally:
                await response.aclose()

        except httpx.TimeoutException:
            return ToolResult(success=False, error="Request timed out")
        except httpx.ConnectError as exc:
            return ToolResult(success=False, error=f"Connection failed: {exc}")
        except Exception as exc:
            logger.error("Summarize page failed: %s", exc)
            return ToolResult(success=False, error=str(exc))

    def _get_timeout(self, params: dict[str, Any]) -> float | None:
        """Get timeout from params or use default.

        Args:
            params: The tool parameters.

        Returns:
            Timeout value clamped to MAX_TIMEOUT, or None if invalid.
            Returns None if timeout is not a valid number or is <= 0.

        Raises:
            Nothing - callers must check for None return value.
        """
        raw_timeout = params.get("timeout", self.default_timeout)
        try:
            timeout = float(raw_timeout)
        except (ValueError, TypeError):
            return None  # Signal error

        if timeout <= 0:
            return None  # Signal error

        return min(timeout, MAX_TIMEOUT)

    async def _read_response_with_limit(
        self,
        response: httpx.Response,
    ) -> tuple[bytes, bool]:
        """Read response body with configurable size limit.

        Reads incrementally to avoid loading large responses into memory.

        Args:
            response: The httpx response object (streaming).

        Returns:
            Tuple of (body_bytes, truncated). truncated=True if limit was hit.
        """
        chunks: list[bytes] = []
        total_bytes = 0
        truncated = False

        async for chunk in response.aiter_bytes(chunk_size=65536):
            if total_bytes >= self.max_response_size:
                truncated = True
                break

            available = self.max_response_size - total_bytes
            if len(chunk) > available:
                chunk = chunk[:available]
                truncated = True

            chunks.append(chunk)
            total_bytes += len(chunk)

            if truncated:
                break

        return b"".join(chunks), truncated

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
