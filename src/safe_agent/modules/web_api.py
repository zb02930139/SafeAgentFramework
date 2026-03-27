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

"""Web API module for raw HTTP/API calls to external services."""

from __future__ import annotations

import base64
import ipaddress
import logging
import re
import socket
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from safe_agent.modules.base import (
    BaseModule,
    ModuleDescriptor,
    ToolDescriptor,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Valid HTTP methods supported by the module
_VALID_HTTP_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE"}
)

# Valid URL schemes (SSRF protection: only http/https)
_VALID_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# Header name validation pattern (RFC 7230 compliant subset)
# Must start with letter, can contain letters, digits, hyphens
_HEADER_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9\-]*$")

# Header value CRLF pattern (injection protection)
_HEADER_CRLF_PATTERN = re.compile(r"[\r\n]")

# Response size limit default (5MB)
_DEFAULT_MAX_RESPONSE_SIZE = 5 * 1024 * 1024

# Default timeout in seconds
_DEFAULT_TIMEOUT = 30.0

# Maximum allowed timeout (prevent indefinitely long requests)
_MAX_TIMEOUT = 300.0


def _is_valid_header_name(name: str) -> bool:
    """Validate header name against RFC 7230 subset.

    Args:
        name: The header name to validate.

    Returns:
        True if the header name is valid, False otherwise.
    """
    if not name:
        return False
    return bool(_HEADER_NAME_PATTERN.match(name))


def _is_internal_ip(host: str) -> bool:
    """Check if a hostname is an internal/private IP address (SSRF protection).

    Supports IPv4, IPv6, and hostnames that resolve to internal IPs.
    Performs DNS resolution to prevent DNS rebinding attacks.

    Args:
        host: The hostname or IP address to check.

    Returns:
        True if the host is a private/internal IP, False for public.
    """
    # Remove port if present (handle IPv6 correctly)
    host_part = host
    if ":" in host:
        # IPv6 addresses have colons; check if it's IPv6 format
        if host.startswith("["):
            # IPv6 with port: [::1]:8080
            if "]:" in host:
                host_part = host.rsplit(":", 1)[0].strip("[]")
            else:
                host_part = host.strip("[]")
        elif host.count(":") == 1:
            # IPv4 with port: 127.0.0.1:8080
            host_part = host.rsplit(":", 1)[0]
        # else: IPv6 without port, keep as-is

    # Check if it's a direct IP address
    try:
        addr = ipaddress.ip_address(host_part)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        pass

    # Resolve hostname via DNS to prevent DNS rebinding attacks
    try:
        # getaddrinfo returns list of (family, type, proto, canonname, sockaddr)
        # sockaddr is (ip, port) for IPv4 or (ip, port, flow, scope) for IPv6
        addr_info = socket.getaddrinfo(host_part, None)
        for _family, _type, _proto, _canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                addr = ipaddress.ip_address(ip_str)
                # Block private, loopback, link-local, and reserved addresses
                if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                    return True
            except ValueError:
                continue
    except (socket.gaierror, socket.herror):
        # DNS resolution failed - can't verify, let request proceed
        # and fail naturally at the network layer
        pass

    return False


def _validate_url(url: str) -> tuple[str, str, str]:
    """Validate URL for SSRF protection and return domain, path components.

    Args:
        url: The URL string to validate.

    Returns:
        Tuple of (domain, path, raw_path).

    Raises:
        ValueError: If URL scheme is invalid or SSRF checks fail.
    """
    parsed = urlparse(url)

    # Scheme validation (SSRF protection)
    if parsed.scheme not in _VALID_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: {parsed.scheme!r}. "
            f"Only {', '.join(sorted(_VALID_SCHEMES))} allowed."
        )

    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    # Handle port in domain
    if parsed.port:
        domain = f"{parsed.hostname}:{parsed.port}"
    else:
        domain = parsed.hostname

    # Block internal IP addresses (SSRF protection)
    if _is_internal_ip(parsed.hostname):
        raise ValueError("Access to internal IP addresses is not allowed")

    # Path handling
    path = parsed.path if parsed.path else "/"
    raw_path = parsed.path if parsed.path else "/"
    decoded_path = unquote(path)

    return domain, decoded_path, raw_path


def _validate_method(method: str) -> str:
    """Validate and normalize HTTP method.

    Args:
        method: The HTTP method string to validate.

    Returns:
        Uppercase validated method string.

    Raises:
        ValueError: If method is not in allowed set.
    """
    normalized = method.upper().strip()
    if normalized not in _VALID_HTTP_METHODS:
        valid_methods = ", ".join(sorted(_VALID_HTTP_METHODS))
        raise ValueError(
            f"Invalid HTTP method: {method!r}. Valid methods: {valid_methods}"
        )
    return normalized


class WebApiModule(BaseModule):
    """Provide controlled HTTP/API calls to external services.

    This module exposes tools for making raw HTTP requests with tight
    controls over URL parsing, method validation, header sanitization,
    response size limits, and timeouts.
    """

    def __init__(
        self,
        default_timeout: float = _DEFAULT_TIMEOUT,
        max_response_size: int = _DEFAULT_MAX_RESPONSE_SIZE,
        allowed_methods: list[str] | None = None,
        proxy_url: str | None = None,
        follow_redirects: bool = False,
    ) -> None:
        """Initialize web API execution settings.

        Args:
            default_timeout: Timeout used when a tool call does not override.
                Must be > 0 and <= _MAX_TIMEOUT.
            max_response_size: Maximum bytes to read from response (default 5MB).
                Prevents memory exhaustion from large responses.
            allowed_methods: Optional whitelist of HTTP methods. If None,
                all standard methods (GET, POST, PUT, PATCH, DELETE) are allowed.
            proxy_url: Optional forward proxy URL for environments where
                outbound HTTP must route through a corporate proxy.
            follow_redirects: Whether to follow HTTP redirects. Default False
                for security - redirects could bypass policy checks to
                internal addresses. When enabled, redirects are followed
                without policy re-evaluation (known limitation).
        """
        if default_timeout <= 0:
            raise ValueError("default_timeout must be > 0")
        if default_timeout > _MAX_TIMEOUT:
            raise ValueError(f"default_timeout must be <= {_MAX_TIMEOUT} seconds")
        self.default_timeout = default_timeout

        if max_response_size <= 0:
            raise ValueError("max_response_size must be > 0")
        self.max_response_size = max_response_size

        # Validate and store allowed methods
        if allowed_methods is not None:
            normalized = set()
            for m in allowed_methods:
                normalized.add(m.upper().strip())
            invalid = normalized - _VALID_HTTP_METHODS
            if invalid:
                valid_list = ", ".join(sorted(_VALID_HTTP_METHODS))
                raise ValueError(
                    f"Invalid methods in allowed_methods: {invalid}. "
                    f"Valid: {valid_list}"
                )
            self.allowed_methods: frozenset[str] = frozenset(normalized)
        else:
            self.allowed_methods = _VALID_HTTP_METHODS

        self.proxy_url = proxy_url
        self.follow_redirects = follow_redirects

        # httpx.Timeout accepts: a number (total), or each component
        # Use _MAX_TIMEOUT seconds total, with a shorter connect phase
        base_timeout = httpx.Timeout(
            _MAX_TIMEOUT, connect=5.0, read=_MAX_TIMEOUT, write=_MAX_TIMEOUT
        )
        if proxy_url:
            self._client = httpx.AsyncClient(
                proxy=proxy_url,
                follow_redirects=follow_redirects,
                timeout=base_timeout,
            )
        else:
            self._client = httpx.AsyncClient(
                follow_redirects=follow_redirects,
                timeout=base_timeout,
            )

    def describe(self) -> ModuleDescriptor:
        """Return the web_api module descriptor and tool definitions."""
        return ModuleDescriptor(
            namespace="web_api",
            description=(
                "Make raw HTTP/API calls to external services with "
                "controlled security and resource limits."
            ),
            tools=[
                ToolDescriptor(
                    name="web_api:http_request",
                    description=(
                        "Make an HTTP request to an API endpoint with "
                        "configurable method, headers, body, and timeout."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Target URL (http:// or https://).",
                            },
                            "method": {
                                "type": "string",
                                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                                "description": "HTTP method to use.",
                            },
                            "headers": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                                "description": "Optional HTTP headers.",
                            },
                            "body": {
                                "type": "string",
                                "description": "Optional request body.",
                            },
                            "timeout": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": int(_MAX_TIMEOUT),
                                "description": "Request timeout in seconds.",
                            },
                            "content_type": {
                                "type": "string",
                                "description": (
                                    "Content-Type header value "
                                    "(e.g., application/json)."
                                ),
                            },
                        },
                        "required": ["url", "method"],
                        "additionalProperties": False,
                    },
                    action="web_api:HttpRequest",
                    resource_param=["url"],
                    condition_keys=[
                        "web_api:Method",
                        "web_api:Domain",
                        "web_api:Path",
                    ],
                )
            ],
        )

    async def resolve_conditions(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve web_api condition values from URL and method.

        Derives condition keys for policy evaluation:
            - web_api:Method: Uppercase HTTP method from params.
            - web_api:Domain: Target hostname (with port if present).
            - web_api:Path: URL path component (normalized, decoded).

        Args:
            tool_name: The name of the tool being invoked.
            params: The input parameters for the invocation.

        Returns:
            Dict mapping condition keys to resolved values.
        """
        if tool_name.removeprefix("web_api:") != "http_request":
            return {}

        url = str(params.get("url", ""))
        method = str(params.get("method", "GET")).upper().strip()

        conditions: dict[str, Any] = {}

        # Method condition
        if method in _VALID_HTTP_METHODS:
            conditions["web_api:Method"] = method

        # URL validation and parsing for domain/path
        try:
            domain, path, _ = _validate_url(url)
            conditions["web_api:Domain"] = domain
            conditions["web_api:Path"] = path
        except ValueError:
            # Invalid URL - still report method, empty domain/path
            conditions["web_api:Domain"] = ""
            conditions["web_api:Path"] = ""

        return conditions

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Execute an HTTP request and return the result.

        Args:
            tool_name: The name of the tool to execute.
            params: The input parameters for the tool.

        Returns:
            A ToolResult indicating success or failure with response data.
        """
        if tool_name.removeprefix("web_api:") != "http_request":
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")

        try:
            return await self._http_request(params)
        except Exception as exc:
            logger.error("HTTP request failed: %s", exc)
            return ToolResult(success=False, error=str(exc))

    async def _http_request(
        self,
        params: dict[str, Any],
    ) -> ToolResult[dict[str, Any]]:
        """Execute the HTTP request with validation and size limits."""
        # Validate required params
        url = params.get("url")
        if not url or not isinstance(url, str):
            return ToolResult(
                success=False, error="url is required and must be a string"
            )

        method_raw = params.get("method", "GET")
        if not isinstance(method_raw, str):
            return ToolResult(success=False, error="method must be a string")

        # Validate HTTP method
        try:
            method = _validate_method(method_raw)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        # Check method against allowed whitelist
        if method not in self.allowed_methods:
            allowed = ", ".join(sorted(self.allowed_methods))
            return ToolResult(
                success=False,
                error=f"Method {method!r} not allowed. Allowed: {allowed}",
            )

        # Validate URL for SSRF protection and parse domain/path
        try:
            _domain, _path, _raw_path = _validate_url(url)
        except ValueError as exc:
            return ToolResult(success=False, error=f"Invalid URL: {exc}")

        # Build headers
        headers: dict[str, str] = {}
        raw_headers = params.get("headers")
        if raw_headers is not None:
            if not isinstance(raw_headers, dict):
                return ToolResult(
                    success=False,
                    error="headers must be an object of string key/value pairs",
                )
            for key, value in raw_headers.items():
                key_str = str(key)
                if not _is_valid_header_name(key_str):
                    # Log security concern
                    logger.warning("Rejected invalid header name: %r", key_str)
                    continue
                value_str = str(value)
                # Check for CRLF injection in header values
                if _HEADER_CRLF_PATTERN.search(value_str):
                    logger.warning(
                        "Rejected header value with CRLF injection: %r",
                        key_str,
                    )
                    continue
                headers[key_str] = value_str

        # Add Content-Type if body is present and content_type specified
        content_type = params.get("content_type")
        if content_type and isinstance(content_type, str):
            headers["Content-Type"] = content_type

        # Get body
        body = params.get("body")
        content: str | bytes | None = None
        if body is not None:
            content = str(body)

        # Determine timeout
        timeout = self.default_timeout
        raw_timeout = params.get("timeout")
        if raw_timeout is not None:
            try:
                requested = float(raw_timeout)
                if requested <= 0:
                    return ToolResult(success=False, error="timeout must be > 0")
                # Clamp to max
                timeout = min(requested, _MAX_TIMEOUT)
            except (ValueError, TypeError):
                return ToolResult(success=False, error="timeout must be a number")

        # Execute request with size limiting
        try:
            response = await self._execute_request(
                url=url,
                method=method,
                headers=headers,
                content=content,
                timeout=timeout,
            )
            return response
        except httpx.TimeoutException:
            return ToolResult(success=False, error="Request timed out")
        except httpx.ConnectError as exc:
            return ToolResult(success=False, error=f"Connection failed: {exc}")
        except httpx.HTTPStatusError as exc:
            # Capture response body even on error status
            error_body = await self._read_error_body(exc.response)
            response_data = self._build_response_data(
                exc.response,
                truncated=error_body[2],  # Truncated flag
                bytes_read=error_body[1],  # Bytes read
            )
            response_data["body"] = error_body[0]  # Body content
            response_data["error_status"] = exc.response.status_code
            return ToolResult(
                success=False,
                error=f"HTTP error: {exc.response.status_code}",
                data=response_data,
            )
        except Exception as exc:
            logger.error("HTTP request exception: %s", exc)
            return ToolResult(success=False, error=str(exc))

    async def _execute_request(
        self,
        url: str,
        method: str,
        headers: dict[str, str],
        content: str | bytes | None,
        timeout: float,
    ) -> ToolResult[dict[str, Any]]:
        """Execute the HTTP request with response size limiting."""
        # Create request with specific timeout
        request = self._client.build_request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            timeout=timeout,
        )

        # Send request
        response = await self._client.send(request, stream=True)

        try:
            # Read response with size limit
            body_bytes, truncated = await self._read_response_with_limit(response)

            # Build result
            response_data = self._build_response_data(
                response,
                truncated=truncated,
                bytes_read=len(body_bytes),
            )

            # Try to decode body as text
            try:
                body_text = body_bytes.decode("utf-8")
                response_data["body"] = body_text
            except UnicodeDecodeError:
                # Return as base64 for binary content
                response_data["body_b64"] = base64.b64encode(body_bytes).decode("ascii")
                response_data["encoding"] = "base64"

            return ToolResult(success=True, data=response_data)

        finally:
            await response.aclose()

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

    def _build_response_data(
        self,
        response: httpx.Response,
        truncated: bool,
        bytes_read: int,
    ) -> dict[str, Any]:
        """Build the response data structure."""
        result: dict[str, Any] = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "url": str(response.url),
            "bytes_read": bytes_read,
        }

        if truncated:
            result["truncated"] = True
            result["max_response_size"] = self.max_response_size

        return result

    async def _read_error_body(
        self,
        response: httpx.Response,
    ) -> tuple[str, int, bool]:
        """Read error response body with size limiting.

        Args:
            response: The httpx error response.

        Returns:
            Tuple of (body_text, bytes_read, truncated).
        """
        body_bytes, truncated = await self._read_response_with_limit(response)
        total_bytes = len(body_bytes)

        # Try to decode as text, fall back to base64 for binary
        try:
            body_text = body_bytes.decode("utf-8")
        except UnicodeDecodeError:
            body_text = base64.b64encode(body_bytes).decode("ascii")

        return body_text, total_bytes, truncated

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
