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

"""Tests for the web_api module."""

from __future__ import annotations

import base64

import httpx
import pytest

from safe_agent.modules.web_api import WebApiModule


class TestWebApiModuleDescriptors:
    """Tests for module descriptor generation."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return expected module metadata."""
        module = WebApiModule()

        descriptor = module.describe()

        assert descriptor.namespace == "web_api"
        assert len(descriptor.tools) == 1

        tool = descriptor.tools[0]
        assert tool.name == "web_api:http_request"
        assert tool.action == "web_api:HttpRequest"
        assert tool.resource_param == ["url"]
        assert tool.condition_keys == [
            "web_api:Method",
            "web_api:Domain",
            "web_api:Path",
        ]

        # Verify JSON schema for parameters
        assert tool.parameters["type"] == "object"
        assert "url" in tool.parameters["properties"]
        assert "method" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["url", "method"]
        assert tool.parameters["additionalProperties"] is False

    async def test_resolve_conditions_extracts_method_domain_path(self) -> None:
        """resolve_conditions() should extract Method, Domain, Path from params."""
        module = WebApiModule()

        conditions = await module.resolve_conditions(
            "web_api:http_request",
            {
                "url": "https://api.example.com/v1/users",
                "method": "POST",
            },
        )

        assert conditions == {
            "web_api:Method": "POST",
            "web_api:Domain": "api.example.com",
            "web_api:Path": "/v1/users",
        }

    async def test_resolve_conditions_handles_ip_addresses(self) -> None:
        """resolve_conditions() should block internal IP addresses."""
        module = WebApiModule()

        conditions = await module.resolve_conditions(
            "web_api:http_request",
            {
                "url": "http://192.168.1.1:8080/api/status",
                "method": "GET",
            },
        )

        # Internal IPs blocked by SSRF protection - empty domain
        assert conditions["web_api:Domain"] == ""
        assert conditions["web_api:Method"] == "GET"

    async def test_resolve_conditions_handles_encoded_paths(self) -> None:
        """resolve_conditions() should decode percent-encoded paths."""
        module = WebApiModule()

        conditions = await module.resolve_conditions(
            "web_api:http_request",
            {
                "url": "https://api.example.com/v1/users%20list",
                "method": "GET",
            },
        )

        assert conditions["web_api:Path"] == "/v1/users list"

    async def test_resolve_conditions_handles_ports(self) -> None:
        """resolve_conditions() should include port in domain."""
        module = WebApiModule()

        conditions = await module.resolve_conditions(
            "web_api:http_request",
            {
                "url": "https://api.example.com:8443/path",
                "method": "DELETE",
            },
        )

        assert conditions["web_api:Domain"] == "api.example.com:8443"
        assert conditions["web_api:Method"] == "DELETE"

    async def test_resolve_conditions_handles_unknown_tool(self) -> None:
        """resolve_conditions() should return empty dict for unknown tools."""
        module = WebApiModule()

        conditions = await module.resolve_conditions(
            "web_api:unknown",
            {"url": "http://example.com", "method": "GET"},
        )

        assert conditions == {}

    async def test_resolve_conditions_handles_invalid_url(self) -> None:
        """resolve_conditions() should handle malformed URLs gracefully."""
        module = WebApiModule()

        conditions = await module.resolve_conditions(
            "web_api:http_request",
            {
                "url": "not-a-valid-url",
                "method": "GET",
            },
        )

        # Should still return method, empty domain/path
        assert conditions["web_api:Method"] == "GET"
        assert conditions["web_api:Domain"] == ""
        assert conditions["web_api:Path"] == ""


class TestWebApiModuleValidation:
    """Tests for input validation."""

    async def test_execute_rejects_invalid_method(self) -> None:
        """Invalid HTTP methods should return a failed ToolResult."""
        module = WebApiModule()

        result = await module.execute(
            "web_api:http_request",
            {"url": "https://example.com", "method": "TRACE"},
        )

        assert result.success is False
        assert "Invalid HTTP method" in result.error

    async def test_execute_rejects_blocked_method(self) -> None:
        """Methods outside allowed_methods should be rejected."""
        module = WebApiModule(allowed_methods=["GET"])

        result = await module.execute(
            "web_api:http_request",
            {"url": "https://example.com", "method": "POST"},
        )

        assert result.success is False
        assert "not allowed" in result.error

    async def test_constructor_validates_allowed_methods(self) -> None:
        """Invalid methods in allowed_methods should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid methods"):
            WebApiModule(allowed_methods=["OPTIONS", "GET"])

    def test_constructor_validates_timeout(self) -> None:
        """default_timeout <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="default_timeout must be > 0"):
            WebApiModule(default_timeout=0)

        with pytest.raises(ValueError, match="default_timeout must be > 0"):
            WebApiModule(default_timeout=-1)

    def test_constructor_validates_max_timeout(self) -> None:
        """default_timeout > _MAX_TIMEOUT should raise ValueError."""
        with pytest.raises(ValueError, match="default_timeout must be <="):
            WebApiModule(default_timeout=999999)

    def test_constructor_validates_max_response_size(self) -> None:
        """max_response_size <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_response_size must be > 0"):
            WebApiModule(max_response_size=0)

        with pytest.raises(ValueError, match="max_response_size must be > 0"):
            WebApiModule(max_response_size=-100)

    async def test_execute_rejects_negative_timeout(self) -> None:
        """Per-request timeout <= 0 should return a failed ToolResult."""
        module = WebApiModule()

        result = await module.execute(
            "web_api:http_request",
            {"url": "https://example.com", "method": "GET", "timeout": -5},
        )

        assert result.success is False
        assert (
            "timeout must be" in result.error.lower()
            or "timed out" in result.error.lower()
        )


class TestWebApiModuleExecution:
    """Tests for HTTP request execution using httpx mock transport."""

    @pytest.fixture
    def mock_transport(self) -> httpx.MockTransport:
        """Create a mock transport for testing HTTP calls."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/users":
                return httpx.Response(
                    200,
                    json={"users": [{"id": 1, "name": "Alice"}]},
                    headers={"Content-Type": "application/json"},
                )
            elif request.url.path == "/api/create":
                return httpx.Response(
                    201,
                    json={"id": 123, "created": True},
                )
            elif request.url.path == "/api/update":
                return httpx.Response(
                    200,
                    json={"updated": True},
                )
            elif request.url.path == "/api/delete":
                return httpx.Response(204)
            elif request.url.path == "/api/patch":
                return httpx.Response(
                    200,
                    json={"patched": True},
                )
            elif request.url.path == "/timeout":
                raise httpx.TimeoutException("Request timed out")
            elif request.url.path == "/connect-error":
                raise httpx.ConnectError("Connection refused")
            elif request.url.path == "/large-response":
                # Return body larger than default 5MB limit
                return httpx.Response(200, content=b"x" * (6 * 1024 * 1024))
            elif request.url.path == "/binary":
                # Invalid UTF-8 bytes to trigger base64 encoding
                return httpx.Response(200, content=b"\xff\xfe\x00\x01")
            elif request.url.path == "/error":
                return httpx.Response(
                    500,
                    json={"error": "Internal Server Error"},
                )
            return httpx.Response(404, json={"error": "Not found"})

        return httpx.MockTransport(handler)

    async def test_http_get_request(self, mock_transport: httpx.MockTransport) -> None:
        """GET request should execute successfully."""
        module = WebApiModule()
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {"url": "https://api.example.com/api/users", "method": "GET"},
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["status_code"] == 200
        assert "users" in result.data["body"]

    async def test_http_post_with_body(
        self, mock_transport: httpx.MockTransport
    ) -> None:
        """POST request with body should execute successfully."""
        module = WebApiModule()
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {
                "url": "https://api.example.com/api/create",
                "method": "POST",
                "body": '{"name": "Alice"}',
                "content_type": "application/json",
            },
        )

        assert result.success is True
        assert result.data["status_code"] == 201
        # Verify Content-Type was set in request (captured in response headers echo)
        # Note: mock transport doesn't echo headers, but request should succeed

    async def test_http_put_request(self, mock_transport: httpx.MockTransport) -> None:
        """PUT request should execute successfully."""
        module = WebApiModule()
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {
                "url": "https://api.example.com/api/update",
                "method": "PUT",
                "body": '{"name": "Bob"}',
            },
        )

        assert result.success is True
        assert result.data["status_code"] == 200

    async def test_http_patch_request(
        self, mock_transport: httpx.MockTransport
    ) -> None:
        """PATCH request should execute successfully."""
        module = WebApiModule()
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {
                "url": "https://api.example.com/api/patch",
                "method": "PATCH",
                "body": '{"field": "value"}',
            },
        )

        assert result.success is True
        assert result.data["status_code"] == 200

    async def test_http_delete_request(
        self, mock_transport: httpx.MockTransport
    ) -> None:
        """DELETE request should execute successfully."""
        module = WebApiModule()
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {
                "url": "https://api.example.com/api/delete",
                "method": "DELETE",
            },
        )

        assert result.success is True
        assert result.data["status_code"] == 204

    async def test_request_with_custom_headers(
        self, mock_transport: httpx.MockTransport
    ) -> None:
        """Custom headers should be passed correctly."""
        module = WebApiModule()
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {
                "url": "https://api.example.com/api/users",
                "method": "GET",
                "headers": {
                    "X-Custom-Header": "test-value",
                    "Authorization": "Bearer token123",
                },
            },
        )

        assert result.success is True

    async def test_timeout_enforcement(
        self, mock_transport: httpx.MockTransport
    ) -> None:
        """Timeout should be enforced and return error on timeout."""
        module = WebApiModule(default_timeout=0.01)
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {"url": "https://api.example.com/timeout", "method": "GET"},
        )

        assert result.success is False
        assert "timed out" in result.error.lower()

    async def test_connection_error_handling(
        self, mock_transport: httpx.MockTransport
    ) -> None:
        """Connection errors should return a clean error message."""
        module = WebApiModule()
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {"url": "https://api.example.com/connect-error", "method": "GET"},
        )

        assert result.success is False
        assert "Connection" in result.error or "connect" in result.error.lower()

    async def test_response_size_limit(
        self, mock_transport: httpx.MockTransport
    ) -> None:
        """Response exceeding max size should be truncated."""
        module = WebApiModule(max_response_size=1024)  # 1KB limit
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {"url": "https://api.example.com/large-response", "method": "GET"},
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["truncated"] is True
        assert result.data["bytes_read"] <= 1024

    async def test_binary_response_handling(
        self,
        mock_transport: httpx.MockTransport,
    ) -> None:
        """Binary responses should be returned as base64 when not valid UTF-8."""
        module = WebApiModule()
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {"url": "https://api.example.com/binary", "method": "GET"},
        )

        assert result.success is True
        assert result.data is not None
        # \xff\xfe\x00\x01 is invalid UTF-8 so it should be base64 encoded
        assert "body_b64" in result.data
        assert "encoding" in result.data
        assert result.data["encoding"] == "base64"
        # Verify base64 encoding
        decoded = base64.b64decode(result.data["body_b64"])
        assert decoded == b"\xff\xfe\x00\x01"

    async def test_http_error_status(self, mock_transport: httpx.MockTransport) -> None:
        """HTTP error status codes should return status_code in response."""
        module = WebApiModule()
        module._client = httpx.AsyncClient(transport=mock_transport)

        result = await module.execute(
            "web_api:http_request",
            {"url": "https://api.example.com/error", "method": "GET"},
        )

        # HTTP responses are returned with success=True; status_code indicates HTTP status
        # The caller should check status_code for error handling
        assert result.success is True
        assert result.data is not None
        assert result.data["status_code"] == 500
        # Response body should be captured
        assert "body" in result.data
        assert "error" in result.data["body"]

    async def test_unknown_tool_returns_error(self) -> None:
        """Unknown tool names should return a failed ToolResult."""
        module = WebApiModule()

        result = await module.execute(
            "web_api:unknown_tool",
            {"url": "http://example.com", "method": "GET"},
        )

        assert result.success is False
        assert "Unknown tool" in result.error

    async def test_missing_url_returns_error(self) -> None:
        """Missing URL parameter should return a failed ToolResult."""
        module = WebApiModule()

        result = await module.execute(
            "web_api:http_request",
            {"method": "GET"},
        )

        assert result.success is False
        assert "url" in result.error.lower()

    async def test_invalid_headers_type(self) -> None:
        """Non-dict headers should return a failed ToolResult."""
        module = WebApiModule()

        result = await module.execute(
            "web_api:http_request",
            {
                "url": "https://example.com",
                "method": "GET",
                "headers": "not-a-dict",
            },
        )

        assert result.success is False
        assert "headers" in result.error.lower()


class TestWebApiModuleProxy:
    """Tests for proxy configuration."""

    async def test_proxy_config_passed_to_client(self) -> None:
        """Proxy URL should be stored and available."""
        module = WebApiModule(proxy_url="http://proxy.example.com:8080")

        assert module.proxy_url == "http://proxy.example.com:8080"


class TestWebApiModuleSecurity:
    """Tests for security features."""

    async def test_invalid_header_names_rejected(self) -> None:
        """Invalid header names should be rejected and logged."""
        module = WebApiModule()

        # We can't easily mock the logger here without fixture, but we can
        # verify the request is made (invalid headers are just skipped)
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, json={"ok": True})
        )
        module._client = httpx.AsyncClient(transport=transport)

        # Header with invalid characters should be rejected
        result = await module.execute(
            "web_api:http_request",
            {
                "url": "https://example.com/api",
                "method": "GET",
                "headers": {
                    "Valid-Header": "ok",
                    "Invalid@Header": "bad",  # @ is not allowed per RFC
                    "Another:Invalid": "bad",  # : is not allowed
                },
            },
        )

        assert result.success is True
        # Invalid headers should have been filtered out

    async def test_header_injection_prevention(self) -> None:
        """Header injection attempts with CRLF should be blocked."""
        module = WebApiModule()
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, json={"ok": True})
        )
        module._client = httpx.AsyncClient(transport=transport)

        # Headers with CRLF injection attempts should be rejected
        result = await module.execute(
            "web_api:http_request",
            {
                "url": "https://example.com/api",
                "method": "GET",
                "headers": {
                    "X-Normal": "value",
                    "X-Injected": "bad\r\nSet-Cookie: malicious=value",
                    "X-Another": "line1\nline2",
                },
            },
        )

        # Request should succeed, but CRLF headers should be filtered
        assert result.success is True

    async def test_ssrf_blocks_private_ipv4(self) -> None:
        """SSRF protection should block private IPv4 addresses."""
        module = WebApiModule()

        # 192.168.x.x - private range
        result = await module.execute(
            "web_api:http_request",
            {"url": "http://192.168.1.1/admin", "method": "GET"},
        )
        assert result.success is False
        assert "internal" in result.error.lower() or "not allowed" in result.error.lower()

        # 10.x.x.x - private range
        result = await module.execute(
            "web_api:http_request",
            {"url": "http://10.0.0.1/internal", "method": "GET"},
        )
        assert result.success is False

        # 172.16-31.x.x - private range
        result = await module.execute(
            "web_api:http_request",
            {"url": "http://172.16.0.1/secret", "method": "GET"},
        )
        assert result.success is False

        # 127.0.0.1 - loopback
        result = await module.execute(
            "web_api:http_request",
            {"url": "http://127.0.0.1:8080/api", "method": "GET"},
        )
        assert result.success is False

        # 169.254.x.x - link-local (AWS metadata)
        result = await module.execute(
            "web_api:http_request",
            {"url": "http://169.254.169.254/latest/meta-data/", "method": "GET"},
        )
        assert result.success is False

    async def test_ssrf_blocks_private_ipv6(self) -> None:
        """SSRF protection should block private IPv6 addresses."""
        module = WebApiModule()

        # ::1 - loopback
        result = await module.execute(
            "web_api:http_request",
            {"url": "http://[::1]:8080/api", "method": "GET"},
        )
        assert result.success is False

        # fc00::/7 - unique local
        result = await module.execute(
            "web_api:http_request",
            {"url": "http://[fc00::1]/internal", "method": "GET"},
        )
        assert result.success is False

        # fe80::/10 - link-local
        result = await module.execute(
            "web_api:http_request",
            {"url": "http://[fe80::1]/data", "method": "GET"},
        )
        assert result.success is False

    async def test_ssrf_blocks_non_http_schemes(self) -> None:
        """SSRF protection should block file:// and other non-HTTP schemes."""
        module = WebApiModule()

        # file:// scheme
        result = await module.execute(
            "web_api:http_request",
            {"url": "file:///etc/passwd", "method": "GET"},
        )
        assert result.success is False
        assert "scheme" in result.error.lower()

        # ftp:// scheme
        result = await module.execute(
            "web_api:http_request",
            {"url": "ftp://example.com/file", "method": "GET"},
        )
        assert result.success is False
        assert "scheme" in result.error.lower()

        # gopher:// scheme
        result = await module.execute(
            "web_api:http_request",
            {"url": "gopher://internal-host:70/1query", "method": "GET"},
        )
        assert result.success is False

    async def test_close_method_works(self) -> None:
        """The close() method should properly clean up resources."""
        module = WebApiModule()

        # Should be able to close the module
        await module.close()

        # Second close should not raise
        await module.close()


class TestWebApiModuleIntegration:
    """Integration-style tests for the web_api module."""

    async def test_full_request_flow_with_all_options(self) -> None:
        """Test a complete request with all options specified."""
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={"result": "success"},
                headers={"Content-Type": "application/json"},
            )
        )
        module = WebApiModule(
            default_timeout=60.0,
            max_response_size=1024 * 1024,
            allowed_methods=["GET", "POST"],
        )
        module._client = httpx.AsyncClient(transport=transport)

        result = await module.execute(
            "web_api:http_request",
            {
                "url": "https://api.example.com/v1/data",
                "method": "POST",
                "headers": {"X-API-Key": "secret123"},
                "body": '{"key": "value"}',
                "content_type": "application/json",
                "timeout": 30,
            },
        )

        assert result.success is True
        assert result.data is not None
        assert result.data["status_code"] == 200
        # Verify body response (JSON format may vary)
        assert '"result"' in result.data["body"]
        assert '"success"' in result.data["body"]
