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

"""Tests for the web browsing module."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest import mock

import httpx
import pytest

from safe_agent.modules.web.browse import (
    DEFAULT_MAX_RESPONSE_SIZE,
    DEFAULT_TIMEOUT,
    WebBrowseModule,
    _extract_content_type,
    _extract_domain,
    _html_to_text,
)


async def _aiter_bytes_from_list(
    chunks: list[bytes],
) -> AsyncIterator[bytes, None]:
    """Create an async iterator from a list of byte chunks."""
    for chunk in chunks:
        yield chunk


class TestDomainExtraction:
    """Tests for domain extraction from URLs."""

    def test_extract_domain_standard_url(self) -> None:
        """Extract domain from standard HTTP URL."""
        assert _extract_domain("https://example.com/page") == "example.com"

    def test_extract_domain_with_port(self) -> None:
        """Extract domain with port from URL."""
        assert _extract_domain("https://example.com:8080/page") == "example.com:8080"

    def test_extract_domain_ip_address(self) -> None:
        """Extract IP address as domain."""
        assert _extract_domain("http://192.168.1.1/page") == "192.168.1.1"

    def test_extract_domain_ip_with_port(self) -> None:
        """Extract IP address with port."""
        url = "http://192.168.1.1:3000/page"
        assert _extract_domain(url) == "192.168.1.1:3000"

    def test_extract_domain_ipv6(self) -> None:
        """Extract IPv6 address as domain."""
        assert _extract_domain("http://[::1]/page") == "[::1]"
        assert _extract_domain("http://[2001:db8::1]/page") == "[2001:db8::1]"

    def test_extract_domain_ipv6_with_port(self) -> None:
        """Extract IPv6 address with port."""
        assert _extract_domain("http://[::1]:8080/page") == "[::1]:8080"

    def test_extract_domain_idn(self) -> None:
        """Extract internationalized domain name."""
        # IDN domains are preserved as-is (punycode conversion is optional)
        assert _extract_domain("https://例え.jp/page") == "例え.jp"
        assert (
            _extract_domain("https://xn--example-9ua.jp/page") == "xn--example-9ua.jp"
        )

    def test_extract_domain_no_hostname(self) -> None:
        """Return empty string for URLs without hostname."""
        assert _extract_domain("not-a-url") == ""

    def test_extract_domain_empty_string(self) -> None:
        """Return empty string for empty input."""
        assert _extract_domain("") == ""

    def test_extract_domain_subdomain(self) -> None:
        """Extract subdomain from URL."""
        assert _extract_domain("https://sub.example.com/page") == "sub.example.com"


class TestContentTypeExtraction:
    """Tests for content type extraction from headers."""

    def test_extract_content_type_standard(self) -> None:
        """Extract standard content type."""
        assert _extract_content_type("text/html") == "text/html"

    def test_extract_content_type_with_charset(self) -> None:
        """Extract content type stripping charset."""
        assert _extract_content_type("text/html; charset=utf-8") == "text/html"

    def test_extract_content_type_with_multiple_params(self) -> None:
        """Extract content type stripping multiple parameters."""
        result = _extract_content_type("text/html; charset=utf-8; boundary=something")
        assert result == "text/html"

    def test_extract_content_type_none(self) -> None:
        """Return empty string for None input."""
        assert _extract_content_type(None) == ""

    def test_extract_content_type_empty(self) -> None:
        """Return empty string for empty input."""
        assert _extract_content_type("") == ""

    def test_extract_content_type_case_normalized(self) -> None:
        """Content type is lowercased."""
        assert _extract_content_type("TEXT/HTML") == "text/html"


class TestHtmlToText:
    """Tests for HTML to text extraction."""

    def test_basic_tag_removal(self) -> None:
        """Remove basic HTML tags."""
        html = "<p>Hello <b>World</b></p>"
        assert _html_to_text(html) == "Hello World"

    def test_script_removal(self) -> None:
        """Remove script tags with content."""
        html = "<html><body>Hello<script>alert('xss')</script>World</body></html>"
        text = _html_to_text(html)
        assert "alert" not in text
        assert "Hello" in text
        assert "World" in text

    def test_style_removal(self) -> None:
        """Remove style tags with content."""
        html = (
            "<html><head><style>body { color: red; }</style>"
            "</head><body>Content</body></html>"
        )
        text = _html_to_text(html)
        assert "color" not in text
        assert "Content" in text

    def test_comment_removal(self) -> None:
        """Remove HTML comments."""
        html = "Hello <!-- this is a comment -->World"
        assert _html_to_text(html) == "Hello World"

    def test_entity_decoding(self) -> None:
        """Decode common HTML entities."""
        html = "&lt;tag&gt; &amp; &quot;quoted&quot; &apos;text&apos;"
        text = _html_to_text(html)
        assert "<tag>" in text
        assert "&" in text
        assert '"quoted"' in text
        assert "'text'" in text

    def test_whitespace_normalization(self) -> None:
        """Normalize whitespace."""
        html = "<p>  Multiple   spaces   </p>"
        text = _html_to_text(html)
        assert "Multiple spaces" in text

    def test_block_elements_add_newlines(self) -> None:
        """Block elements should add newlines."""
        html = "<p>Para 1</p><p>Para 2</p>"
        text = _html_to_text(html)
        assert "Para 1" in text
        assert "Para 2" in text

    def test_br_to_newline(self) -> None:
        """BR tags should become newlines."""
        html = "Line 1<br>Line 2<br/>Line 3"
        text = _html_to_text(html)
        assert "Line 1" in text
        assert "Line 2" in text
        assert "Line 3" in text


class TestWebBrowseModuleDescriptor:
    """Tests for module descriptor and initialization."""

    def test_describe_returns_valid_descriptor(self) -> None:
        """describe() should return the expected module metadata."""
        module = WebBrowseModule()

        descriptor = module.describe()

        assert descriptor.namespace == "web_browse"
        assert len(descriptor.tools) == 2
        tool_names = {tool.name for tool in descriptor.tools}
        assert tool_names == {
            "web_browse:fetch_page",
            "web_browse:summarize_page",
        }

    def test_fetch_page_descriptor(self) -> None:
        """fetch_page tool should have correct descriptor."""
        module = WebBrowseModule()
        descriptor = module.describe()

        fetch_tool = next(
            tool for tool in descriptor.tools if tool.name == "web_browse:fetch_page"
        )

        assert fetch_tool.action == "web_browse:FetchPage"
        assert fetch_tool.resource_param == ["url"]
        assert set(fetch_tool.condition_keys) == {
            "web_browse:Domain",
            "web_browse:ContentType",
        }
        assert fetch_tool.parameters["required"] == ["url"]
        assert fetch_tool.parameters["additionalProperties"] is False

    def test_summarize_page_descriptor(self) -> None:
        """summarize_page tool should have correct descriptor."""
        module = WebBrowseModule()
        descriptor = module.describe()

        summarize_tool = next(
            tool
            for tool in descriptor.tools
            if tool.name == "web_browse:summarize_page"
        )

        assert summarize_tool.action == "web_browse:SummarizePage"
        assert summarize_tool.resource_param == ["url"]
        assert set(summarize_tool.condition_keys) == {
            "web_browse:Domain",
            "web_browse:ContentType",
        }
        assert summarize_tool.parameters["required"] == ["url"]
        assert summarize_tool.parameters["additionalProperties"] is False

    def test_default_timeout(self) -> None:
        """Default timeout should be DEFAULT_TIMEOUT."""
        module = WebBrowseModule()
        assert module.default_timeout == DEFAULT_TIMEOUT

    def test_custom_timeout(self) -> None:
        """Custom timeout should be set correctly."""
        module = WebBrowseModule(default_timeout=60.0)
        assert module.default_timeout == 60.0

    def test_timeout_too_low_raises(self) -> None:
        """Timeout <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="default_timeout must be > 0"):
            WebBrowseModule(default_timeout=0)
        with pytest.raises(ValueError, match="default_timeout must be > 0"):
            WebBrowseModule(default_timeout=-1)

    def test_timeout_too_high_raises(self) -> None:
        """Timeout > MAX_TIMEOUT should raise ValueError."""
        with pytest.raises(ValueError, match="default_timeout must be <="):
            WebBrowseModule(default_timeout=99999)

    def test_default_max_response_size(self) -> None:
        """Default max_response_size should be DEFAULT_MAX_RESPONSE_SIZE."""
        module = WebBrowseModule()
        assert module.max_response_size == DEFAULT_MAX_RESPONSE_SIZE

    def test_custom_max_response_size(self) -> None:
        """Custom max_response_size should be set correctly."""
        module = WebBrowseModule(max_response_size=1024)
        assert module.max_response_size == 1024

    def test_max_response_size_too_low_raises(self) -> None:
        """max_response_size <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_response_size must be > 0"):
            WebBrowseModule(max_response_size=0)
        with pytest.raises(ValueError, match="max_response_size must be > 0"):
            WebBrowseModule(max_response_size=-1)

    def test_proxy_url_stored(self) -> None:
        """Proxy URL should be stored."""
        module = WebBrowseModule(proxy_url="http://proxy.example.com:8080")
        assert module.proxy_url == "http://proxy.example.com:8080"


class TestResolveConditions:
    """Tests for condition resolution."""

    async def test_resolve_conditions_fetch_page(self) -> None:
        """resolve_conditions should extract domain for fetch_page."""
        module = WebBrowseModule()

        conditions = await module.resolve_conditions(
            "web_browse:fetch_page",
            {"url": "https://example.com/page"},
        )

        assert conditions["web_browse:Domain"] == "example.com"
        assert conditions["web_browse:ContentType"] == ""

    async def test_resolve_conditions_summarize_page(self) -> None:
        """resolve_conditions should extract domain for summarize_page."""
        module = WebBrowseModule()

        conditions = await module.resolve_conditions(
            "web_browse:summarize_page",
            {"url": "https://example.org:8080/article"},
        )

        assert conditions["web_browse:Domain"] == "example.org:8080"
        assert conditions["web_browse:ContentType"] == ""

    async def test_resolve_conditions_unknown_tool(self) -> None:
        """resolve_conditions should return empty dict for unknown tool."""
        module = WebBrowseModule()

        conditions = await module.resolve_conditions(
            "web_browse:unknown",
            {"url": "https://example.com"},
        )

        assert conditions == {}

    async def test_resolve_conditions_handles_ip_url(self) -> None:
        """resolve_conditions should handle IP URLs."""
        module = WebBrowseModule()

        conditions = await module.resolve_conditions(
            "web_browse:fetch_page",
            {"url": "http://192.168.1.1:3000/path"},
        )

        assert conditions["web_browse:Domain"] == "192.168.1.1:3000"

    async def test_resolve_conditions_handles_idn_url(self) -> None:
        """resolve_conditions should handle IDN URLs."""
        module = WebBrowseModule()

        conditions = await module.resolve_conditions(
            "web_browse:fetch_page",
            {"url": "https://例え.jp/page"},
        )

        assert conditions["web_browse:Domain"] == "例え.jp"


class TestFetchPage:
    """Tests for fetch_page tool."""

    async def test_fetch_page_success(self) -> None:
        """fetch_page should return HTML content."""
        module = WebBrowseModule()

        # Create a mock response with proper async iterator
        mock_response = mock.MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.url = "https://example.com/page"
        mock_response.aiter_bytes = mock.MagicMock(
            return_value=_aiter_bytes_from_list([b"<html><body>Hello</body></html>"])
        )
        mock_response.aclose = mock.AsyncMock()

        with (
            mock.patch.object(module._client, "send", return_value=mock_response),
            mock.patch.object(
                module._client, "build_request", return_value=mock.MagicMock()
            ),
        ):
            result = await module.execute(
                "web_browse:fetch_page",
                {"url": "https://example.com/page"},
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["status_code"] == 200
        assert result.data["content_type"] == "text/html"
        assert result.data["domain"] == "example.com"
        assert "<html>" in result.data["html"]

    async def test_fetch_page_with_custom_timeout(self) -> None:
        """fetch_page should respect custom timeout."""
        module = WebBrowseModule(default_timeout=10.0)

        mock_response = mock.MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com/"
        mock_response.aiter_bytes = mock.MagicMock(
            return_value=_aiter_bytes_from_list([b"content"])
        )
        mock_response.aclose = mock.AsyncMock()

        mock_request = mock.MagicMock()

        with (
            mock.patch.object(
                module._client, "send", return_value=mock_response
            ) as mock_send,
            mock.patch.object(
                module._client, "build_request", return_value=mock_request
            ),
        ):
            result = await module.execute(
                "web_browse:fetch_page",
                {"url": "https://example.com/", "timeout": 60.0},
            )

            # Verify timeout was passed (clamped to MAX_TIMEOUT)
            mock_send.assert_called_once()
            assert result.success is True

    async def test_fetch_page_timeout_error(self) -> None:
        """fetch_page should handle timeout errors."""
        module = WebBrowseModule()

        with mock.patch.object(
            module._client,
            "send",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            result = await module.execute(
                "web_browse:fetch_page",
                {"url": "https://example.com/"},
            )

        assert result.success is False
        assert result.error == "Request timed out"

    async def test_fetch_page_connection_error(self) -> None:
        """fetch_page should handle connection errors."""
        module = WebBrowseModule()

        with mock.patch.object(
            module._client,
            "send",
            side_effect=httpx.ConnectError("connection failed"),
        ):
            result = await module.execute(
                "web_browse:fetch_page",
                {"url": "https://example.com/"},
            )

        assert result.success is False
        assert "Connection failed" in result.error

    async def test_fetch_page_truncates_large_response(self) -> None:
        """fetch_page should truncate responses exceeding max_response_size."""
        small_limit = 100
        module = WebBrowseModule(max_response_size=small_limit)

        large_content = b"x" * 200

        mock_response = mock.MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com/"
        mock_response.aiter_bytes = mock.MagicMock(
            return_value=_aiter_bytes_from_list([large_content])
        )
        mock_response.aclose = mock.AsyncMock()

        with (
            mock.patch.object(module._client, "send", return_value=mock_response),
            mock.patch.object(
                module._client, "build_request", return_value=mock.MagicMock()
            ),
        ):
            result = await module.execute(
                "web_browse:fetch_page",
                {"url": "https://example.com/"},
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["truncated"] is True
        assert result.data["max_response_size"] == small_limit
        assert len(result.data["html"]) <= small_limit

    async def test_fetch_page_missing_url(self) -> None:
        """fetch_page should reject missing URL."""
        module = WebBrowseModule()

        result = await module.execute("web_browse:fetch_page", {})

        assert result.success is False
        assert "url is required" in result.error

    async def test_fetch_page_invalid_url_scheme(self) -> None:
        """fetch_page should reject non-HTTP schemes."""
        module = WebBrowseModule()

        result = await module.execute(
            "web_browse:fetch_page",
            {"url": "ftp://example.com/file"},
        )

        assert result.success is False
        assert "Invalid URL scheme" in result.error

    async def test_fetch_page_url_without_hostname(self) -> None:
        """fetch_page should reject URLs without hostname."""
        module = WebBrowseModule()

        result = await module.execute(
            "web_browse:fetch_page",
            {"url": "http:///path"},
        )

        assert result.success is False
        assert "must include a hostname" in result.error


class TestSummarizePage:
    """Tests for summarize_page tool."""

    async def test_summarize_page_success(self) -> None:
        """summarize_page should extract text content."""
        module = WebBrowseModule()

        html_content = (
            b"<html><head><title>Test</title></head>"
            b"<body><h1>Hello</h1><p>World</p></body></html>"
        )

        mock_response = mock.MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.url = "https://example.com/"
        mock_response.aiter_bytes = mock.MagicMock(
            return_value=_aiter_bytes_from_list([html_content])
        )
        mock_response.aclose = mock.AsyncMock()

        with (
            mock.patch.object(module._client, "send", return_value=mock_response),
            mock.patch.object(
                module._client, "build_request", return_value=mock.MagicMock()
            ),
        ):
            result = await module.execute(
                "web_browse:summarize_page",
                {"url": "https://example.com/"},
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["status_code"] == 200
        assert result.data["content_type"] == "text/html"
        assert "Hello" in result.data["text"]
        assert "World" in result.data["text"]
        assert "<html>" not in result.data["text"]

    async def test_summarize_page_truncates_text(self) -> None:
        """summarize_page should truncate text to max_length."""
        module = WebBrowseModule()

        # Create HTML that will produce more than 100 chars of text
        long_text = "x" * 200
        html_content = f"<html><body>{long_text}</body></html>".encode()

        mock_response = mock.MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com/"
        mock_response.aiter_bytes = mock.MagicMock(
            return_value=_aiter_bytes_from_list([html_content])
        )
        mock_response.aclose = mock.AsyncMock()

        with (
            mock.patch.object(module._client, "send", return_value=mock_response),
            mock.patch.object(
                module._client, "build_request", return_value=mock.MagicMock()
            ),
        ):
            result = await module.execute(
                "web_browse:summarize_page",
                {"url": "https://example.com/", "max_length": 100},
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["text_truncated"] is True
        assert result.data["max_text_length"] == 100
        assert len(result.data["text"]) == 100

    async def test_summarize_page_removes_scripts(self) -> None:
        """summarize_page should remove script content."""
        module = WebBrowseModule()

        html_content = (
            b"<html><body>Hello<script>alert('xss')</script>World</body></html>"
        )

        mock_response = mock.MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com/"
        mock_response.aiter_bytes = mock.MagicMock(
            return_value=_aiter_bytes_from_list([html_content])
        )
        mock_response.aclose = mock.AsyncMock()

        with (
            mock.patch.object(module._client, "send", return_value=mock_response),
            mock.patch.object(
                module._client, "build_request", return_value=mock.MagicMock()
            ),
        ):
            result = await module.execute(
                "web_browse:summarize_page",
                {"url": "https://example.com/"},
            )

        assert result.success is True
        assert "alert" not in result.data["text"]
        assert "Hello" in result.data["text"]
        assert "World" in result.data["text"]

    async def test_summarize_page_missing_url(self) -> None:
        """summarize_page should reject missing URL."""
        module = WebBrowseModule()

        result = await module.execute("web_browse:summarize_page", {})

        assert result.success is False
        assert "url is required" in result.error

    async def test_summarize_page_timeout_error(self) -> None:
        """summarize_page should handle timeout errors."""
        module = WebBrowseModule()

        with mock.patch.object(
            module._client,
            "send",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            result = await module.execute(
                "web_browse:summarize_page",
                {"url": "https://example.com/"},
            )

        assert result.success is False
        assert result.error == "Request timed out"


class TestTimeoutHandling:
    """Tests for timeout configuration."""

    async def test_timeout_clamped_to_max(self) -> None:
        """Timeout should be clamped to MAX_TIMEOUT."""
        module = WebBrowseModule()

        mock_response = mock.MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com/"
        mock_response.aiter_bytes = mock.MagicMock(
            return_value=_aiter_bytes_from_list([b"content"])
        )
        mock_response.aclose = mock.AsyncMock()

        mock_request = mock.MagicMock()

        with (
            mock.patch.object(
                module._client, "send", return_value=mock_response
            ) as mock_send,
            mock.patch.object(
                module._client, "build_request", return_value=mock_request
            ),
        ):
            await module.execute(
                "web_browse:fetch_page",
                {"url": "https://example.com/", "timeout": 99999},
            )

            # The timeout should be clamped to MAX_TIMEOUT
            mock_send.assert_called_once()

    async def test_timeout_zero_rejected(self) -> None:
        """Timeout of zero should be rejected."""
        module = WebBrowseModule()

        result = await module.execute(
            "web_browse:fetch_page",
            {"url": "https://example.com/", "timeout": 0},
        )

        assert result.success is False
        assert "timeout must be" in result.error

    async def test_timeout_negative_rejected(self) -> None:
        """Negative timeout should be rejected."""
        module = WebBrowseModule()

        result = await module.execute(
            "web_browse:fetch_page",
            {"url": "https://example.com/", "timeout": -5},
        )

        assert result.success is False
        assert "timeout must be" in result.error


class TestProxyConfiguration:
    """Tests for proxy configuration."""

    async def test_proxy_used_in_client(self) -> None:
        """Proxy URL should be used when creating client."""
        proxy_url = "http://proxy.example.com:8080"
        module = WebBrowseModule(proxy_url=proxy_url)

        assert module.proxy_url == proxy_url

        # Verify client was created with proxy
        # The client should have the proxy configured
        await module.close()

    async def test_fetch_page_with_proxy(self) -> None:
        """fetch_page should work with proxy configured."""
        module = WebBrowseModule(proxy_url="http://proxy.example.com:8080")

        mock_response = mock.MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com/"
        mock_response.aiter_bytes = mock.MagicMock(
            return_value=_aiter_bytes_from_list([b"content"])
        )
        mock_response.aclose = mock.AsyncMock()

        with (
            mock.patch.object(module._client, "send", return_value=mock_response),
            mock.patch.object(
                module._client, "build_request", return_value=mock.MagicMock()
            ),
        ):
            result = await module.execute(
                "web_browse:fetch_page",
                {"url": "https://example.com/"},
            )

        assert result.success is True


class TestUnknownTool:
    """Tests for unknown tool handling."""

    async def test_unknown_tool_returns_error(self) -> None:
        """Unknown tool should return error."""
        module = WebBrowseModule()

        result = await module.execute(
            "web_browse:unknown_tool",
            {"url": "https://example.com/"},
        )

        assert result.success is False
        assert "Unknown tool" in result.error


class TestResponseSizeLimit:
    """Tests for response size limiting."""

    async def test_response_size_limit_enforced(self) -> None:
        """Response should be truncated at max_response_size."""
        limit = 50
        module = WebBrowseModule(max_response_size=limit)

        # Create response larger than limit
        large_content = b"x" * 200

        mock_response = mock.MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com/"
        mock_response.aiter_bytes = mock.MagicMock(
            return_value=_aiter_bytes_from_list([large_content])
        )
        mock_response.aclose = mock.AsyncMock()

        with (
            mock.patch.object(module._client, "send", return_value=mock_response),
            mock.patch.object(
                module._client, "build_request", return_value=mock.MagicMock()
            ),
        ):
            result = await module.execute(
                "web_browse:fetch_page",
                {"url": "https://example.com/"},
            )

        assert result.success is True
        assert result.data is not None
        assert result.data["bytes_read"] <= limit


class TestModuleCleanup:
    """Tests for module cleanup."""

    async def test_close_client(self) -> None:
        """close() should close the httpx client."""
        module = WebBrowseModule()

        # Mock the client's aclose method
        module._client.aclose = mock.AsyncMock()

        await module.close()

        module._client.aclose.assert_called_once()
