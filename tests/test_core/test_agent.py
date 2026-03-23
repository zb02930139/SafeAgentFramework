"""Unit tests for Agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from safe_agent import Agent
from safe_agent.core.audit import AuditLogger
from safe_agent.core.dispatcher import ToolDispatcher
from safe_agent.core.event_loop import EventLoop
from safe_agent.core.gateway import Gateway
from safe_agent.core.llm import LLMClient
from safe_agent.core.session import SessionManager
from safe_agent.iam.evaluator import PolicyEvaluator
from safe_agent.iam.policy import PolicyStore
from safe_agent.modules.base import BaseModule
from safe_agent.modules.registry import ModuleRegistry


class MockLLM(LLMClient):
    """Test double for LLMClient."""
    
    def __init__(self) -> None:
        self.chat_calls: list[tuple[list, list]] = []
    
    async def chat(self, messages: list, tools: list) -> None:
        self.chat_calls.append((messages, tools))
        return Mock()


class MockModule(BaseModule):
    """Test module for testing."""
    
    def describe(self):
        from safe_agent.modules.base import ModuleDescriptor, ToolDescriptor
        return ModuleDescriptor(
            namespace="test",
            description="Test module",
            tools=[
                ToolDescriptor(
                    name="test:echo",
                    description="Echo a message",
                    parameters={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                        },
                        "required": ["message"],
                    },
                    action="test:Echo",
                    resource_param="message",
                )
            ]
        )
    
    async def resolve_conditions(self, tool_name, params):
        return {"test:message_length": len(str(params.get("message", "")))}
    
    async def execute(self, tool_name, params):
        from safe_agent.modules.base import ToolResult
        return ToolResult(
            success=True,
            data={"echoed": params.get("message", "")},
            error=None,
            metadata={}
        )


@pytest.fixture
def mock_policy_dir(tmp_path: Path) -> Path:
    """Create a temporary policy directory."""
    policy_dir = tmp_path / "policies"
    policy_dir.mkdir()
    (policy_dir / "policy.json").write_text('{"Version": "2025-01", "Statement": []}')
    return policy_dir


@pytest.fixture
def mock_llm() -> MockLLM:
    """Create a mock LLM client."""
    return MockLLM()


@pytest.mark.asyncio
async def test_agent_constructor_with_missing_policy_dir(tmp_path: Path) -> None:
    """Test constructor with missing policy directory."""
    missing_dir = tmp_path / "nonexistent"
    
    # This should not raise an error since PolicyStore.load() doesn't check
    # if directory exists - it just returns empty policy set
    agent = Agent(
        policy_dir=missing_dir,
        llm_client=MockLLM(),
        modules=[MockModule()],
    )
    
    assert agent.policy_store is not None
    # Store should be empty since no JSON files were found
    assert len(agent.policy_store.get_all_statements()) == 0


@pytest.mark.asyncio
async def test_agent_constructor_with_empty_modules_list(mock_policy_dir: Path, mock_llm: MockLLM) -> None:
    """Test constructor with empty modules list."""
    agent = Agent(
        policy_dir=mock_policy_dir,
        llm_client=mock_llm,
        modules=[],
    )
    
    assert agent.registry is not None
    assert len(agent.registry._namespace_map) == 0
    assert agent.policy_store is not None
    assert agent.policy_evaluator is not None
    assert agent.dispatcher is not None
    assert agent.session_manager is not None
    assert agent.event_loop is not None
    assert agent.gateway is not None


@pytest.mark.asyncio
async def test_agent_constructor_with_discover_true(mock_policy_dir: Path, mock_llm: MockLLM) -> None:
    """Test constructor with discover=True path."""
    with patch('safe_agent.modules.registry.ModuleRegistry.discover') as mock_discover:
        agent = Agent(
            policy_dir=mock_policy_dir,
            llm_client=mock_llm,
            modules=None,
        )
        
        assert agent.registry is not None
        mock_discover.assert_called_once()


@pytest.mark.asyncio
async def test_agent_constructor_with_custom_modules(mock_policy_dir: Path, mock_llm: MockLLM) -> None:
    """Test constructor with custom modules."""
    module1 = MockModule()
    # Create a second module with different namespace and tool name
    module2 = MockModule()
    # Override describe method for module2 to have different namespace and tool
    original_describe = module2.describe
    def describe2():
        desc = original_describe()
        desc.namespace = "test2"
        desc.tools[0].name = "test2:echo"
        desc.tools[0].action = "test2:Echo"
        return desc
    module2.describe = describe2
    
    agent = Agent(
        policy_dir=mock_policy_dir,
        llm_client=mock_llm,
        modules=[module1, module2],
    )
    
    assert agent.registry is not None
    assert len(agent.registry._namespace_map) == 2


@pytest.mark.asyncio
async def test_agent_constructor_freezes_policy_store(mock_policy_dir: Path, mock_llm: MockLLM) -> None:
    """Test that freeze() is called during construction."""
    agent = Agent(
        policy_dir=mock_policy_dir,
        llm_client=mock_llm,
        modules=[MockModule()],
    )
    
    assert agent.policy_store._frozen is True
    with pytest.raises(RuntimeError, match="frozen"):
        agent.policy_store.add_policy(Mock())


@pytest.mark.asyncio
async def test_agent_chat_delegates_to_gateway(mock_policy_dir: Path, mock_llm: MockLLM) -> None:
    """Test that chat() delegates correctly to gateway."""
    agent = Agent(
        policy_dir=mock_policy_dir,
        llm_client=mock_llm,
        modules=[MockModule()],
    )
    
    with patch.object(agent.gateway, 'submit') as mock_submit:
        mock_submit.return_value = ("test response", "test-session-id")
        
        response, session_id = await agent.chat("test message")
        
        mock_submit.assert_called_once_with("test message", None)
        assert response == "test response"
        assert session_id == "test-session-id"


@pytest.mark.asyncio
async def test_agent_chat_with_session_id(mock_policy_dir: Path, mock_llm: MockLLM) -> None:
    """Test chat() with existing session ID."""
    agent = Agent(
        policy_dir=mock_policy_dir,
        llm_client=mock_llm,
        modules=[MockModule()],
    )
    
    with patch.object(agent.gateway, 'submit') as mock_submit:
        mock_submit.return_value = ("test response", "existing-session-id")
        
        response, session_id = await agent.chat("test message", "existing-session-id")
        
        mock_submit.assert_called_once_with("test message", "existing-session-id")
        assert response == "test response"
        assert session_id == "existing-session-id"





@pytest.mark.asyncio
async def test_agent_constructor_with_audit_log_path(mock_policy_dir: Path, mock_llm: MockLLM) -> None:
    """Test constructor with custom audit log path."""
    custom_log_path = mock_policy_dir / "custom_audit.jsonl"
    
    agent = Agent(
        policy_dir=mock_policy_dir,
        llm_client=mock_llm,
        modules=[MockModule()],
        audit_log_path=custom_log_path,
    )
    
    assert agent.audit_logger._log_path == custom_log_path


@pytest.mark.asyncio
async def test_agent_constructor_with_max_turns(mock_policy_dir: Path, mock_llm: MockLLM) -> None:
    """Test constructor with custom max_turns."""
    agent = Agent(
        policy_dir=mock_policy_dir,
        llm_client=mock_llm,
        modules=[MockModule()],
        max_turns=5,
    )
    
    assert agent.event_loop._max_turns == 5