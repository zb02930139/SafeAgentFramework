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

"""Tests for access policy Pydantic models."""

import pytest
from pydantic import ValidationError

from safe_agent.access.models import (
    AuthorizationRequest,
    AuthorizationResult,
    Decision,
    Policy,
    Statement,
)


def make_allow_statement(**kwargs):
    """Build a minimal Allow statement dict with optional overrides."""
    defaults = {
        "Effect": "Allow",
        "Action": ["agent:*"],
        "Resource": ["*"],
    }
    defaults.update(kwargs)
    return defaults


class TestDecision:
    """Tests for the Decision enum."""

    def test_values(self):
        assert Decision.ALLOWED == "ALLOWED"
        assert Decision.DENIED_EXPLICIT == "DENIED_EXPLICIT"
        assert Decision.DENIED_IMPLICIT == "DENIED_IMPLICIT"


class TestStatement:
    """Tests for the Statement model."""

    def test_basic_allow(self):
        data = {
            "Effect": "Allow",
            "Action": ["agent:RunTool"],
            "Resource": ["arn:tool:bash"],
        }
        stmt = Statement.model_validate(data)
        assert stmt.effect == "Allow"
        assert stmt.action == ["agent:RunTool"]
        assert stmt.resource == ["arn:tool:bash"]
        assert stmt.sid is None
        assert stmt.condition is None

    def test_basic_deny(self):
        data = {"Effect": "Deny", "Action": ["agent:*"], "Resource": ["*"]}
        stmt = Statement.model_validate(data)
        assert stmt.effect == "Deny"

    def test_with_sid(self):
        data = {
            "Sid": "MyRule",
            "Effect": "Allow",
            "Action": ["*"],
            "Resource": ["*"],
        }
        stmt = Statement.model_validate(data)
        assert stmt.sid == "MyRule"

    def test_with_condition(self):
        data = {
            "Effect": "Allow",
            "Action": ["agent:RunTool"],
            "Resource": ["*"],
            "Condition": {"StringEquals": {"env": "prod"}},
        }
        stmt = Statement.model_validate(data)
        assert stmt.condition == {"StringEquals": {"env": "prod"}}

    def test_invalid_effect(self):
        data = {"Effect": "Maybe", "Action": ["agent:*"], "Resource": ["*"]}
        with pytest.raises(ValidationError):
            Statement.model_validate(data)

    def test_missing_effect(self):
        data = {"Action": ["agent:*"], "Resource": ["*"]}
        with pytest.raises(ValidationError):
            Statement.model_validate(data)


class TestPolicy:
    """Tests for the Policy model."""

    def test_basic_policy(self):
        data = {
            "Version": "2025-01",
            "Statement": [
                {"Effect": "Allow", "Action": ["agent:*"], "Resource": ["*"]},
            ],
        }
        policy = Policy.model_validate(data)
        assert policy.version == "2025-01"
        assert len(policy.statements) == 1

    def test_multiple_statements(self):
        data = {
            "Version": "2025-01",
            "Statement": [
                {"Effect": "Allow", "Action": ["agent:Read"], "Resource": ["*"]},
                {"Effect": "Deny", "Action": ["agent:Write"], "Resource": ["*"]},
            ],
        }
        policy = Policy.model_validate(data)
        assert len(policy.statements) == 2

    def test_missing_version(self):
        data = {"Statement": [{"Effect": "Allow", "Action": ["*"], "Resource": ["*"]}]}
        with pytest.raises(ValidationError):
            Policy.model_validate(data)


class TestAuthorizationRequest:
    """Tests for the AuthorizationRequest model."""

    def test_basic(self):
        req = AuthorizationRequest(action="agent:RunTool", resource="arn:tool:bash")
        assert req.action == "agent:RunTool"
        assert req.resource == "arn:tool:bash"
        assert req.context == {}

    def test_with_context(self):
        req = AuthorizationRequest(
            action="agent:RunTool",
            resource="arn:tool:bash",
            context={"env": "prod", "user": "alice"},
        )
        assert req.context["env"] == "prod"


class TestAuthorizationResult:
    """Tests for the AuthorizationResult model."""

    def test_basic(self):
        result = AuthorizationResult(decision=Decision.ALLOWED, reason="Test")
        assert result.decision == Decision.ALLOWED
        assert result.matched_statements == []
