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

"""Tests for access policy model validation (issue #138)."""

import pytest
from pydantic import ValidationError

from safe_agent.access.models import Policy, Statement


class TestStatementValidation:
    """Tests for Statement model validation."""

    def test_valid_statement(self):
        """A valid statement should be accepted."""
        stmt = Statement(
            effect="Allow",
            action=["agent:RunTool"],
            resource=["arn:tool:bash"],
        )
        assert stmt.effect == "Allow"
        assert stmt.action == ["agent:RunTool"]
        assert stmt.resource == ["arn:tool:bash"]

    def test_empty_action_list_rejected(self):
        """Empty action list should be rejected (issue #138)."""
        with pytest.raises(ValidationError, match="action"):
            Statement(
                effect="Allow",
                action=[],
                resource=["*"],
            )

    def test_empty_resource_list_rejected(self):
        """Empty resource list should be rejected (issue #138)."""
        with pytest.raises(ValidationError, match="resource"):
            Statement(
                effect="Allow",
                action=["agent:*"],
                resource=[],
            )

    def test_empty_both_action_and_resource_rejected(self):
        """Both empty lists should be rejected."""
        with pytest.raises(ValidationError):
            Statement(
                effect="Allow",
                action=[],
                resource=[],
            )

    def test_valid_condition_structure(self):
        """Valid condition structure should be accepted."""
        stmt = Statement(
            effect="Allow",
            action=["agent:*"],
            resource=["*"],
            condition={"StringEquals": {"env": "prod"}},
        )
        assert stmt.condition == {"StringEquals": {"env": "prod"}}

    def test_malformed_condition_non_dict_value_rejected(self):
        """Condition with non-dict value should be rejected (issue #138).

        The condition field should require dict[str, dict[str, Any]] structure.
        A condition like {"StringEquals": "not_a_dict"} should fail validation.
        """
        with pytest.raises(ValidationError, match="condition"):
            Statement(
                effect="Allow",
                action=["agent:*"],
                resource=["*"],
                condition={"StringEquals": "not_a_dict"},
            )

    def test_condition_with_numeric_value_accepted(self):
        """Condition with numeric value should be accepted (valid Any)."""
        stmt = Statement(
            effect="Allow",
            action=["agent:*"],
            resource=["*"],
            condition={"NumericEquals": {"count": 42}},  # 42 is valid numeric
        )
        assert stmt.condition["NumericEquals"]["count"] == 42

    def test_condition_with_multiple_operators(self):
        """Condition with multiple operators should be accepted."""
        stmt = Statement(
            effect="Allow",
            action=["agent:*"],
            resource=["*"],
            condition={
                "StringEquals": {"env": "prod"},
                "NumericGreaterThan": {"level": 5},
            },
        )
        assert "StringEquals" in stmt.condition
        assert "NumericGreaterThan" in stmt.condition

    def test_condition_with_list_values(self):
        """Condition with list values should be accepted."""
        stmt = Statement(
            effect="Allow",
            action=["agent:*"],
            resource=["*"],
            condition={"StringEquals": {"env": ["prod", "staging"]}},
        )
        assert stmt.condition["StringEquals"]["env"] == ["prod", "staging"]


class TestPolicyValidation:
    """Tests for Policy model validation."""

    def test_valid_policy(self):
        """A valid policy should be accepted."""
        policy = Policy(
            version="2025-01",
            statements=[
                Statement(
                    effect="Allow",
                    action=["agent:*"],
                    resource=["*"],
                )
            ],
        )
        assert policy.version == "2025-01"
        assert len(policy.statements) == 1

    def test_policy_with_invalid_statement_fails(self):
        """Policy with invalid statement should fail validation."""
        with pytest.raises(ValidationError):
            Policy(
                version="2025-01",
                statements=[
                    Statement(
                        effect="Allow",
                        action=[],  # Invalid: empty list
                        resource=["*"],
                    )
                ],
            )

    def test_policy_from_json_with_valid_data(self):
        """Policy should be constructable from JSON dict."""
        data = {
            "Version": "2025-01",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["agent:RunTool"],
                    "Resource": ["arn:tool:bash"],
                }
            ],
        }
        policy = Policy.model_validate(data)
        assert policy.version == "2025-01"
        assert len(policy.statements) == 1

    def test_policy_from_json_with_empty_action_rejected(self):
        """Policy JSON with empty action list should be rejected."""
        data = {
            "Version": "2025-01",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [],  # Invalid
                    "Resource": ["*"],
                }
            ],
        }
        with pytest.raises(ValidationError):
            Policy.model_validate(data)

    def test_policy_from_json_with_malformed_condition_rejected(self):
        """Policy JSON with malformed condition should be rejected."""
        data = {
            "Version": "2025-01",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["agent:*"],
                    "Resource": ["*"],
                    "Condition": {"StringEquals": "not_a_dict"},  # Invalid
                }
            ],
        }
        with pytest.raises(ValidationError):
            Policy.model_validate(data)
