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


"""Tests for PolicyEvaluator."""

from safe_agent.access.evaluator import PolicyEvaluator
from safe_agent.access.models import (
    AuthorizationRequest,
    Decision,
    Policy,
)
from safe_agent.access.policy import PolicyStore


def make_store(*policy_dicts) -> PolicyStore:
    """Build a PolicyStore from one or more raw policy dicts."""
    store = PolicyStore()
    for d in policy_dicts:
        store.add_policy(Policy.model_validate(d))
    return store


def make_request(action: str, resource: str, **ctx) -> AuthorizationRequest:
    """Build an AuthorizationRequest with optional context kwargs."""
    return AuthorizationRequest(action=action, resource=resource, context=ctx)


# ---------------------------------------------------------------------------
# Helper policies
# ---------------------------------------------------------------------------

ALLOW_ALL = {
    "Version": "2025-01",
    "Statement": [{"Effect": "Allow", "Action": ["*"], "Resource": ["*"]}],
}

DENY_ALL = {
    "Version": "2025-01",
    "Statement": [{"Effect": "Deny", "Action": ["*"], "Resource": ["*"]}],
}

ALLOW_AGENT_TOOLS = {
    "Version": "2025-01",
    "Statement": [
        {"Effect": "Allow", "Action": ["agent:RunTool"], "Resource": ["arn:tool:*"]}
    ],
}


class TestDefaultDeny:
    """Verify that requests are denied when no matching Allow statement exists."""

    def test_no_policies_implicit_deny(self):
        store = PolicyStore()
        ev = PolicyEvaluator(store)
        result = ev.evaluate(make_request("agent:RunTool", "arn:tool:bash"))
        assert result.decision == Decision.DENIED_IMPLICIT
        assert result.matched_statements == []

    def test_no_matching_statement_implicit_deny(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {"Effect": "Allow", "Action": ["agent:Read"], "Resource": ["*"]}
                ],
            }
        )
        ev = PolicyEvaluator(store)
        result = ev.evaluate(make_request("agent:Write", "arn:tool:bash"))
        assert result.decision == Decision.DENIED_IMPLICIT


class TestExplicitAllow:
    """Verify that matching Allow statements grant access."""

    def test_wildcard_allow(self):
        ev = PolicyEvaluator(make_store(ALLOW_ALL))
        result = ev.evaluate(make_request("agent:RunTool", "arn:tool:bash"))
        assert result.decision == Decision.ALLOWED

    def test_exact_match_allow(self):
        ev = PolicyEvaluator(make_store(ALLOW_AGENT_TOOLS))
        result = ev.evaluate(make_request("agent:RunTool", "arn:tool:bash"))
        assert result.decision == Decision.ALLOWED

    def test_action_wildcard_resource_exact(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["arn:tool:bash"],
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:RunTool", "arn:tool:bash")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:RunTool", "arn:tool:python")).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_question_mark_wildcard(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:Ru?Tool"],
                        "Resource": ["*"],
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:RunTool", "*")).decision == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:ReadTool", "*")).decision
            == Decision.DENIED_IMPLICIT
        )


class TestExplicitDeny:
    """Verify that Deny statements take precedence over Allow statements."""

    def test_deny_overrides_allow(self):
        store = make_store(ALLOW_ALL, DENY_ALL)
        ev = PolicyEvaluator(store)
        result = ev.evaluate(make_request("agent:RunTool", "arn:tool:bash"))
        assert result.decision == Decision.DENIED_EXPLICIT

    def test_deny_specific_action(self):
        store = make_store(
            ALLOW_ALL,
            {
                "Version": "2025-01",
                "Statement": [
                    {"Effect": "Deny", "Action": ["agent:Delete"], "Resource": ["*"]}
                ],
            },
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Delete", "arn:tool:bash")).decision
            == Decision.DENIED_EXPLICIT
        )
        assert (
            ev.evaluate(make_request("agent:RunTool", "arn:tool:bash")).decision
            == Decision.ALLOWED
        )

    def test_deny_matched_statements_populated(self):
        ev = PolicyEvaluator(make_store(DENY_ALL))
        result = ev.evaluate(make_request("agent:RunTool", "*"))
        assert result.decision == Decision.DENIED_EXPLICIT
        assert len(result.matched_statements) == 1
        assert result.matched_statements[0].effect == "Deny"

    def test_conditional_deny_with_unmet_condition_should_not_deny(self):
        """Deny with unmet condition should NOT deny (allow passes through)."""
        store = make_store(
            ALLOW_ALL,
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Deny",
                        "Action": ["agent:Delete"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": "prod"}},
                    }
                ],
            },
        )
        ev = PolicyEvaluator(store)
        # Request with env=dev should NOT be denied by the conditional deny
        result = ev.evaluate(make_request("agent:Delete", "arn:tool:bash", env="dev"))
        assert result.decision == Decision.ALLOWED

    def test_conditional_deny_with_met_condition_should_deny(self):
        """Deny with met condition should deny."""
        store = make_store(
            ALLOW_ALL,
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Deny",
                        "Action": ["agent:Delete"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": "prod"}},
                    }
                ],
            },
        )
        ev = PolicyEvaluator(store)
        # Request with env=prod SHOULD be denied by the conditional deny
        result = ev.evaluate(make_request("agent:Delete", "arn:tool:bash", env="prod"))
        assert result.decision == Decision.DENIED_EXPLICIT
        assert len(result.matched_statements) == 1
        assert result.matched_statements[0].effect == "Deny"

    def test_unconditional_deny_still_works(self):
        """Deny without conditions should deny unconditionally."""
        store = make_store(
            ALLOW_ALL,
            {
                "Version": "2025-01",
                "Statement": [
                    {"Effect": "Deny", "Action": ["agent:Delete"], "Resource": ["*"]}
                ],
            },
        )
        ev = PolicyEvaluator(store)
        # Any request matching the deny should be denied regardless of context
        result = ev.evaluate(make_request("agent:Delete", "arn:tool:bash", env="dev"))
        assert result.decision == Decision.DENIED_EXPLICIT
        result = ev.evaluate(make_request("agent:Delete", "arn:tool:bash", env="prod"))
        assert result.decision == Decision.DENIED_EXPLICIT

    def test_conditional_deny_with_missing_context_key_should_not_deny(self):
        """Deny with condition referencing missing key should NOT deny.

        Important security edge case: _conditions_satisfied() returns False when
        a context key referenced in a condition is absent from the request. This
        means a conditional Deny can be bypassed by omitting the context key.
        """
        store = make_store(
            ALLOW_ALL,
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Deny",
                        "Action": ["agent:Delete"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": "prod"}},
                    }
                ],
            },
        )
        ev = PolicyEvaluator(store)
        # Request WITHOUT env key should NOT be denied (condition evaluates to False)
        result = ev.evaluate(make_request("agent:Delete", "arn:tool:bash"))
        assert result.decision == Decision.ALLOWED

    def test_multiple_denies_mixed_conditions(self):
        """Multiple deny statements with mixed condition satisfaction."""
        store = make_store(
            ALLOW_ALL,
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Deny",
                        "Action": ["agent:Delete"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": "prod"}},
                    },
                    {
                        "Effect": "Deny",
                        "Action": ["agent:Delete"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"region": "eu-west"}},
                    },
                ],
            },
        )
        ev = PolicyEvaluator(store)
        # Neither condition met → request allowed
        result = ev.evaluate(
            make_request("agent:Delete", "*", env="dev", region="us-east")
        )
        assert result.decision == Decision.ALLOWED
        # First condition met → denied
        result = ev.evaluate(
            make_request("agent:Delete", "*", env="prod", region="us-east")
        )
        assert result.decision == Decision.DENIED_EXPLICIT
        # Second condition met → denied
        result = ev.evaluate(
            make_request("agent:Delete", "*", env="dev", region="eu-west")
        )
        assert result.decision == Decision.DENIED_EXPLICIT
        # Both conditions met → denied (explicit deny)
        result = ev.evaluate(
            make_request("agent:Delete", "*", env="prod", region="eu-west")
        )
        assert result.decision == Decision.DENIED_EXPLICIT


class TestConditions:
    """Verify condition evaluation for all supported operators."""

    def test_string_equals_satisfied(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": "prod"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:RunTool", "*", env="prod")).decision
            == Decision.ALLOWED
        )

    def test_string_equals_not_satisfied(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": "prod"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:RunTool", "*", env="dev")).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_string_equals_missing_key(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": "prod"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # No env key in context
        assert (
            ev.evaluate(make_request("agent:RunTool", "*")).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_string_not_equals(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"StringNotEquals": {"env": "prod"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:RunTool", "*", env="dev")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:RunTool", "*", env="prod")).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_string_like(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"StringLike": {"user": "alice*"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", user="alice123")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", user="bob")).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_string_not_like(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"StringNotLike": {"user": "admin*"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", user="alice")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", user="admin_user")).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_numeric_equals(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericEquals": {"level": 5}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", level=5)).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", level=3)).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_numeric_not_equals(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericNotEquals": {"level": 0}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", level=1)).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", level=0)).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_numeric_less_than(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericLessThan": {"age": 18}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", age=10)).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", age=20)).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_numeric_greater_than(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericGreaterThan": {"score": 50}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", score=100)).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", score=50)).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_numeric_less_than_or_equal(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericLessThanEquals": {"count": 10}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", count=10)).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", count=11)).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_numeric_greater_than_or_equal(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericGreaterThanEquals": {"count": 10}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", count=10)).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", count=9)).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_multiple_conditions_anded(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {
                            "StringEquals": {"env": "prod"},
                            "NumericGreaterThan": {"level": 3},
                        },
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", env="prod", level=5)).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", env="prod", level=2)).decision
            == Decision.DENIED_IMPLICIT
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", env="dev", level=5)).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_multiple_values_ored(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"StringEquals": {"env": ["prod", "staging"]}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", env="prod")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", env="staging")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", env="dev")).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_unknown_condition_operator_denies(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"UnknownOp": {"env": "prod"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", env="prod")).decision
            == Decision.DENIED_IMPLICIT
        )


class TestBoolCondition:
    """Verify Bool condition operator behavior."""

    def test_bool_true_satisfied(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"Bool": {"filesystem:IsDirectory": "true"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(
                make_request("agent:Read", "*", **{"filesystem:IsDirectory": "true"})
            ).decision
            == Decision.ALLOWED
        )

    def test_bool_false_satisfied(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"Bool": {"filesystem:IsDirectory": "false"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(
                make_request("agent:Read", "*", **{"filesystem:IsDirectory": "false"})
            ).decision
            == Decision.ALLOWED
        )

    def test_bool_case_insensitive(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"Bool": {"flag": "True"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # Context value with different case
        assert (
            ev.evaluate(make_request("agent:Run", "*", flag="TRUE")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", flag="true")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", flag="True")).decision
            == Decision.ALLOWED
        )

    def test_bool_python_bool_values(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"Bool": {"flag": True}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # Python bool values should work
        assert (
            ev.evaluate(make_request("agent:Run", "*", flag=True)).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", flag=False)).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_bool_not_satisfied(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"Bool": {"flag": "true"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", flag="false")).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_bool_missing_key(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"Bool": {"flag": "true"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # No flag in context
        assert (
            ev.evaluate(make_request("agent:Run", "*")).decision
            == Decision.DENIED_IMPLICIT
        )

    def test_bool_multiple_values_ored(self):
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"Bool": {"flag": ["true", "false"]}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # Either value should match
        assert (
            ev.evaluate(make_request("agent:Run", "*", flag="true")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", flag="false")).decision
            == Decision.ALLOWED
        )

    def test_bool_non_boolean_value(self):
        """Bool operator compares stringified values, so non-bools work too."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"Bool": {"status": "yes"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        assert (
            ev.evaluate(make_request("agent:Run", "*", status="yes")).decision
            == Decision.ALLOWED
        )
        assert (
            ev.evaluate(make_request("agent:Run", "*", status="no")).decision
            == Decision.DENIED_IMPLICIT
        )


class TestMatchedStatementsInResult:
    """Verify the matched_statements field in AuthorizationResult."""

    def test_allow_matched_statements(self):
        ev = PolicyEvaluator(make_store(ALLOW_ALL))
        result = ev.evaluate(make_request("agent:Run", "*"))
        assert result.decision == Decision.ALLOWED
        assert len(result.matched_statements) == 1

    def test_implicit_deny_no_statements(self):
        ev = PolicyEvaluator(make_store(ALLOW_AGENT_TOOLS))
        result = ev.evaluate(make_request("agent:Delete", "arn:tool:bash"))
        assert result.decision == Decision.DENIED_IMPLICIT
        assert result.matched_statements == []


class TestInfNanGuard:
    """Verify that inf and nan values are rejected in numeric conditions."""

    def test_context_value_inf_denied(self):
        """Inf in context should deny, preventing NumericLessThan bypass."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericLessThan": {"count": 100}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # "inf" would bypass NumericLessThan (anything < inf is True)
        result = ev.evaluate(make_request("agent:Run", "*", count="inf"))
        assert result.decision == Decision.DENIED_IMPLICIT

    def test_context_value_infinity_denied(self):
        """'infinity' string should also be denied."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericLessThan": {"count": 100}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        result = ev.evaluate(make_request("agent:Run", "*", count="infinity"))
        assert result.decision == Decision.DENIED_IMPLICIT

    def test_context_value_nan_denied(self):
        """Nan in context should deny (nan comparisons are always False)."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericEquals": {"count": 42}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        result = ev.evaluate(make_request("agent:Run", "*", count="nan"))
        assert result.decision == Decision.DENIED_IMPLICIT

    def test_condition_value_inf_denied(self):
        """Inf in condition value should also deny."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericGreaterThan": {"score": "inf"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # No finite number is > inf, but we should reject before comparison
        result = ev.evaluate(make_request("agent:Run", "*", score=1000))
        assert result.decision == Decision.DENIED_IMPLICIT

    def test_condition_value_nan_denied(self):
        """Nan in condition value should deny."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericEquals": {"count": "nan"}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        result = ev.evaluate(make_request("agent:Run", "*", count=42))
        assert result.decision == Decision.DENIED_IMPLICIT

    def test_negative_inf_denied(self):
        """Negative infinity should also be rejected."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericGreaterThan": {"count": -1000}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        result = ev.evaluate(make_request("agent:Run", "*", count="-inf"))
        assert result.decision == Decision.DENIED_IMPLICIT

    def test_valid_float_still_works(self):
        """Normal float values should still work after inf/nan guard."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericLessThan": {"count": 100}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # String representation of a valid float
        result = ev.evaluate(make_request("agent:Run", "*", count="50.5"))
        assert result.decision == Decision.ALLOWED

    def test_valid_int_still_works(self):
        """Integer values should still work."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericEquals": {"count": 42}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        result = ev.evaluate(make_request("agent:Run", "*", count=42))
        assert result.decision == Decision.ALLOWED


class TestInvalidNumericConditionValues:
    """Verify that invalid numeric values in condition lists are gracefully skipped.

    Tests for issue #131: Invalid numeric condition value should not poison
    entire OR list. The values list in numeric conditions is treated as OR -
    any match should satisfy. Invalid (un-parseable) values should be skipped,
    not cause immediate failure.
    """

    def test_numeric_equals_mixed_valid_invalid_values_matches_valid(self):
        """NumericEquals with mixed valid/invalid values should still match valid ones.

        Per issue #131: [25, "bad", 30] with context=30 should match (skip "bad").
        """
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericEquals": {"age": [25, "bad", 30]}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # Should match the 30 value (skipping "bad")
        result = ev.evaluate(make_request("agent:Run", "*", age=30))
        assert result.decision == Decision.ALLOWED

    def test_numeric_equals_all_invalid_values_fails_condition(self):
        """NumericEquals with all-invalid values list should fail the condition."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {
                            "NumericEquals": {"age": ["bad", "worse", "ugly"]}
                        },
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # No valid values to match against
        result = ev.evaluate(make_request("agent:Run", "*", age=30))
        assert result.decision == Decision.DENIED_IMPLICIT

    def test_numeric_less_than_mixed_valid_invalid_values(self):
        """NumericLessThan with mixed values should skip invalid values."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {
                            "NumericLessThan": {"count": ["invalid", 100, "bad"]}
                        },
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # count=50 < 100 (should match after skipping "invalid")
        result = ev.evaluate(make_request("agent:Run", "*", count=50))
        assert result.decision == Decision.ALLOWED

    def test_numeric_greater_than_mixed_valid_invalid_values(self):
        """NumericGreaterThan with mixed values should skip invalid values."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {
                            "NumericGreaterThan": {"score": ["bad", 50, "worse"]}
                        },
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # score=75 > 50 (should match after skipping "bad")
        result = ev.evaluate(make_request("agent:Run", "*", score=75))
        assert result.decision == Decision.ALLOWED

    def test_invalid_value_followed_by_valid_match(self):
        """Invalid value at start of list should not prevent later valid match."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericEquals": {"level": ["not-a-number", 5]}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # Should match level=5 after skipping "not-a-number"
        result = ev.evaluate(make_request("agent:Run", "*", level=5))
        assert result.decision == Decision.ALLOWED

    def test_valid_value_before_invalid_still_matches(self):
        """Valid value before invalid should still match."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericEquals": {"count": [25, "invalid", 30]}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # count=25 should match before even reaching "invalid"
        result = ev.evaluate(make_request("agent:Run", "*", count=25))
        assert result.decision == Decision.ALLOWED

    def test_inf_nan_in_condition_values_list_are_skipped(self):
        """inf/nan in condition values list should be skipped."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {"NumericEquals": {"count": ["inf", 42, "nan"]}},
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # Should match 42 after skipping inf and nan
        result = ev.evaluate(make_request("agent:Run", "*", count=42))
        assert result.decision == Decision.ALLOWED

    def test_mixed_invalid_types_in_values_list(self):
        """Various non-numeric types in values list should all be skipped."""
        store = make_store(
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["agent:*"],
                        "Resource": ["*"],
                        "Condition": {
                            "NumericEquals": {"id": [None, "string", [], 100]}
                        },
                    }
                ],
            }
        )
        ev = PolicyEvaluator(store)
        # Should match 100 after skipping None, string, and list
        result = ev.evaluate(make_request("agent:Run", "*", id=100))
        assert result.decision == Decision.ALLOWED
