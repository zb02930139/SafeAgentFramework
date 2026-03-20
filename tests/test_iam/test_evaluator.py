"""Tests for PolicyEvaluator."""

from safe_agent.iam.evaluator import PolicyEvaluator
from safe_agent.iam.models import (
    AuthorizationRequest,
    Decision,
    Policy,
)
from safe_agent.iam.policy import PolicyStore


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
