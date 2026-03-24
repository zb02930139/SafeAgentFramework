"""PolicyEvaluator: evaluates authorization requests against a PolicyStore."""

from __future__ import annotations

import fnmatch
import math
from typing import Any

from safe_agent.iam.models import (
    AuthorizationRequest,
    AuthorizationResult,
    Decision,
    Statement,
)
from safe_agent.iam.policy import PolicyStore


def _safe_float(value: Any) -> float | None:
    """Safely convert a value to float, rejecting inf and nan.

    Args:
        value: The value to convert (typically a string or number).

    Returns:
        The float value if valid and finite, or ``None`` if conversion fails
        or the result is inf/nan.
    """
    try:
        f = float(value)
        if math.isinf(f) or math.isnan(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Condition operator helpers
# ---------------------------------------------------------------------------

_NUMERIC_OPS = {
    "NumericEquals",
    "NumericNotEquals",
    "NumericLessThan",
    "NumericGreaterThan",
    "NumericLessThanEquals",
    "NumericGreaterThanEquals",
}

_STRING_OPS = {
    "StringEquals",
    "StringNotEquals",
    "StringLike",
    "StringNotLike",
}

_BOOL_OPS = {
    "Bool",
}


def _match_string_op(operator: str, ctx_val: str, condition_val: str) -> bool:
    """Evaluate a single string condition comparison.

    Args:
        operator: One of the ``String*`` operator names.
        ctx_val: The actual value from the request context.
        condition_val: The pattern/value from the condition block.

    Returns:
        ``True`` if the comparison holds, ``False`` otherwise.

    Raises:
        ValueError: If *operator* is not a recognised string operator.
    """
    if operator == "StringEquals":
        return ctx_val == condition_val
    if operator == "StringNotEquals":
        return ctx_val != condition_val
    if operator == "StringLike":
        return fnmatch.fnmatchcase(ctx_val, condition_val)
    if operator == "StringNotLike":
        return not fnmatch.fnmatchcase(ctx_val, condition_val)
    raise ValueError(f"Unknown string operator: {operator!r}")  # pragma: no cover


def _match_numeric_op(operator: str, ctx_val: float, condition_val: float) -> bool:
    """Evaluate a single numeric condition comparison.

    Args:
        operator: One of the ``Numeric*`` operator names.
        ctx_val: The actual numeric value from the request context.
        condition_val: The numeric value from the condition block.

    Returns:
        ``True`` if the comparison holds, ``False`` otherwise.

    Raises:
        ValueError: If *operator* is not a recognised numeric operator.
    """
    if operator == "NumericEquals":
        return ctx_val == condition_val
    if operator == "NumericNotEquals":
        return ctx_val != condition_val
    if operator == "NumericLessThan":
        return ctx_val < condition_val
    if operator == "NumericGreaterThan":
        return ctx_val > condition_val
    if operator == "NumericLessThanEquals":
        return ctx_val <= condition_val
    if operator == "NumericGreaterThanEquals":
        return ctx_val >= condition_val
    raise ValueError(f"Unknown numeric operator: {operator!r}")  # pragma: no cover


def _match_bool_op(ctx_val: Any, condition_val: Any) -> bool:
    """Evaluate a boolean condition comparison.

    Both values are converted to lowercase strings and compared for equality.
    This supports common truthy/falsy representations like "true", "True",
    "false", "False", as well as boolean ``True``/``False`` values.

    Args:
        ctx_val: The actual value from the request context.
        condition_val: The expected boolean value from the condition block.

    Returns:
        ``True`` if both values normalize to the same boolean string.
    """
    return str(ctx_val).lower() == str(condition_val).lower()


def _evaluate_condition_block(
    condition: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    """Evaluate a full condition block against a request context.

    Multiple operators are ANDed together. For each operator, multiple
    context keys are ANDed. For each context key, multiple values are ORed.

    A missing context key causes the entire condition to fail (return ``False``).

    Args:
        condition: The ``Condition`` dict from a
            :class:`~safe_agent.iam.models.Statement`.
        context: The ``context`` dict from the
            :class:`~safe_agent.iam.models.AuthorizationRequest`.

    Returns:
        ``True`` if all condition clauses are satisfied, ``False`` otherwise.
    """
    for operator, key_value_map in condition.items():
        for ctx_key, allowed_values in key_value_map.items():
            if ctx_key not in context:
                return False
            ctx_val = context[ctx_key]
            # allowed_values may be a single value or a list
            values_list: list[Any] = (
                allowed_values if isinstance(allowed_values, list) else [allowed_values]
            )
            # OR across all values for this key
            matched_any = False
            if operator in _STRING_OPS:
                str_ctx = str(ctx_val)
                for v in values_list:
                    if _match_string_op(operator, str_ctx, str(v)):
                        matched_any = True
                        break
            elif operator in _NUMERIC_OPS:
                num_ctx = _safe_float(ctx_val)
                if num_ctx is None:
                    return False
                for v in values_list:
                    num_v = _safe_float(v)
                    if num_v is None:
                        return False
                    if _match_numeric_op(operator, num_ctx, num_v):
                        matched_any = True
                        break
            elif operator in _BOOL_OPS:
                for v in values_list:
                    if _match_bool_op(ctx_val, v):
                        matched_any = True
                        break
            else:
                # Unknown operator — fail safe
                return False

            if not matched_any:
                return False
    return True


class PolicyEvaluator:
    """Evaluates :class:`~safe_agent.iam.models.AuthorizationRequest` objects.

    Implements a 5-step AWS IAM-style evaluation algorithm:

    1. Default deny.
    2. Collect statements whose ``Action`` and ``Resource`` patterns match the
       request (using ``fnmatch``-style wildcards).
    3. If any matching statement has ``Effect: Deny`` → ``DENIED_EXPLICIT``.
    4. If any matching ``Allow`` statement has all conditions satisfied → ``ALLOWED``.
    5. Otherwise → ``DENIED_IMPLICIT``.

    Example:
        >>> store = PolicyStore()
        >>> store.add_policy(my_policy)
        >>> evaluator = PolicyEvaluator(store)
        >>> result = evaluator.evaluate(request)
    """

    def __init__(self, store: PolicyStore) -> None:
        """Initialise the evaluator with a :class:`PolicyStore`.

        Args:
            store: The :class:`PolicyStore` containing the policies to evaluate.
        """
        self._store = store

    def evaluate(self, request: AuthorizationRequest) -> AuthorizationResult:
        """Evaluate *request* against all loaded policies.

        Args:
            request: The :class:`~safe_agent.iam.models.AuthorizationRequest`
                to evaluate.

        Returns:
            An :class:`~safe_agent.iam.models.AuthorizationResult` describing
            the decision, reason, and matched statements.
        """
        # Step 1: Default deny
        all_statements = self._store.get_all_statements()

        # Step 2: Collect matching statements (action + resource wildcards)
        matching: list[Statement] = []
        for stmt in all_statements:
            if self._matches_action_resource(stmt, request):
                matching.append(stmt)

        # Step 3: Explicit Deny takes highest precedence
        deny_stmts = [s for s in matching if s.effect == "Deny"]
        if deny_stmts:
            return AuthorizationResult(
                decision=Decision.DENIED_EXPLICIT,
                reason="Explicit Deny statement matched.",
                matched_statements=deny_stmts,
            )

        # Step 4: Allow if any Allow statement has satisfied conditions
        allow_stmts = [s for s in matching if s.effect == "Allow"]
        satisfied_allows: list[Statement] = []
        for stmt in allow_stmts:
            if self._conditions_satisfied(stmt, request):
                satisfied_allows.append(stmt)

        if satisfied_allows:
            return AuthorizationResult(
                decision=Decision.ALLOWED,
                reason="Allow statement matched with satisfied conditions.",
                matched_statements=satisfied_allows,
            )

        # Step 5: Implicit deny
        return AuthorizationResult(
            decision=Decision.DENIED_IMPLICIT,
            reason="No Allow statement matched; implicit deny.",
            matched_statements=[],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_action_resource(
        stmt: Statement,
        request: AuthorizationRequest,
    ) -> bool:
        """Return ``True`` if *request* matches any action/resource pattern in *stmt*.

        Args:
            stmt: The statement to test.
            request: The incoming authorization request.

        Returns:
            ``True`` when both action and resource match at least one pattern.
        """
        action_match = any(
            fnmatch.fnmatchcase(request.action, pattern) for pattern in stmt.action
        )
        if not action_match:
            return False
        resource_match = any(
            fnmatch.fnmatchcase(request.resource, pattern) for pattern in stmt.resource
        )
        return resource_match

    @staticmethod
    def _conditions_satisfied(
        stmt: Statement,
        request: AuthorizationRequest,
    ) -> bool:
        """Return ``True`` if all conditions in *stmt* are satisfied by *request*.

        If the statement has no ``Condition`` block the result is always ``True``.

        Args:
            stmt: The statement whose conditions to evaluate.
            request: The incoming authorization request supplying context values.

        Returns:
            ``True`` when all conditions are satisfied (or there are none).
        """
        if stmt.condition is None:
            return True
        return _evaluate_condition_block(stmt.condition, request.context)
