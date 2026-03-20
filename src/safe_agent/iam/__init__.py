"""IAM Policy Engine for SafeAgent.

Provides AWS IAM-style policy evaluation with Allow/Deny statements,
wildcard action/resource matching, and condition operators.
"""

from safe_agent.iam.evaluator import PolicyEvaluator
from safe_agent.iam.models import (
    AuthorizationRequest,
    AuthorizationResult,
    Decision,
    Policy,
    Statement,
)
from safe_agent.iam.policy import PolicyStore

__all__ = [
    "PolicyEvaluator",
    "PolicyStore",
    "Policy",
    "Statement",
    "AuthorizationRequest",
    "AuthorizationResult",
    "Decision",
]
