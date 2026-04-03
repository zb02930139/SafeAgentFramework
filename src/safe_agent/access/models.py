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

"""Pydantic models for the Access Policy Engine.

Defines the core data structures used for policy representation and
authorization request/response handling, using JSON field aliases.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Decision(StrEnum):
    """The outcome of an authorization evaluation.

    Attributes:
        ALLOWED: The request was explicitly allowed by a matching Allow statement.
        DENIED_EXPLICIT: The request was explicitly denied by a Deny statement.
        DENIED_IMPLICIT: No matching Allow statement was found; default deny applies.
    """

    ALLOWED = "ALLOWED"
    DENIED_EXPLICIT = "DENIED_EXPLICIT"
    DENIED_IMPLICIT = "DENIED_IMPLICIT"


class Statement(BaseModel):
    """A single policy statement describing an Allow or Deny rule.

    Attributes:
        sid: Optional statement identifier.
        effect: Whether this statement allows or denies matching requests.
        action: List of action patterns (supports ``*`` and ``?`` wildcards).
        resource: List of resource patterns (supports ``*`` and ``?`` wildcards).
        condition: Optional mapping of condition operator to key/value pairs.
    """

    model_config = ConfigDict(populate_by_name=True)

    sid: str | None = Field(default=None, alias="Sid")
    effect: Literal["Allow", "Deny"] = Field(alias="Effect")
    action: list[str] = Field(alias="Action", min_length=1)
    resource: list[str] = Field(alias="Resource", min_length=1)
    condition: dict[str, dict[str, Any]] | None = Field(default=None, alias="Condition")


class Policy(BaseModel):
    """A top-level access policy document.

    Attributes:
        version: The policy format version. Must be ``"2025-01"``.
        statements: The list of statements that make up this policy.
    """

    model_config = ConfigDict(populate_by_name=True)

    version: str = Field(alias="Version")
    statements: list[Statement] = Field(alias="Statement")


class AuthorizationRequest(BaseModel):
    """A request to be evaluated against loaded policies.

    Attributes:
        action: The action being requested (e.g. ``"agent:RunTool"``).
        resource: The resource being acted upon (e.g. ``"arn:tool:bash"``).
        context: Arbitrary key/value pairs used for condition evaluation.
    """

    action: str
    resource: str
    context: dict[str, Any] = Field(default_factory=dict)


class AuthorizationResult(BaseModel):
    """The result of evaluating an authorization request.

    Attributes:
        decision: The authorization decision.
        reason: A human-readable explanation of the decision.
        matched_statements: The statements that contributed to the decision.
    """

    decision: Decision
    reason: str
    matched_statements: list[Statement] = Field(default_factory=list)
