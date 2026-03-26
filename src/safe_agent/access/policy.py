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

"""PolicyStore for loading and managing access policy documents."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from safe_agent.access.models import Policy, Statement

VALID_VERSIONS = {"2025-01"}


class PolicyStore:
    """Stores and manages a collection of access policies.

    Policies can be loaded from JSON files or added programmatically.
    Once frozen, no further policies may be added.

    Example:
        >>> store = PolicyStore()
        >>> store.load(Path("/etc/agent/policies"))
        >>> store.freeze()
        >>> statements = store.get_all_statements()
    """

    def __init__(self) -> None:
        """Initialise an empty, mutable policy store."""
        self._policies: list[Policy] = []
        self._frozen: bool = False
        self._cached_statements: tuple[Statement, ...] | None = None

    def load(self, directory: Path) -> None:
        """Load all ``.json`` policy files from *directory*.

        Each file is parsed as a :class:`~safe_agent.access.models.Policy`.
        Files whose ``Version`` field is not ``"2025-01"`` are rejected
        with a :class:`ValueError`.

        This method uses an atomic all-or-nothing pattern: all policies are
        validated first, and only committed to the store if all succeed.
        If any file fails validation, the store remains unchanged.

        Args:
            directory: Path to a directory containing ``.json`` policy files.

        Raises:
            ValueError: If a file contains an unrecognised ``Version`` value
                or fails Pydantic validation.
            RuntimeError: If the store has been frozen via :meth:`freeze`.
            FileNotFoundError: If *directory* does not exist.
        """
        if self._frozen:
            raise RuntimeError(
                "PolicyStore is frozen; no further policies may be added."
            )
        # Stage all policies first; only commit if all succeed
        staged: list[Policy] = []
        for json_file in sorted(directory.glob("*.json")):
            raw = json_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            version = data.get("Version", "")
            if version not in VALID_VERSIONS:
                raise ValueError(
                    f"{json_file.name}: unrecognised policy version {version!r}. "
                    f"Expected one of {sorted(VALID_VERSIONS)}."
                )
            try:
                policy = Policy.model_validate(data)
            except ValidationError as exc:
                raise ValueError(
                    f"{json_file.name}: policy validation failed: {exc}"
                ) from exc
            staged.append(policy)
        # All validated successfully — commit atomically
        self._policies.extend(staged)

    def add_policy(self, policy: Policy) -> None:
        """Add a single policy to the store.

        Args:
            policy: A validated :class:`~safe_agent.access.models.Policy` instance.

        Raises:
            ValueError: If the policy's ``version`` is not ``"2025-01"``.
            RuntimeError: If the store has been frozen via :meth:`freeze`.
        """
        if self._frozen:
            raise RuntimeError(
                "PolicyStore is frozen; no further policies may be added."
            )
        if policy.version not in VALID_VERSIONS:
            raise ValueError(
                f"Unrecognised policy version {policy.version!r}. "
                f"Expected one of {sorted(VALID_VERSIONS)}."
            )
        self._policies.append(policy)

    def freeze(self) -> None:
        """Make the store immutable and cache statements for fast access.

        After calling :meth:`freeze`, any call to :meth:`add_policy` or
        :meth:`load` will raise a :class:`RuntimeError`.

        The cached statements are stored as an immutable tuple to enforce
        the frozen state guarantee — callers cannot mutate the returned
        sequence and corrupt the policy store's internal state.
        """
        # Build cache as immutable tuple before setting flag to avoid race condition
        self._cached_statements = tuple(s for p in self._policies for s in p.statements)
        self._frozen = True

    def get_all_statements(self) -> Sequence[Statement]:
        """Return all statements across all loaded policies.

        Once the store is frozen, this returns a cached tuple for performance.
        Before freezing, the list is rebuilt on each call.

        The returned sequence is immutable after freeze — callers cannot
        modify the cached statements.

        Returns:
            A flat sequence of :class:`~safe_agent.access.models.Statement` objects
            in the order they were added.
        """
        if self._frozen:
            # Cache is populated by freeze() before _frozen is set
            return cast(tuple[Statement, ...], self._cached_statements)
        statements: list[Statement] = []
        for policy in self._policies:
            statements.extend(policy.statements)
        return statements
