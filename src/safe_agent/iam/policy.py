"""PolicyStore for loading and managing IAM policy documents."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from safe_agent.iam.models import Policy, Statement

VALID_VERSIONS = {"2025-01"}


class PolicyStore:
    """Stores and manages a collection of IAM policies.

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

    def load(self, directory: Path) -> None:
        """Load all ``.json`` policy files from *directory*.

        Each file is parsed as a :class:`~safe_agent.iam.models.Policy`.
        Files whose ``Version`` field is not ``"2025-01"`` are rejected
        with a :class:`ValueError`.

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
            self._policies.append(policy)

    def add_policy(self, policy: Policy) -> None:
        """Add a single policy to the store.

        Args:
            policy: A validated :class:`~safe_agent.iam.models.Policy` instance.

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
        """Make the store immutable.

        After calling :meth:`freeze`, any call to :meth:`add_policy` or
        :meth:`load` will raise a :class:`RuntimeError`.
        """
        self._frozen = True

    def get_all_statements(self) -> list[Statement]:
        """Return all statements across all loaded policies.

        Returns:
            A flat list of :class:`~safe_agent.iam.models.Statement` objects
            in the order they were added.
        """
        statements: list[Statement] = []
        for policy in self._policies:
            statements.extend(policy.statements)
        return statements
