"""Tests for PolicyStore."""

import json
from pathlib import Path

import pytest

from safe_agent.iam.models import Policy
from safe_agent.iam.policy import PolicyStore

VALID_POLICY = {
    "Version": "2025-01",
    "Statement": [
        {"Effect": "Allow", "Action": ["agent:*"], "Resource": ["*"]},
    ],
}


def write_json(directory: Path, filename: str, data: dict) -> Path:
    """Write a JSON file into *directory* and return its path."""
    p = directory / filename
    p.write_text(json.dumps(data))
    return p


class TestPolicyStoreLoad:
    """Tests for PolicyStore.load()."""

    def test_load_single_file(self, tmp_path):
        write_json(tmp_path, "policy1.json", VALID_POLICY)
        store = PolicyStore()
        store.load(tmp_path)
        stmts = store.get_all_statements()
        assert len(stmts) == 1
        assert stmts[0].effect == "Allow"

    def test_load_multiple_files(self, tmp_path):
        write_json(tmp_path, "a.json", VALID_POLICY)
        write_json(
            tmp_path,
            "b.json",
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Effect": "Deny",
                        "Action": ["agent:Delete"],
                        "Resource": ["*"],
                    },
                ],
            },
        )
        store = PolicyStore()
        store.load(tmp_path)
        stmts = store.get_all_statements()
        assert len(stmts) == 2

    def test_load_ignores_non_json(self, tmp_path):
        write_json(tmp_path, "policy.json", VALID_POLICY)
        (tmp_path / "readme.txt").write_text("ignored")
        store = PolicyStore()
        store.load(tmp_path)
        assert len(store.get_all_statements()) == 1

    def test_load_invalid_version(self, tmp_path):
        bad = {**VALID_POLICY, "Version": "1999-01"}
        write_json(tmp_path, "bad.json", bad)
        store = PolicyStore()
        with pytest.raises(ValueError, match="unrecognised policy version"):
            store.load(tmp_path)

    def test_load_missing_version(self, tmp_path):
        bad = {"Statement": [{"Effect": "Allow", "Action": ["*"], "Resource": ["*"]}]}
        write_json(tmp_path, "bad.json", bad)
        store = PolicyStore()
        with pytest.raises(ValueError):
            store.load(tmp_path)

    def test_load_invalid_statement(self, tmp_path):
        bad = {
            "Version": "2025-01",
            "Statement": [{"Effect": "Maybe", "Action": ["*"], "Resource": ["*"]}],
        }
        write_json(tmp_path, "bad.json", bad)
        store = PolicyStore()
        with pytest.raises(ValueError, match="policy validation failed"):
            store.load(tmp_path)

    def test_load_empty_directory(self, tmp_path):
        store = PolicyStore()
        store.load(tmp_path)
        assert store.get_all_statements() == []

    def test_load_after_freeze_raises(self, tmp_path):
        write_json(tmp_path, "policy.json", VALID_POLICY)
        store = PolicyStore()
        store.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            store.load(tmp_path)

    def test_partial_load_failure_leaves_store_unchanged(self, tmp_path):
        """If a file fails validation, store should remain empty."""
        # First file is valid
        write_json(tmp_path, "a.json", VALID_POLICY)
        # Second file is valid
        write_json(
            tmp_path,
            "b.json",
            {
                "Version": "2025-01",
                "Statement": [{"Effect": "Deny", "Action": ["*"], "Resource": ["*"]}],
            },
        )
        # Third file has invalid version
        write_json(tmp_path, "c.json", {**VALID_POLICY, "Version": "1999-01"})
        store = PolicyStore()
        with pytest.raises(ValueError, match="unrecognised policy version"):
            store.load(tmp_path)
        # Store should still be empty — no partial state
        assert store.get_all_statements() == []

    def test_successful_load_after_failed_attempt(self, tmp_path):
        """After a failed load, a corrected load should work correctly."""
        # Write a bad policy
        write_json(tmp_path, "bad.json", {**VALID_POLICY, "Version": "1999-01"})
        store = PolicyStore()
        with pytest.raises(ValueError):
            store.load(tmp_path)
        assert store.get_all_statements() == []

        # Remove bad file, write valid one
        (tmp_path / "bad.json").unlink()
        write_json(tmp_path, "good.json", VALID_POLICY)
        store.load(tmp_path)
        assert len(store.get_all_statements()) == 1

    def test_partial_load_with_existing_policies_preserves_them(self, tmp_path):
        """If load fails, existing policies added via add_policy should remain."""
        store = PolicyStore()
        # Add a policy manually first
        existing = Policy.model_validate(VALID_POLICY)
        store.add_policy(existing)
        assert len(store.get_all_statements()) == 1

        # Try to load a bad policy file
        write_json(tmp_path, "bad.json", {**VALID_POLICY, "Version": "1999-01"})
        with pytest.raises(ValueError):
            store.load(tmp_path)

        # Existing policy should still be there
        assert len(store.get_all_statements()) == 1


class TestPolicyStoreAddPolicy:
    """Tests for PolicyStore.add_policy()."""

    def test_add_valid_policy(self):
        policy = Policy.model_validate(VALID_POLICY)
        store = PolicyStore()
        store.add_policy(policy)
        assert len(store.get_all_statements()) == 1

    def test_add_invalid_version(self):
        # Policy model does not restrict version; PolicyStore.add_policy does
        policy = Policy.model_validate(
            {
                "Version": "bad-ver",
                "Statement": [{"Effect": "Allow", "Action": ["*"], "Resource": ["*"]}],
            }
        )
        store = PolicyStore()
        with pytest.raises(ValueError, match="Unrecognised policy version"):
            store.add_policy(policy)

    def test_add_after_freeze_raises(self):
        store = PolicyStore()
        store.freeze()
        policy = Policy.model_validate(VALID_POLICY)
        with pytest.raises(RuntimeError, match="frozen"):
            store.add_policy(policy)


class TestPolicyStoreFreeze:
    """Tests for PolicyStore.freeze()."""

    def test_freeze_prevents_add(self):
        store = PolicyStore()
        store.freeze()
        policy = Policy.model_validate(VALID_POLICY)
        with pytest.raises(RuntimeError):
            store.add_policy(policy)

    def test_freeze_idempotent(self):
        store = PolicyStore()
        store.freeze()
        store.freeze()  # second freeze should not raise

    def test_get_all_statements_after_freeze(self):
        policy = Policy.model_validate(VALID_POLICY)
        store = PolicyStore()
        store.add_policy(policy)
        store.freeze()
        assert len(store.get_all_statements()) == 1

    def test_cached_statements_returned_after_freeze(self):
        """Verify that get_all_statements returns the cached list after freeze."""
        policy = Policy.model_validate(VALID_POLICY)
        store = PolicyStore()
        store.add_policy(policy)
        store.freeze()
        first = store.get_all_statements()
        second = store.get_all_statements()
        assert first is second  # same object, not a copy

    def test_freeze_empty_store_returns_empty_list(self):
        """Freezing an empty store should return empty list from cache."""
        store = PolicyStore()
        store.freeze()
        assert store.get_all_statements() == []

    def test_double_freeze_preserves_cache(self):
        """Double freeze should still return valid cached statements."""
        policy = Policy.model_validate(VALID_POLICY)
        store = PolicyStore()
        store.add_policy(policy)
        store.freeze()
        first = store.get_all_statements()
        store.freeze()  # second freeze rebuilds cache
        second = store.get_all_statements()
        # Cache may be rebuilt on second freeze, but contents must match
        assert first == second


class TestGetAllStatements:
    """Tests for PolicyStore.get_all_statements()."""

    def test_empty_store(self):
        store = PolicyStore()
        assert store.get_all_statements() == []

    def test_order_preserved(self, tmp_path):
        write_json(
            tmp_path,
            "a.json",
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Sid": "First",
                        "Effect": "Allow",
                        "Action": ["a:Read"],
                        "Resource": ["*"],
                    },
                ],
            },
        )
        write_json(
            tmp_path,
            "b.json",
            {
                "Version": "2025-01",
                "Statement": [
                    {
                        "Sid": "Second",
                        "Effect": "Deny",
                        "Action": ["a:Write"],
                        "Resource": ["*"],
                    },
                ],
            },
        )
        store = PolicyStore()
        store.load(tmp_path)
        stmts = store.get_all_statements()
        assert stmts[0].sid == "First"
        assert stmts[1].sid == "Second"

    def test_returns_new_list_before_freeze(self):
        """Before freeze, each call rebuilds the list (not cached)."""
        policy = Policy.model_validate(VALID_POLICY)
        store = PolicyStore()
        store.add_policy(policy)
        first = store.get_all_statements()
        second = store.get_all_statements()
        assert first is not second  # different objects
        assert first == second  # but same contents
