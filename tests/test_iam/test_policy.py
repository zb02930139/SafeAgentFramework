"""Tests for PolicyStore."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from safe_agent.iam.models import Policy, Statement
from safe_agent.iam.policy import PolicyStore

VALID_POLICY = {
    "Version": "2025-01",
    "Statement": [
        {"Effect": "Allow", "Action": ["agent:*"], "Resource": ["*"]},
    ],
}


def write_json(directory: Path, filename: str, data: dict) -> Path:
    p = directory / filename
    p.write_text(json.dumps(data))
    return p


class TestPolicyStoreLoad:
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
                    {"Effect": "Deny", "Action": ["agent:Delete"], "Resource": ["*"]},
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


class TestPolicyStoreAddPolicy:
    def test_add_valid_policy(self):
        policy = Policy.model_validate(VALID_POLICY)
        store = PolicyStore()
        store.add_policy(policy)
        assert len(store.get_all_statements()) == 1

    def test_add_invalid_version(self):
        # Policy model does not restrict version; PolicyStore.add_policy does
        policy = Policy.model_validate(
            {"Version": "bad-ver", "Statement": [{"Effect": "Allow", "Action": ["*"], "Resource": ["*"]}]}
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


class TestGetAllStatements:
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
                    {"Sid": "First", "Effect": "Allow", "Action": ["a:Read"], "Resource": ["*"]},
                ],
            },
        )
        write_json(
            tmp_path,
            "b.json",
            {
                "Version": "2025-01",
                "Statement": [
                    {"Sid": "Second", "Effect": "Deny", "Action": ["a:Write"], "Resource": ["*"]},
                ],
            },
        )
        store = PolicyStore()
        store.load(tmp_path)
        stmts = store.get_all_statements()
        assert stmts[0].sid == "First"
        assert stmts[1].sid == "Second"
