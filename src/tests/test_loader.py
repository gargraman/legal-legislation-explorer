"""Tests for src.loader.

PySpark is intentionally not installed in the test environment, so these tests
cover what can be verified without Spark: the module imports cleanly, the
constraint/query DDL is well-formed, and the constraint setup issues every
statement. The Spark transform itself requires an integration environment.
"""

import pytest

import src.loader as loader
from src.loader import CONSTRAINTS, LegislationGraphLoader, setup_neo4j_constraints


def test_module_imports_without_pyspark():
    # In the test env pyspark is absent; the module must still import.
    assert loader._HAS_PYSPARK is False
    assert loader.col is None


def test_constraints_are_unique_and_well_formed():
    assert len(CONSTRAINTS) == 13
    for stmt in CONSTRAINTS:
        assert "IF NOT EXISTS" in stmt
        assert "IS UNIQUE" in stmt
    labels = [s.split("FOR (")[1].split(":")[1].split(")")[0].split(" ")[0] for s in CONSTRAINTS]
    assert "Legislation" in labels and "Commentary" in labels and "CitationSubRef" in labels


class _FakeSession:
    def __init__(self, recorder):
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.recorder.append(query)


class _FakeDriver:
    def __init__(self):
        self.recorder = []
        self.closed = False

    def session(self, database=None):
        return _FakeSession(self.recorder)

    def close(self):
        self.closed = True


def test_setup_neo4j_constraints_runs_all(monkeypatch):
    driver = _FakeDriver()
    import neo4j

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **k: driver)
    setup_neo4j_constraints("bolt://x", "u", "p", "neo4j")
    assert driver.recorder == CONSTRAINTS
    assert driver.closed is True


def test_load_raises_without_pyspark():
    ldr = LegislationGraphLoader("bolt://x", "u", "p", "json_out")
    with pytest.raises(RuntimeError, match="PySpark is required"):
        ldr.load_full_hierarchy_to_neo4j()
