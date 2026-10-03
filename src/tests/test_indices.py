"""Tests for src.indices."""

from src.indices import INDEX_NAMES, INDEX_STATEMENTS, create_indexes, verify_indexes


class _FakeResult:
    def __init__(self, data=None):
        self._data = data or []

    def consume(self):
        return None

    def data(self):
        return self._data


class _FakeSession:
    def __init__(self, recorder, data=None):
        self.recorder = recorder
        self._data = data or []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.recorder.append((query, params))
        return _FakeResult(self._data)


class _FakeDriver:
    def __init__(self, data=None):
        self.recorder = []
        self._data = data or []

    def session(self, database=None):
        self.recorder.append(("__session__", {"database": database}))
        return _FakeSession(self.recorder, self._data)


def test_index_statements_shape():
    assert len(INDEX_STATEMENTS) == len(INDEX_NAMES)
    for stmt in INDEX_STATEMENTS:
        assert "IF NOT EXISTS" in stmt
        assert stmt.upper().startswith("CREATE")
    # every declared name appears in exactly one statement
    for name in INDEX_NAMES:
        assert any(name in s for s in INDEX_STATEMENTS)


def test_create_indexes_runs_every_statement():
    driver = _FakeDriver()
    create_indexes(driver, "neo4j")
    ran = [q for (q, _p) in driver.recorder if q != "__session__"]
    assert ran == INDEX_STATEMENTS


def test_verify_indexes_passes_names():
    driver = _FakeDriver(data=[{"name": "legislation_uri_text", "state": "ONLINE"}])
    rows = verify_indexes(driver, "neo4j")
    assert rows and rows[0]["name"] == "legislation_uri_text"
    # the SHOW INDEXES query was issued with the name filter
    show = [p for (q, p) in driver.recorder if "SHOW INDEXES" in q]
    assert show and show[0]["names"] == INDEX_NAMES
