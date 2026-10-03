"""Shared pytest fixtures and lightweight fakes for the pipeline tests."""

from __future__ import annotations

import pathlib
import sys

import pytest

# Make the repo root importable so `import src.<module>` works regardless of
# where pytest is launched from.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


@pytest.fixture()
def sample_xml_bytes() -> bytes:
    return (DATA_DIR / "sample_act.xml").read_bytes()


@pytest.fixture()
def sample_soup(sample_xml_bytes):
    from bs4 import BeautifulSoup

    return BeautifulSoup(sample_xml_bytes, "xml")


@pytest.fixture()
def crawler(tmp_path):
    from src.crawler import LegislationCrawler

    return LegislationCrawler(
        url_prefix="http://www.legislation.gov.uk",
        json_output_dir=str(tmp_path / "json_out"),
        max_depth=2,
        cache_dir=str(tmp_path / ".cache"),
    )


class FakeAnalysis:
    """Records Cypher calls and returns scripted results.

    ``responses`` maps a substring of the query to the value returned. If no
    key matches, ``default`` is returned. Every call is recorded in ``calls``.
    """

    def __init__(self, responses=None, default=None):
        self.responses = responses or {}
        self.default = default if default is not None else []
        self.calls = []
        self.closed = False

    def run_query(self, query, params=None):
        self.calls.append((query, params or {}))
        for needle, value in self.responses.items():
            if needle in query:
                return value
        return self.default

    def run_query_viz(self, query, params=None):
        self.calls.append((query, params or {}))
        return {"nodes": [], "relationships": []}

    def close(self):
        self.closed = True


class FakeEmbeddings:
    def __init__(self, vector=None):
        self.vector = vector or [0.1, 0.2, 0.3]
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return self.vector


@pytest.fixture()
def fake_analysis_factory():
    return FakeAnalysis


@pytest.fixture()
def fake_embeddings():
    return FakeEmbeddings()
