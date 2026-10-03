"""Tests for src.vectorize (helpers + graph tools, no live DB/model)."""

import pytest

from src import vectorize as V
from src.vectorize import (
    CREATE_VECTOR_INDEX_QUERY,
    DEFAULT_EMBED_MODEL,
    LegislationTools,
    check_embedding_consistency,
    embed_texts,
    is_write_query,
    normalize_rel_alternation,
    parse_payload,
    run_embeddings,
)


# ---------------------------------------------------------------- pure helpers
def test_parse_payload_json():
    assert parse_payload('{"q":"tax","k":5}') == {"q": "tax", "k": 5}


def test_parse_payload_plain_text_becomes_q():
    assert parse_payload("corporation tax") == {"q": "corporation tax"}


def test_parse_payload_empty():
    assert parse_payload("") == {}
    assert parse_payload(None) == {}


@pytest.mark.parametrize(
    "q,expected",
    [
        ("MATCH (n) RETURN n", False),
        ("CREATE (n:X)", True),
        ("match (n) merge (m)", True),
        ("MATCH (n) DETACH DELETE n", True),
        ("MATCH (n) SET n.x = 1", True),
    ],
)
def test_is_write_query(q, expected):
    assert is_write_query(q) is expected


def test_normalize_rel_alternation():
    assert normalize_rel_alternation("[:A|:B|:C]") == "[:A|B|C]"
    assert normalize_rel_alternation("[:A|B]") == "[:A|B]"


# ----------------------------------------------------- embedding-model gap fix
def test_default_model_used_for_both_sides():
    # Document and query embedding default to the same model (the bug fix).
    assert DEFAULT_EMBED_MODEL == "nlpaueb/legal-bert-base-uncased"
    assert check_embedding_consistency(DEFAULT_EMBED_MODEL, DEFAULT_EMBED_MODEL) is True


def test_check_embedding_consistency_warns_on_mismatch():
    with pytest.warns(UserWarning, match="mismatch"):
        ok = check_embedding_consistency("BAAI/bge-base-en-v1.5", DEFAULT_EMBED_MODEL)
    assert ok is False


def test_vector_index_query_has_dims_and_similarity():
    assert "768" in CREATE_VECTOR_INDEX_QUERY
    assert "cosine" in CREATE_VECTOR_INDEX_QUERY
    assert "text_embeddings_index" in CREATE_VECTOR_INDEX_QUERY


# --------------------------------------------------------------- embedding fns
class _FakeVectors:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _FakeModel:
    def encode(self, texts, **kwargs):
        return _FakeVectors([[0.0, 1.0] for _ in texts])


def test_embed_texts_returns_lists():
    out = embed_texts(_FakeModel(), ["a", "b"])
    assert out == [[0.0, 1.0], [0.0, 1.0]]


class _ScriptedAnalysis:
    """Returns fetch rows once, then empty; records update payloads."""

    def __init__(self):
        self.fetch_calls = 0
        self.updates = []

    def run_query(self, query, params=None):
        if "n.text_embedding IS NULL" in query:
            self.fetch_calls += 1
            if self.fetch_calls == 1:
                return [{"node_id": "n1", "text": "hello"}, {"node_id": "n2", "text": "world"}]
            return []
        if "SET n.text_embedding" in query:
            self.updates.append(params["rows"])
            return [{"updated": len(params["rows"])}]
        return []


def test_run_embeddings_loops_until_exhausted():
    analysis = _ScriptedAnalysis()
    total = run_embeddings(analysis, _FakeModel(), fetch_limit=10, batch_size=2)
    assert total == 2
    assert analysis.fetch_calls == 2  # second fetch returns [] and stops
    assert analysis.updates[0][0]["node_id"] == "n1"


# ------------------------------------------------------------- graph tools
def test_vector_hits_blank_query_returns_empty(fake_analysis_factory, fake_embeddings):
    tools = LegislationTools(fake_analysis_factory(), fake_embeddings, k=5)
    assert tools._vector_hits("  ") == []
    assert fake_embeddings.queries == []  # never embedded a blank query


def test_find_legislation_no_hits(fake_analysis_factory, fake_embeddings):
    analysis = fake_analysis_factory(responses={"db.index.vector.queryNodes": []})
    tools = LegislationTools(analysis, fake_embeddings)
    assert tools.find_legislation('{"q":"tax"}') == []


def test_find_legislation_with_hits(fake_analysis_factory, fake_embeddings):
    analysis = fake_analysis_factory(
        responses={
            "db.index.vector.queryNodes": [{"node_id": "n1", "score": 0.9}],
            "RETURN DISTINCT l.title AS title": [{"title": "Corporation Tax Act 2010"}],
        }
    )
    tools = LegislationTools(analysis, fake_embeddings)
    out = tools.find_legislation('{"q":"corporation tax"}')
    assert out == [{"title": "Corporation Tax Act 2010"}]
    assert fake_embeddings.queries == ["corporation tax"]


def test_read_only_cypher_blocks_writes(fake_analysis_factory, fake_embeddings):
    tools = LegislationTools(fake_analysis_factory(), fake_embeddings)
    res = tools.read_only_cypher("CREATE (n:X)")
    assert "error" in res


def test_read_only_cypher_normalizes_and_runs(fake_analysis_factory, fake_embeddings):
    analysis = fake_analysis_factory(default=[{"ok": 1}])
    tools = LegislationTools(analysis, fake_embeddings)
    tools.read_only_cypher("MATCH (a)-[:A|:B]->(b) RETURN b")
    ran = analysis.calls[-1][0]
    assert "|:" not in ran  # alternation normalized


def test_legislation_by_uri_requires_input(fake_analysis_factory, fake_embeddings):
    tools = LegislationTools(fake_analysis_factory(), fake_embeddings)
    assert "error" in tools.legislation_by_uri("{}")


def test_citation_counts_requires_input(fake_analysis_factory, fake_embeddings):
    tools = LegislationTools(fake_analysis_factory(), fake_embeddings)
    assert "error" in tools.citation_counts("{}")


def test_hierarchy_path_resolver_requires_input(fake_analysis_factory, fake_embeddings):
    tools = LegislationTools(fake_analysis_factory(), fake_embeddings)
    assert "error" in tools.hierarchy_path_resolver("{}")


def test_hierarchy_path_resolver_by_uri_runs(fake_analysis_factory, fake_embeddings):
    analysis = fake_analysis_factory(default=[{"legislation_title": "X"}])
    tools = LegislationTools(analysis, fake_embeddings)
    out = tools.hierarchy_path_resolver('{"uri":"ukpga/2010/4"}')
    assert out == [{"legislation_title": "X"}]
    assert analysis.calls[-1][1] == {"uri": "ukpga/2010/4"}


def test_answer_grounder_dedups_evidence(fake_analysis_factory, fake_embeddings):
    tools = LegislationTools(fake_analysis_factory(), fake_embeddings)
    payload = (
        '{"answer":"A","evidence":['
        '{"uri":"u1","title":"t1","node_id":"n1"},'
        '{"uri":"u1","title":"t1","node_id":"n1"}]}'
    )
    out = tools.answer_grounder(payload)
    assert out["grounded_answer"] == "A"
    assert out["citation_count"] == 1  # duplicate collapsed


def test_build_tools_names_when_langchain_available(fake_analysis_factory, fake_embeddings):
    pytest.importorskip("langchain_core")
    tools = LegislationTools(fake_analysis_factory(), fake_embeddings).build_tools()
    names = {t.name for t in tools}
    assert "Graph_Schema_Navigator" in names
    assert "Read_Only_Cypher" in names
    assert "Citation_Counts" in names
    assert "Hierarchy_Path_Resolver" in names
    assert "Answer_Grounder" in names
