"""Tests for src.queries (constants + synthetic-relationship builder)."""

from src.queries import (
    BUILD_LINKED_TO_QUERY,
    COLORS,
    SCHEMA_VIZ_QUERY,
    VIZ_QUERIES,
    build_linked_to,
)


def test_colors_cover_core_labels():
    for label in ("Legislation", "Part", "Chapter", "Section", "Paragraph"):
        assert label in COLORS and COLORS[label].startswith("#")


def test_viz_queries_return_paths():
    assert VIZ_QUERIES  # non-empty
    for name, cypher in VIZ_QUERIES.items():
        assert "RETURN p" in cypher, f"{name} should return a path variable p"


def test_schema_viz_filters_text_label():
    assert "Text" in SCHEMA_VIZ_QUERY
    assert "db.schema.visualization()" in SCHEMA_VIZ_QUERY


def test_build_linked_to_is_a_write_and_runs(fake_analysis_factory):
    assert "MERGE" in BUILD_LINKED_TO_QUERY and "LINKED_TO" in BUILD_LINKED_TO_QUERY
    analysis = fake_analysis_factory()
    build_linked_to(analysis)
    assert analysis.calls[-1][0] == BUILD_LINKED_TO_QUERY
