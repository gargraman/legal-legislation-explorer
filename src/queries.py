"""Query + visualization gallery.

Python equivalent of ``examples.ipynb``. Collects the Cypher used to explore the
corpus into named constants, exposes the synthetic ``LINKED_TO`` builder (the
one *write* step the examples perform), and renders the ``renderings/*.png``
graph images via ``neo4j_viz`` + the headless-Chromium helper in
``neo4j_analysis.py``.

``neo4j_viz`` / ``Neo4jAnalysis`` are imported lazily so the constants and the
helper wiring can be imported and tested without those heavy dependencies.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

from .config import Config, load_config

# Node-label -> colour, used by every viz render.
COLORS: Dict[str, str] = {
    "Legislation": "#1f77b4",
    "Part": "#ff7f0e",
    "Chapter": "#2ca02c",
    "Section": "#d62728",
    "Paragraph": "#9467bd",
    "Schedule": "#8c564b",
    "ScheduleParagraph": "#e377c2",
    "ScheduleSubparagraph": "#7f7f7f",
    "Commentary": "#bcbd22",
    "Citation": "#17becf",
    "CitationSubRef": "#aec7e8",
    "ExplanatoryNotes": "#ffbb78",
    "ExplanatoryNotesParagraph": "#98df8a",
}

# Schema visualization (Text label filtered out).
SCHEMA_VIZ_QUERY = """
CALL db.schema.visualization()
YIELD nodes, relationships
WITH [n IN nodes WHERE labels(n)[0] <> 'Text'] AS filtered_nodes, relationships
WITH filtered_nodes,
    [r IN relationships WHERE startNode(r) IN filtered_nodes AND endNode(r) IN filtered_nodes] AS filtered_rels
RETURN filtered_nodes AS nodes, filtered_rels AS relationships
"""

# The one WRITE in the examples: build synthetic LINKED_TO edges by collapsing
# structural citation paths between Acts into a single weighted relationship.
BUILD_LINKED_TO_QUERY = """
MATCH p = (source:Legislation)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_PARAGRAPH|HAS_SCHEDULE|HAS_SUBPARAGRAPH|HAS_COMMENTARY|HAS_CITATION|HAS_SUBREF*1..10]->(citation_link)
MATCH (citation_link)-[:CITES|REFERENCES]->(target:Legislation)
WHERE source.uri <> target.uri
  AND ALL(n IN nodes(p)[1..] WHERE NOT n:Legislation)
WITH source, target, count(citation_link) AS citation_count
MERGE (source)-[rel:LINKED_TO]->(target)
SET rel.weight = citation_count
"""

# Path-returning queries rendered to PNGs (name -> (cypher, output filename)).
VIZ_QUERIES: Dict[str, str] = {
    "legislation_example": """
        MATCH p=(l:Legislation)-[:HAS_PART]->(:Part)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(:Section)
        WHERE l.uri CONTAINS "ukpga/2010/4"
        RETURN p
    """,
    "legislation_example_detail": """
        MATCH p=(l:Legislation)-[:HAS_PART]->(part:Part)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(section:Section)-[:HAS_PARAGRAPH]->(para:Paragraph)-[:HAS_COMMENTARY]->(comm:Commentary)
        WHERE l.uri CONTAINS "ukpga/2010/4" AND part.order=2
        RETURN p
    """,
    "commentary_network": """
        MATCH p=(:Commentary)-[:HAS_CITATION]->(:Citation)-[:CITES]->(l:Legislation)
        WHERE l.uri CONTAINS "ukpga/2018/12"
        RETURN p
    """,
    "schedules": """
        MATCH p=(l:Legislation)-[:HAS_SCHEDULE]->(sc:Schedule)-[:HAS_PARAGRAPH]->(scp:ScheduleParagraph)-[:HAS_SUBPARAGRAPH]->(scsp:ScheduleSubparagraph)
        WHERE l.uri CONTAINS "ukpga/2010/4"
        OPTIONAL MATCH (scp)-[:HAS_COMMENTARY]-(:Commentary)-[:HAS_CITATION]-(:Citation)-[:HAS_SUBREF]->(:CitationSubRef)
        RETURN p
    """,
    "citation_network": """
        MATCH p = (l1:Legislation)-[r:LINKED_TO]->(l2:Legislation)
        RETURN p
        LIMIT 100
    """,
    "superseded_network": """
        MATCH p=(:Legislation)-[:SUPERSEDED_BY|SUPERSEDES]-(:Legislation)
        RETURN p
    """,
    "point_in_time_legislation": """
        MATCH p=(l:Legislation)-[:HAS_PART]->(part:Part)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SECTION]->(section:Section)-[:HAS_PARAGRAPH]->(para:Paragraph)-[:HAS_COMMENTARY]->(comm:Commentary)
        WHERE l.uri CONTAINS "ukpga/2010/4" AND (para.restrict_start_date >= date("2020-01-01") OR para.restrict_start_date IS NULL)
        RETURN p
    """,
    "shortest_path": """
        MATCH (source:Legislation) MATCH (target:Legislation)
        WHERE source.title = "Value Added Tax Act 1994" AND target.title = "Gambling Act 2005"
        MATCH p = shortestPath((source)-[:LINKED_TO*1..10]-(target))
        RETURN p
    """,
    "filtered_shortest_path": """
        MATCH (source:Legislation {title: "Value Added Tax Act 1994"})
        MATCH (target:Legislation {title: "Gambling Act 2005"})
        MATCH p = allShortestPaths((source)-[:LINKED_TO*1..5]-(target))
        WHERE ALL(n IN nodes(p)[1..-1] WHERE n.enactment_date.year <= 1990)
        RETURN p
    """,
    "data_protection_network": """
        MATCH (l:Legislation)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_PARAGRAPH|HAS_SCHEDULE*1..6]->(element)
        WHERE (element:Chapter OR element:Section OR element:Paragraph)
          AND toLower(element.title) CONTAINS toLower("data protection")
        WITH collect(DISTINCT l) AS topic_acts
        UNWIND topic_acts AS source_act
        MATCH p = (source_act)-[r:LINKED_TO]->(target_act:Legislation)
        WHERE target_act IN topic_acts AND source_act.enactment_date.year > 2000 AND target_act.enactment_date.year > 2000
        RETURN p
    """,
}

# Tabular queries returned as DataFrames.
CORPUS_QUERY = """
MATCH p=(l:Legislation)
RETURN l.category AS Category, l.status AS Status, l.title AS Title, l.uri AS URI, l.enactment_date AS Enactment
ORDER BY Enactment
"""


def build_linked_to(analysis) -> None:
    """Create/refresh the synthetic LINKED_TO relationships (a write op)."""
    analysis.run_query(BUILD_LINKED_TO_QUERY)


async def render_query_to_png(
    analysis,
    query: str,
    output_path: str,
    width: int = 1080,
    layout: str = "forcedirected",
) -> None:
    """Render a path-returning query to a PNG using neo4j_viz + Chromium."""
    from neo4j_viz.neo4j import ColorSpace, from_neo4j

    results = analysis.run_query_viz(query)
    vg = from_neo4j(results)
    vg.color_nodes(field="caption", color_space=ColorSpace.DISCRETE, colors=COLORS)
    generated_html = vg.render(layout=layout)
    await analysis.capture_graph_to_png(generated_html, output_path, width=width)


async def render_all(analysis, output_dir: str = "renderings") -> None:
    """Render every query in VIZ_QUERIES to ``<output_dir>/<name>.png``."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    # Schema first, then the gallery.
    await render_query_to_png(
        analysis, SCHEMA_VIZ_QUERY, os.path.join(output_dir, "schema_graph.png")
    )
    for name, query in VIZ_QUERIES.items():
        await render_query_to_png(analysis, query, os.path.join(output_dir, f"{name}.png"))


def _get_analysis(config: Config):
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from neo4j_analysis import Neo4jAnalysis

    return Neo4jAnalysis(
        config.neo4j_uri, config.neo4j_user, config.neo4j_password, config.neo4j_database
    )


def main(config: Optional[Config] = None, build_links: bool = True) -> None:
    """Build LINKED_TO edges then render the full image gallery."""
    config = (config or load_config()).require_neo4j()
    analysis = _get_analysis(config)
    try:
        if build_links:
            build_linked_to(analysis)
        asyncio.run(render_all(analysis))
    finally:
        analysis.close()


if __name__ == "__main__":
    main()
