"""Create supporting Neo4j indexes.

Python equivalent of ``indices.ipynb``. The index DDL is exposed as the
module-level :data:`INDEX_STATEMENTS` so it can be inspected/tested without a
live database; :func:`create_indexes` applies them.
"""

from __future__ import annotations

from typing import List, Optional

from .config import Config, load_config

INDEX_STATEMENTS: List[str] = [
    # High-impact lookups
    "CREATE TEXT INDEX legislation_uri_text IF NOT EXISTS FOR (l:Legislation) ON (l.uri)",
    "CREATE RANGE INDEX legislation_uri_range IF NOT EXISTS FOR (l:Legislation) ON (l.uri)",
    "CREATE RANGE INDEX part_order_range IF NOT EXISTS FOR (p:Part) ON (p.order)",
    "CREATE RANGE INDEX paragraph_restrict_start_date_range IF NOT EXISTS FOR (p:Paragraph) ON (p.restrict_start_date)",
    "CREATE RANGE INDEX paragraph_restrict_end_date_range IF NOT EXISTS FOR (p:Paragraph) ON (p.restrict_end_date)",
    "CREATE TEXT INDEX paragraph_uri_text IF NOT EXISTS FOR (p:Paragraph) ON (p.uri)",
    "CREATE TEXT INDEX section_uri_text IF NOT EXISTS FOR (s:Section) ON (s.uri)",
    "CREATE FULLTEXT INDEX legislation_title_uri_fulltext IF NOT EXISTS FOR (l:Legislation) ON EACH [l.title, l.uri]",
]

# Names created above, used to verify index state after creation.
INDEX_NAMES: List[str] = [
    "legislation_uri_text",
    "legislation_uri_range",
    "part_order_range",
    "paragraph_restrict_start_date_range",
    "paragraph_restrict_end_date_range",
    "paragraph_uri_text",
    "section_uri_text",
    "legislation_title_uri_fulltext",
]


def create_indexes(driver, database: str, statements: Optional[List[str]] = None) -> None:
    """Run each index statement against ``database`` using an open driver."""
    statements = statements or INDEX_STATEMENTS
    with driver.session(database=database) as session:
        for stmt in statements:
            session.run(stmt).consume()
            print("OK:", stmt)


def verify_indexes(driver, database: str, names: Optional[List[str]] = None) -> list:
    """Return index state rows for the given index names."""
    names = names or INDEX_NAMES
    with driver.session(database=database) as session:
        rows = session.run(
            "SHOW INDEXES YIELD name, type, state, populationPercent, labelsOrTypes, properties "
            "WHERE name IN $names RETURN * ORDER BY name",
            names=names,
        ).data()
    return rows


def main(config: Optional[Config] = None) -> None:
    config = (config or load_config()).require_neo4j()
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
    )
    try:
        print("Connected to", config.neo4j_uri, "database=", config.neo4j_database)
        create_indexes(driver, config.neo4j_database)
        rows = verify_indexes(driver, config.neo4j_database)
        print(f"Verified {len(rows)} indexes.")
    finally:
        driver.close()
        print("Done.")


if __name__ == "__main__":
    main()
