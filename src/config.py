"""Shared configuration loaded from environment variables / ``.env``.

Every notebook started with the same ``load_dotenv()`` + ``os.getenv(...)``
preamble. That logic is centralised here so the pipeline modules share one
source of truth and so importing a module has **no side effects** (the env is
only read when ``load_config()`` is called, never at import time).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Resolved configuration for the pipeline.

    Mirrors the environment variables used across the notebooks. Only the
    fields relevant to a given stage need to be populated.
    """

    # legislation.gov.uk crawl inputs
    legislation_url_prefix: Optional[str] = None
    legislation_uri_list_file: Optional[str] = None
    json_output_dir: str = "json_out"
    depth_limit: int = 2

    # Neo4j connection
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: str = "neo4j"

    # loader / batching
    batch_size: int = 32

    # LLM + retrieval (vectorize / agent)
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    agent_retrieval_k: int = 10

    def require_neo4j(self) -> "Config":
        """Return self after asserting Neo4j credentials are present.

        Raises:
            RuntimeError: if URI / user / password are not all set.
        """
        missing = [
            name
            for name, value in (
                ("NEO4J_URI", self.neo4j_uri),
                ("NEO4J_USER", self.neo4j_user),
                ("NEO4J_PASSWORD", self.neo4j_password),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing Neo4j credentials: " + ", ".join(missing)
            )
        return self


def _get_int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` when unset/blank."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def load_config(load_dotenv_file: bool = True) -> Config:
    """Build a :class:`Config` from the current environment.

    Args:
        load_dotenv_file: when True (default) load a local ``.env`` first,
            matching the notebooks' behaviour. Set False in tests to read only
            the explicitly-set environment.
    """
    if load_dotenv_file:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover - dotenv is a declared dep
            pass

    return Config(
        legislation_url_prefix=os.getenv("LEGISLATION_URL_PREFIX"),
        legislation_uri_list_file=os.getenv("LEGISLATION_URI_LIST_FILE"),
        json_output_dir=os.getenv("JSON_OUTPUT_DIR", "json_out"),
        depth_limit=_get_int("DEPTH_LIMIT", 2),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        batch_size=_get_int("BATCH_SIZE", 32),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        agent_retrieval_k=_get_int("AGENT_RETRIEVAL_K", 10),
    )
