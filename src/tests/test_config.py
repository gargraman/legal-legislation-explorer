"""Tests for src.config."""

import pytest

from src.config import Config, load_config


def test_defaults_when_env_empty(monkeypatch):
    for var in (
        "LEGISLATION_URL_PREFIX",
        "LEGISLATION_URI_LIST_FILE",
        "JSON_OUTPUT_DIR",
        "DEPTH_LIMIT",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEO4J_DATABASE",
        "BATCH_SIZE",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "AGENT_RETRIEVAL_K",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = load_config(load_dotenv_file=False)
    assert cfg.json_output_dir == "json_out"
    assert cfg.depth_limit == 2
    assert cfg.neo4j_database == "neo4j"
    assert cfg.batch_size == 32
    assert cfg.agent_retrieval_k == 10
    assert cfg.neo4j_uri is None


def test_reads_env(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://host:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("DEPTH_LIMIT", "5")
    monkeypatch.setenv("BATCH_SIZE", "64")

    cfg = load_config(load_dotenv_file=False)
    assert cfg.neo4j_uri == "bolt://host:7687"
    assert cfg.depth_limit == 5
    assert cfg.batch_size == 64


def test_blank_int_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DEPTH_LIMIT", "   ")
    cfg = load_config(load_dotenv_file=False)
    assert cfg.depth_limit == 2


def test_require_neo4j_raises_when_missing():
    cfg = Config()
    with pytest.raises(RuntimeError) as exc:
        cfg.require_neo4j()
    assert "NEO4J_URI" in str(exc.value)


def test_require_neo4j_ok_when_present():
    cfg = Config(neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert cfg.require_neo4j() is cfg
