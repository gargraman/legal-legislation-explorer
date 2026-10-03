# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Neo4j knowledge graph of UK primary legislation (sourced from legislation.gov.uk's CLML XML) plus a Streamlit **GraphRAG agent** that answers legal questions over that graph. The pipeline is split across Jupyter notebooks (build the graph) and `app.py` (serve the agent).

The notebooks are also mirrored as an importable, testable Python package under **`src/`** (one module per notebook, a shared `config.py`, and a `pytest` suite in `src/tests/`); see `src/README.md`. The notebooks remain the reference and are what produced the images in `renderings/`. No linter is configured. When changing pipeline logic, keep the notebook and its `src/` counterpart in sync (the one deliberate exception is the crawler — see below).

Self-contained HTML architecture diagrams live in `docs/diagrams/` (`build-pipeline.html` — the crawl→load→vectorize→index flow; `graphrag-runtime.html` — the agent/tooling at query time).

## Environment & commands

```bash
# Create / activate the conda env (name: legal-legislation-explorer, Python 3.11)
conda env create -f environment.yml
conda activate legal-legislation-explorer

# Playwright browser is required by BOTH the crawler and PNG rendering in neo4j_analysis.py
playwright install chromium

# Run the agent app
streamlit run app.py

# Work the pipeline notebooks
jupyter lab
```

The pipeline also runs as Python modules (equivalent to the notebooks). Run from the **repo root**:

```bash
python -m src.crawler     # crawl CLML XML → per-Act JSON (async, via crawl4ai)
python -m src.loader      # PySpark load of JSON → Neo4j
python -m src.vectorize   # embed Text nodes, build indexes, smoke-test the agent
python -m src.indices     # supporting range/text/fulltext indexes
python -m src.queries     # build LINKED_TO edges + render renderings/*.png

# Tests (mock the heavy stack; light deps only — see src/README.md)
pip install -r src/requirements.txt
python -m pytest src/tests -q
```

All configuration is read from a `.env` file (loaded via `python-dotenv`; `.env` is gitignored). Required and notable vars:

- **Neo4j (everywhere):** `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` (default `neo4j`)
- **LLM (app.py):** `GOOGLE_API_KEY` **or** `OPENAI_API_KEY`. If `GOOGLE_API_KEY` is set it uses `gemini-2.5-flash`; otherwise falls back to `gpt-5-mini`.
- **Crawler:** `LEGISLATION_URL_PREFIX`, `LEGISLATION_URI_LIST_FILE`, `JSON_OUTPUT_DIR` (default `json_out`), `DEPTH_LIMIT` (default 2), `BATCH_SIZE`
- **Agent tuning:** `AGENT_RETRIEVAL_K` (10), `AGENT_HISTORY_MESSAGES` (20), `DEBUG_TOOL_CALLS` (set to any value to log tool calls in the UI)

## Pipeline — notebooks run in this order

The graph must be built before `app.py` or `examples.ipynb` will work. Each stage feeds the next:

1. **`crawler.ipynb`** — Recursive crawler. Fetches CLML XML from legislation.gov.uk (Playwright), parses it with BeautifulSoup/lxml, and writes one JSON document per legislation node tree to `json_out/` (gitignored). Seeded from `legislation_list.txt` (a list of `/ukpga/<year>/<number>` URIs). Also accumulates the observed XML schema as it crawls.
2. **`loader.ipynb`** — PySpark job. Reads `json_out/`, transforms the raw JSON, and bulk-`MERGE`s nodes and relationships into Neo4j. PySpark is used for throughput; the README notes the transform is adaptable to plain Python.
3. **`vectorize.ipynb`** — Embeds `Text` nodes with the `nlpaueb/legal-bert-base-uncased` SentenceTransformer and creates the Neo4j vector index **`text_embeddings_index`** (plus a fulltext `text_index`). The app reads these index names directly — keep them in sync if renamed.
4. **`indices.ipynb`** — Creates supporting range/text indexes (e.g. on `Legislation.uri`) for query performance.
5. **`examples.ipynb`** — Not part of the build; a gallery of Cypher queries and `neo4j_viz` visualizations that produced the images in `renderings/`.

## Graph schema (the mental model for any query work)

Root node is **`Legislation`** (properties: URI, title, type, enactment date). Structural hierarchy below it:

```
Legislation → Part → Chapter → Section → Paragraph
Legislation → Schedule → ScheduleParagraph
Legislation → ExplanatoryNotes
```

Plus cross-cutting nodes: **`Citation`** (external references), **`Commentary`** (annotations tied to provisions), and **`Text`** (the embedded/searchable text payloads). Temporal data lives as node properties — notably `restrict_start_date` / `restrict_end_date` — which drive the "point in time" and temporal-diff queries. **Always preserve the full hierarchy context (Legislation > Part > … > Paragraph) when returning content nodes**; the agent's Cypher prompt enforces this and content queries that return bare nodes are considered wrong.

## app.py architecture

A single large Streamlit module. Two things to know:

- **`build_runtime()`** (cached with `@st.cache_resource`) is the composition root. It builds the `Neo4jGraph`, the LLM, HuggingFace legal-bert embeddings, a `GraphCypherQAChain` (with a strict **read-only** Cypher prompt — no CREATE/MERGE/DELETE/SET), a `Neo4jVector` retriever over `text_embeddings_index`, and assembles a LangChain agent via `create_agent(llm, tools, system_prompt=...)`. It returns `(analysis, agent_executor)`.
- **The agent has ~13 tools** (see the `tools = [...]` list ~line 862). They are deliberately layered from specific → general: `Graph_Schema_Navigator`, `Legislation_Title_Resolver`, `Legislation_Finder` (hybrid title+vector), `Contextual_Text_Retriever`, citation/supersedes tools, `Semantic_Search`, and `Text2Cypher_Expert` as the last-resort general query tool. The system prompt tells the model to prefer granular tools before Text2Cypher. When adding a tool, register it in this list and update the system prompt's guidance so the model knows when to reach for it.

The sidebar `st.radio` switches between a **Chat Interface** (the agent) and a set of **use-case graph visualizations** (Legislation Graph, Parts, Commentaries, Schedules, Supersedes, Point in Time, Temporal Diff, etc.), each rendered by `_render_use_case_graph` from a Cypher query returning a path variable `p`.

## neo4j_analysis.py

`Neo4jAnalysis` is the shared Neo4j helper used by both `app.py` and the notebooks: query runners (`run_query`, `run_query_df`, `run_query_single`, `run_query_viz`) and `capture_graph_to_png`, which renders a `neo4j_viz` HTML graph to PNG via headless Chromium (Playwright) — it waits 8s for the force-directed layout to settle and writes a temporary `remove.html`. This is what generated `renderings/*.png`.

## Conventions worth preserving

- **Read-only graph access from the app.** The Cypher-generation prompt and the agent tools are intentionally constrained to read-only queries. Do not relax this when extending the agent. Graph writes belong in the loader/vectorize notebooks only.
- **Vector/fulltext index names are a contract** between `vectorize.ipynb` and `app.py` (`text_embeddings_index`, `text_index`). Changing one requires changing the other.
- **The `src/` crawler diverges from `crawler.ipynb` on purpose.** The notebook fetches synchronously with `requests`; `src/crawler.py` fetches CLML XML asynchronously via **crawl4ai** using its HTTP-only `AsyncHTTPCrawlerStrategy` (no browser — a headless browser renders XML through its XML viewer and corrupts the raw CLML), with a `requests` fetcher as automatic fallback. **crawl4ai is pinned to `>=0.9,<1`:** on that line `AsyncHTTPCrawlerStrategy` is not re-exported from the `crawl4ai` top level and must be imported from `crawl4ai.async_crawler_strategy` (a plain `from crawl4ai import AsyncHTTPCrawlerStrategy` raises ImportError). Verified against crawl4ai 0.9.4.
- The schema is still evolving (per the README): node ordering isn't fully implemented, and some XML constructs — `<InlineAmendment>`, `<Substitution>`, `<Addition>`, unapplied effects — are not yet extracted.
