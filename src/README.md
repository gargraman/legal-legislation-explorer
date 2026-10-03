# `src/` — Python equivalents of the notebooks

This folder contains importable, testable Python versions of the project's
Jupyter notebooks. The original `.ipynb` files are unchanged and remain the
reference; these modules mirror their logic with the notebook-isolated cruft
(top-level execution, hard-coded env reads) refactored into functions.

> **Note — `crawler.py` intentionally diverges from `crawler.ipynb`.** The
> notebook fetches CLML XML synchronously with `requests`. `crawler.py` fetches
> via [crawl4ai](https://github.com/unclecode/crawl4ai) using its **HTTP-only**
> `AsyncHTTPCrawlerStrategy` (no browser — the browser strategy would render XML
> through Chromium and corrupt it), crawling one BFS frontier concurrently per
> wave. A `requests` fetcher remains as an automatic fallback (and the injectable
> `session=`/`fetcher=` seam keeps the tests offline). `crawl()` and `main()` are
> now coroutines; run via `python -m src.crawler` (which wraps `asyncio.run`).
>
> crawl4ai is pinned to `>=0.9,<1`: on that line `AsyncHTTPCrawlerStrategy` is
> **not** re-exported from the `crawl4ai` top-level package and must be imported
> from `crawl4ai.async_crawler_strategy` (verified against crawl4ai 0.9.4).

| Module | Mirrors | Responsibility |
|---|---|---|
| `config.py` | the shared `load_dotenv()` preamble | Resolve env vars into a `Config` (no import-time side effects) |
| `crawler.py` | `crawler.ipynb` | Crawl legislation.gov.uk CLML XML → per-Act JSON (async, via crawl4ai) |
| `loader.py` | `loader.ipynb` | PySpark load of JSON → Neo4j node/edge hierarchy |
| `vectorize.py` | `vectorize.ipynb` | Embed Text nodes, build vector/text indexes, assemble the agent |
| `indices.py` | `indices.ipynb` | Create supporting range/text/fulltext indexes |
| `queries.py` | `examples.ipynb` | Query gallery, synthetic `LINKED_TO` builder, PNG renders |

## Run order

```bash
python -m src.crawler     # needs LEGISLATION_URL_PREFIX, LEGISLATION_URI_LIST_FILE
python -m src.loader      # needs NEO4J_* ; requires pyspark + the Neo4j Spark connector
python -m src.vectorize   # needs NEO4J_* ; embeds + indexes + smoke-tests the agent
python -m src.indices     # needs NEO4J_*
python -m src.queries     # builds LINKED_TO edges + renders renderings/*.png
```

All configuration comes from the environment / `.env` (see `config.py`). Run
from the **repo root** so `vectorize.py` / `queries.py` can import
`neo4j_analysis` (they also add the repo root to `sys.path` as a fallback).

## Tests

```bash
pip install -r src/requirements.txt   # or at minimum the light deps below
python -m pytest src/tests -q
```

The suite runs without the heavy stack (pyspark / sentence-transformers /
langchain / playwright / **crawl4ai**) by mocking external systems; the light
deps it needs are `pytest beautifulsoup4 lxml neo4j requests tqdm python-dotenv
pandas`. The crawler's network layer is injected in tests (an async fake fetcher
or a fake `requests` session), so crawl4ai itself is not needed to run them.

What's covered:
- **crawler.py** — full CLML parsing (identifier, metadata, unapplied effects,
  super links, body hierarchy, schedules, explanatory notes, commentaries) plus
  the pure `_parse_document` link-discovery logic and the async `crawl` loop
  (JSON written, frontier expansion, file-cache reuse, dedup) driven through a
  fake fetcher. This is the deep-coverage module.
- **config.py** — env parsing, defaults, `require_neo4j`.
- **indices.py** — DDL shape + that every statement is executed.
- **loader.py** — imports without Spark; constraint DDL; `setup_neo4j_constraints`
  issues every statement; the Spark transform needs an integration env.
- **vectorize.py** — payload parsing, read-only guard, alternation normalization,
  embedding loop, index DDL, and the graph-tool methods against a fake Neo4j.
- **queries.py** — colour map, every viz query returns a path, `LINKED_TO` build.

## Issues / gaps fixed during conversion

1. **Embedding model mismatch (correctness bug).** `vectorize.ipynb` embedded
   *documents* with `BAAI/bge-base-en-v1.5` but the agent embedded *queries* with
   `nlpaueb/legal-bert-base-uncased`. Two different embedding spaces make vector
   search meaningless. Both sides now default to one `DEFAULT_EMBED_MODEL`
   (legal-bert, matching `app.py`), and `check_embedding_consistency()` warns if
   they ever differ. **Re-run `vectorize` to re-embed with a single model.**
2. **`super` shadowed the builtin** in the crawler → renamed `super_info`.
3. **Shadowed loop counter** — the paragraph loop reused `idx` from the section
   loop → renamed `para_idx`.
4. **Import-time side effects removed** — modules no longer connect to Neo4j,
   read `.env`, or load models on import, so they can be imported and tested.
5. **`requests` now declared** in `environment.yml` (previously a gap). It is the
   crawler's *fallback* fetcher; the default is crawl4ai (see note below).
6. **OpenAI fallback model** standardized to `gpt-5-mini` (matches `app.py`;
   the notebook used `gpt-5`).
