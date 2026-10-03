"""Embed Text nodes, build the vector/text indexes, and assemble the agent.

Python equivalent of ``vectorize.ipynb``. Three concerns, split into testable
pieces:

  1. Embedding  -- :func:`load_embedding_model`, :func:`embed_texts`,
     :func:`run_embeddings` populate ``n.text_embedding`` on content nodes.
  2. Indexing   -- :func:`label_text_nodes`, :func:`create_vector_index`,
     :func:`create_text_index`.
  3. Agent      -- :class:`LegislationTools` (graph query tools) and
     :func:`build_agent` (LLM + tools), mirroring ``app.py``'s runtime.

Fixed gap (embedding model mismatch): the notebook embedded documents with
``BAAI/bge-base-en-v1.5`` but the agent embedded *queries* with
``nlpaueb/legal-bert-base-uncased``. Mixing two embedding spaces makes vector
search semantically meaningless even though both are 768-dim. Here both sides
default to the SAME model (:data:`DEFAULT_EMBED_MODEL`), and
:func:`check_embedding_consistency` warns loudly if they are ever set to differ.
Heavy imports (torch, sentence-transformers, langchain) are loaded lazily so
this module and its tests import without them.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any, Callable, List, Optional

from .config import Config, load_config

# One model for BOTH document embedding and query embedding (see module docstring).
DEFAULT_EMBED_MODEL = "nlpaueb/legal-bert-base-uncased"
# The model the original notebook used to embed documents (kept for reference).
LEGACY_DOC_EMBED_MODEL = "BAAI/bge-base-en-v1.5"

VECTOR_INDEX_NAME = "text_embeddings_index"
TEXT_INDEX_NAME = "text_index"
VECTOR_DIMENSIONS = 768
VECTOR_SIMILARITY = "cosine"

# ---------------------------------------------------------------- Cypher DDL/DQL
FETCH_QUERY = """
MATCH (n)
WHERE n.text_embedding IS NULL
  AND (n.text IS NOT NULL OR n.title IS NOT NULL OR n.description IS NOT NULL)
WITH n,
     trim(
       coalesce(n.title + '\\n', '') +
       coalesce(n.description + '\\n', '') +
       coalesce(n.text, '')
     ) AS combinedText
WHERE combinedText <> ''
RETURN elementId(n) AS node_id, combinedText AS text
LIMIT $limit
"""

UPDATE_QUERY = """
UNWIND $rows AS row
MATCH (n)
WHERE elementId(n) = row.node_id
SET n.text_embedding = row.embedding
RETURN count(n) AS updated
"""

LABEL_TEXT_QUERY = """
MATCH (n)
WHERE n.text_embedding IS NOT NULL
SET n:Text
"""

CREATE_VECTOR_INDEX_QUERY = f"""
CREATE VECTOR INDEX `{VECTOR_INDEX_NAME}` IF NOT EXISTS
FOR (n:Text) ON (n.text_embedding)
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {VECTOR_DIMENSIONS},
    `vector.similarity_function`: '{VECTOR_SIMILARITY}'
  }}
}}
"""

CREATE_TEXT_INDEX_QUERY = f"""
CREATE TEXT INDEX `{TEXT_INDEX_NAME}` IF NOT EXISTS
FOR (n:Text) ON (n.text)
"""

CYPHER_PROMPT_TEMPLATE = """You are an expert Neo4j Cypher generator for a UK legislation graph.
Generate ONLY a valid read-only Cypher query.

Graph schema:
{schema}

Rules you MUST follow:
1) Return ONLY Cypher. No markdown, no commentary.
2) Read-only queries only. Never use CREATE, MERGE, DELETE, SET, CALL dbms.*, or schema/index changes.
3) When searching for titles, themes or topics, prefer the semantic search tool.
4) Prefer exact property names above and valid relationship directions.
5) When user names an Act/title, match with case-insensitive containment.
6) When user references a known legislation.gov.uk id, filter by l.uri CONTAINS 'ukpga/2010/4' style.
7) For network/visualization requests, return a path variable `p` (e.g., MATCH p=... RETURN p).
8) For tabular requests, RETURN explicit aliased columns and use ORDER BY/LIMIT when reasonable.
9) Avoid Cartesian products; always connect patterns.
10) Use OPTIONAL MATCH only when truly optional.
11) Keep traversal bounded for path exploration (e.g., *1..6 or *1..10).
12) CONTEXT IS MANDATORY for structural/text nodes. Include parent context up to Legislation.
13) Do not return a bare content node alone unless explicitly requested.
14) For relationship alternation, use ONE leading colon only, e.g. [:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_SCHEDULE*0..3]. Never write [:HAS_PART|:HAS_CHAPTER|...].

Question: {question}"""

SYSTEM_PROMPT = """You are a highly capable legal AI assistant.
Use the most specific tool first. Use the Graph_Schema_Navigator before other tools to understand the schema. Prefer granular tools before Text2Cypher_Expert.
For retrieval tasks, prefer vector-index-backed tools (Legislation_Finder, Legislation_Context_Summary, Contextual_Text_Retriever, Citation_Network_Explorer, Semantic_Search).
Always preserve legal context (Legislation > Part > Chapter > Section > Paragraph) when answering content questions.
If a tool returns empty results, do not repeat the exact same call. Always include links to relevant legislation, sections and parts in your responses.
Use Legislation_By_URI for exact act lookup, Hierarchy_Path_Resolver for context reconstruction, Citation_Counts for quick citation metrics, and Answer_Grounder to ground final responses."""

_WRITE_CLAUSE_RE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|DROP|REMOVE)\b", re.IGNORECASE)


# ----------------------------------------------------------------- pure helpers
def parse_payload(payload: Any) -> dict:
    """Parse a tool payload (JSON string) into a dict, tolerating plain text."""
    if not payload or not str(payload).strip():
        return {}
    try:
        return json.loads(payload)
    except Exception:
        return {"q": payload}


def is_write_query(payload: str) -> bool:
    """True if the Cypher contains a write clause (CREATE/MERGE/DELETE/...)."""
    return bool(_WRITE_CLAUSE_RE.search(payload or ""))


def normalize_rel_alternation(payload: str) -> str:
    """Fix a common generated-Cypher mistake: ``[:A|:B]`` -> ``[:A|B]``."""
    return (payload or "").replace("|:", "|")


def check_embedding_consistency(doc_model: str, query_model: str) -> bool:
    """Warn if document- and query-embedding models differ. Returns True if OK."""
    if doc_model != query_model:
        warnings.warn(
            "Embedding model mismatch: documents embedded with "
            f"'{doc_model}' but queries with '{query_model}'. Vector search "
            "requires the SAME model on both sides to be meaningful.",
            stacklevel=2,
        )
        return False
    return True


# -------------------------------------------------------------------- embedding
def load_embedding_model(model_name: str = DEFAULT_EMBED_MODEL, max_seq_length: int = 512):
    """Load a SentenceTransformer on the best available device."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    return model


def embed_texts(model, texts: List[str], batch_size: int = 128) -> List[list]:
    """Encode texts to a list of (normalized) embedding vectors."""
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vectors.tolist()


def run_embeddings(analysis, model, fetch_limit: int = 5000, batch_size: int = 128) -> int:
    """Embed every un-embedded content node in batches. Returns total updated."""
    total_updated = 0
    while True:
        rows = analysis.run_query(FETCH_QUERY, {"limit": fetch_limit})
        if not rows:
            break
        texts = [r["text"] for r in rows]
        vectors = embed_texts(model, texts, batch_size=batch_size)
        payload = [
            {"node_id": r["node_id"], "embedding": v} for r, v in zip(rows, vectors)
        ]
        result = analysis.run_query(UPDATE_QUERY, {"rows": payload})
        updated_now = result[0]["updated"] if result else 0
        total_updated += updated_now
        print(f"Updated {updated_now} nodes (total: {total_updated})")
    print(f"Done. Total embedded nodes: {total_updated}")
    return total_updated


# --------------------------------------------------------------------- indexing
def label_text_nodes(analysis) -> None:
    analysis.run_query(LABEL_TEXT_QUERY)


def create_vector_index(analysis) -> None:
    print("Creating vector index...")
    analysis.run_query(CREATE_VECTOR_INDEX_QUERY)
    print("Vector index created successfully!")


def create_text_index(analysis) -> None:
    print("Creating text index...")
    analysis.run_query(CREATE_TEXT_INDEX_QUERY)
    print("Text index created successfully!")


# ----------------------------------------------------------------- graph tools
class LegislationTools:
    """Graph query helpers backing the agent's tools.

    Mirrors the functions defined in ``vectorize.ipynb`` cell 4/5, parameterised
    over a :class:`Neo4jAnalysis` and an embeddings object so they can be
    unit-tested with fakes.
    """

    def __init__(self, analysis, embeddings, k: int = 10, vector_index_name: str = VECTOR_INDEX_NAME):
        self.analysis = analysis
        self.embeddings = embeddings
        self.k = k
        self.vector_index_name = vector_index_name

    def _vector_hits(self, query_text: str, k: Optional[int] = None) -> list:
        if not query_text or not query_text.strip():
            return []
        k = k or self.k
        embedding = self.embeddings.embed_query(query_text)
        query = """
        CALL db.index.vector.queryNodes($index_name, $k, $embedding)
        YIELD node, score
        RETURN elementId(node) AS node_id,
               labels(node) AS labels,
               score,
               coalesce(node.title, node.text, node.description) AS matched_content,
               node.title AS node_title,
               node.uri AS node_uri
        ORDER BY score DESC
        """
        return self.analysis.run_query(
            query, {"index_name": self.vector_index_name, "k": k, "embedding": embedding}
        )

    def schema_navigation(self, _: str = "") -> str:
        node_query = """
        CALL apoc.meta.data()
        YIELD label, property, type, elementType
        WHERE elementType = "node"
          AND type <> "RELATIONSHIP"
          AND label <> "Text"
        RETURN label, collect(property + ': ' + type) AS properties
        """
        nodes = self.analysis.run_query(node_query)

        rel_query = """
        MATCH (a)-[r]->(b)
        WITH [l IN labels(a) WHERE l <> 'Text'] AS start_labels,
            type(r) AS relationship_type,
            [l IN labels(b) WHERE l <> 'Text'] AS end_labels
        WHERE size(start_labels) > 0 AND size(end_labels) > 0
        UNWIND start_labels AS start_label
        UNWIND end_labels AS end_label
        RETURN DISTINCT start_label, relationship_type, end_label
        LIMIT 5000
        """
        rels = self.analysis.run_query(rel_query)

        schema_text = "GRAPH SCHEMA DEFINITION:\n\nNode Labels and Properties:\n"
        for node in nodes:
            props = ", ".join(node["properties"]) if node["properties"] else "No properties"
            schema_text += f"   - (:{node['label']} {{ {props} }})\n"

        schema_text += "\nValid Relationship Connections:\n"
        if rels:
            for rel in rels:
                schema_text += (
                    f"   - (:{rel['start_label']})-[:{rel['relationship_type']}]->"
                    f"(:{rel['end_label']})\n"
                )
        else:
            schema_text += "   - No relationships found.\n"
        return schema_text

    def find_legislation(self, payload: str):
        data = parse_payload(payload)
        q = data.get("q", "")
        k = int(data.get("k", self.k))
        hits = self._vector_hits(q, k=k)
        if not hits:
            return []
        query = """
        UNWIND $hits AS h
        MATCH (hit) WHERE elementId(hit) = h.node_id
        OPTIONAL MATCH (l_direct:Legislation) WHERE elementId(l_direct) = h.node_id
        OPTIONAL MATCH (l_ctx:Legislation)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_PARAGRAPH|HAS_SCHEDULE|HAS_SUBPARAGRAPH|HAS_EXPLANATORY_NOTES*1..6]->(hit)
        WITH h, coalesce(l_direct, l_ctx) AS l
        WHERE l IS NOT NULL
        RETURN DISTINCT l.title AS title,
                        l.uri AS uri,
                        l.enactment_date AS enactment_date,
                        l.status AS status,
                        l.category AS category,
                        h.score AS vector_score
        ORDER BY vector_score DESC, enactment_date DESC
        LIMIT 25
        """
        return self.analysis.run_query(query, {"hits": hits})

    def get_legislation_context(self, payload: str):
        data = parse_payload(payload)
        q = data.get("q", "")
        k = int(data.get("k", self.k))
        hits = self._vector_hits(q, k=k)
        if not hits:
            return []
        query = """
        UNWIND $hits AS h
        MATCH (hit) WHERE elementId(hit) = h.node_id
        OPTIONAL MATCH (l_direct:Legislation) WHERE elementId(l_direct) = h.node_id
        OPTIONAL MATCH (l_ctx:Legislation)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_PARAGRAPH|HAS_SCHEDULE|HAS_SUBPARAGRAPH|HAS_EXPLANATORY_NOTES*1..6]->(hit)
        WITH h, coalesce(l_direct, l_ctx) AS l
        WHERE l IS NOT NULL
        OPTIONAL MATCH (l)-[:HAS_PART]->(part:Part)
        OPTIONAL MATCH (part)-[:HAS_CHAPTER]->(chapter:Chapter)
        OPTIONAL MATCH (chapter)-[:HAS_SECTION]->(section:Section)
        OPTIONAL MATCH (section)-[:HAS_PARAGRAPH]->(paragraph:Paragraph)
        RETURN l.title AS legislation_title,
               l.uri AS legislation_uri,
               count(DISTINCT part) AS part_count,
               count(DISTINCT chapter) AS chapter_count,
               count(DISTINCT section) AS section_count,
               count(DISTINCT paragraph) AS paragraph_count,
               max(h.score) AS vector_score
        ORDER BY vector_score DESC, paragraph_count DESC
        LIMIT 5
        """
        return self.analysis.run_query(query, {"hits": hits})

    def retrieve_text_with_context(self, payload: str):
        data = parse_payload(payload)
        q = data.get("q", "")
        k = int(data.get("k", self.k))
        limit = int(data.get("limit", 15))
        hits = self._vector_hits(q, k=k)
        if not hits:
            return []
        query = """
        UNWIND $hits AS h
        MATCH (n) WHERE elementId(n) = h.node_id
        OPTIONAL MATCH p=(l:Legislation)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_PARAGRAPH|HAS_SCHEDULE|HAS_SUBPARAGRAPH|HAS_EXPLANATORY_NOTES*0..6]->(n)
        WITH h, n, l, p,
             head([x IN nodes(p) WHERE x:Part]) AS part,
             head([x IN nodes(p) WHERE x:Chapter]) AS chapter,
             head([x IN nodes(p) WHERE x:Section]) AS section,
             head([x IN nodes(p) WHERE x:Paragraph]) AS paragraph
        WHERE l IS NOT NULL
        RETURN DISTINCT l.title AS legislation_title,
               l.uri AS legislation_uri,
               part.number AS part_number,
               part.title AS part_title,
               chapter.number AS chapter_number,
               chapter.title AS chapter_title,
               section.number AS section_number,
               section.title AS section_title,
               paragraph.number AS paragraph_number,
               coalesce(paragraph.text, n.text, n.title, n.description) AS matched_text,
               h.score AS vector_score
        ORDER BY vector_score DESC
        LIMIT $limit
        """
        return self.analysis.run_query(query, {"hits": hits, "limit": limit})

    def citation_reasoning(self, payload: str):
        data = parse_payload(payload)
        q = data.get("q", "")
        k = int(data.get("k", self.k))
        hits = self._vector_hits(q, k=k)
        if not hits:
            return []
        query = """
        UNWIND $hits AS h
        MATCH (hit) WHERE elementId(hit) = h.node_id
        OPTIONAL MATCH (source_direct:Legislation) WHERE elementId(source_direct) = h.node_id
        OPTIONAL MATCH (source_ctx:Legislation)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_PARAGRAPH|HAS_SCHEDULE|HAS_SUBPARAGRAPH|HAS_EXPLANATORY_NOTES*1..6]->(hit)
        WITH h, coalesce(source_direct, source_ctx) AS source
        WHERE source IS NOT NULL
        OPTIONAL MATCH (source)-[r:CITES]->(target:Legislation)
        RETURN source.title AS source_title,
               source.uri AS source_uri,
               target.title AS target_title,
               target.uri AS target_uri,
               type(r) AS relationship_type,
               h.score AS vector_score
        ORDER BY vector_score DESC
        LIMIT 20
        """
        return self.analysis.run_query(query, {"hits": hits})

    def supersedes_chain(self, payload: str):
        data = parse_payload(payload)
        q = data.get("q", "")
        query = """
        MATCH (source:Legislation)
        WHERE toLower(coalesce(source.title, "")) CONTAINS toLower($q)
           OR toLower(coalesce(source.uri, "")) CONTAINS toLower($q)
        OPTIONAL MATCH (source)-[:SUPERSEDES]->(target:Legislation)
        RETURN source.title AS source_title, source.uri AS source_uri,
               target.title AS target_title, target.uri AS target_uri
        LIMIT 20
        """
        return self.analysis.run_query(query, {"q": q})

    def superseded_chain(self, payload: str):
        data = parse_payload(payload)
        q = data.get("q", "")
        query = """
        MATCH (source:Legislation)
        WHERE toLower(coalesce(source.title, "")) CONTAINS toLower($q)
           OR toLower(coalesce(source.uri, "")) CONTAINS toLower($q)
        OPTIONAL MATCH (source)-[:SUPERSEDED_BY]->(target:Legislation)
        RETURN source.title AS source_title, source.uri AS source_uri,
               target.title AS target_title, target.uri AS target_uri
        LIMIT 20
        """
        return self.analysis.run_query(query, {"q": q})

    def read_only_cypher(self, payload: str):
        if is_write_query(payload):
            return {"error": "Only read-only Cypher is allowed in this tool."}
        return self.analysis.run_query(normalize_rel_alternation(payload))

    def legislation_by_uri(self, payload: str):
        data = parse_payload(payload)
        uri = data.get("uri") or data.get("q", "")
        if not uri:
            return {"error": "Provide 'uri' (or 'q') in payload."}
        query = """
        MATCH (l:Legislation)
        WHERE l.uri = $uri OR l.uri CONTAINS $uri
        RETURN l.title AS title, l.uri AS uri, l.enactment_date AS enactment_date,
               l.status AS status, l.category AS category
        ORDER BY l.enactment_date DESC
        LIMIT 5
        """
        return self.analysis.run_query(query, {"uri": uri})

    def citation_counts(self, payload: str):
        data = parse_payload(payload)
        q = data.get("uri") or data.get("q", "")
        if not q:
            return {"error": "Provide 'uri' or 'q'."}
        query = """
        MATCH (l:Legislation)
        WHERE l.uri = $q OR l.uri CONTAINS $q OR toLower(coalesce(l.title, "")) CONTAINS toLower($q)
        CALL(l) {
          WITH l
          OPTIONAL MATCH (l)-[:LINKED_TO]->(t:Legislation)
          RETURN count(DISTINCT t) AS outgoing_count, collect(DISTINCT t.title)[0..5] AS top_outgoing_titles
        }
        CALL(l) {
          WITH l
          OPTIONAL MATCH (s:Legislation)-[:LINKED_TO]->(l)
          RETURN count(DISTINCT s) AS incoming_count, collect(DISTINCT s.title)[0..5] AS top_incoming_titles
        }
        RETURN l.title AS legislation_title, l.uri AS legislation_uri,
               outgoing_count, incoming_count, top_outgoing_titles, top_incoming_titles
        LIMIT 5
        """
        return self.analysis.run_query(query, {"q": q})

    def hierarchy_path_resolver(self, payload: str):
        data = parse_payload(payload)
        node_id = data.get("node_id")
        uri = data.get("uri")
        if not node_id and not uri:
            return {"error": "Provide 'node_id' (elementId) or 'uri'."}

        return_block = """
            RETURN labels(n) AS node_labels,
                   coalesce(n.uri, n.id, elementId(n)) AS node_ref,
                   l.title AS legislation_title,
                   l.uri AS legislation_uri,
                   part.number AS part_number,
                   part.title AS part_title,
                   part.restrict_start_date AS part_restrict_start_date,
                   part.restrict_end_date AS part_restrict_end_date,
                   part.restrict_extent AS part_restrict_extent,
                   part.status AS part_status,
                   chapter.number AS chapter_number,
                   chapter.title AS chapter_title,
                   chapter.restrict_start_date AS chapter_restrict_start_date,
                   chapter.restrict_end_date AS chapter_restrict_end_date,
                   chapter.restrict_extent AS chapter_restrict_extent,
                   chapter.status AS chapter_status,
                   section.number AS section_number,
                   section.title AS section_title,
                   section.restrict_start_date AS section_restrict_start_date,
                   section.restrict_end_date AS section_restrict_end_date,
                   section.restrict_extent AS section_restrict_extent,
                   section.status AS section_status,
                   paragraph.number AS paragraph_number,
                   paragraph.restrict_start_date AS paragraph_restrict_start_date,
                   paragraph.restrict_end_date AS paragraph_restrict_end_date,
                   paragraph.restrict_extent AS paragraph_restrict_extent,
                   paragraph.status AS paragraph_status
            LIMIT 10
        """
        with_block = """
        OPTIONAL MATCH p=(l:Legislation)-[:HAS_PART|HAS_CHAPTER|HAS_SECTION|HAS_PARAGRAPH|HAS_SCHEDULE|HAS_SUBPARAGRAPH|HAS_EXPLANATORY_NOTES*0..6]->(n)
        WITH n, l, p,
             head([x IN nodes(p) WHERE x:Part]) AS part,
             head([x IN nodes(p) WHERE x:Chapter]) AS chapter,
             head([x IN nodes(p) WHERE x:Section]) AS section,
             head([x IN nodes(p) WHERE x:Paragraph]) AS paragraph
        """
        if node_id:
            query = "MATCH (n) WHERE elementId(n) = $node_id" + with_block + return_block
            return self.analysis.run_query(query, {"node_id": node_id})
        query = "MATCH (n) WHERE n.uri = $uri OR n.uri CONTAINS $uri" + with_block + return_block
        return self.analysis.run_query(query, {"uri": uri})

    def answer_grounder(self, payload: str):
        data = parse_payload(payload)
        answer = data.get("answer", "")
        evidence = data.get("evidence", [])
        question = data.get("q", "")

        if not evidence and question:
            hits = self._vector_hits(question, k=min(self.k, 8))
            evidence = [
                {
                    "node_id": h.get("node_id"),
                    "uri": h.get("node_uri"),
                    "title": h.get("node_title"),
                    "score": h.get("score"),
                }
                for h in hits
            ]

        normalized = []
        seen = set()
        for e in evidence if isinstance(evidence, list) else []:
            if not isinstance(e, dict):
                continue
            key = (e.get("uri"), e.get("node_id"), e.get("title"))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "uri": e.get("uri"),
                    "title": e.get("title"),
                    "node_id": e.get("node_id"),
                    "score": e.get("score"),
                }
            )
        return {
            "grounded_answer": answer,
            "citation_count": len(normalized),
            "citations": normalized,
        }

    def build_tools(self) -> list:
        """Build the LangChain Tool list (lazy langchain import)."""
        from langchain_core.tools import Tool

        return [
            Tool(
                name="Graph_Schema_Navigator",
                func=self.schema_navigation,
                description="Get graph labels, properties and relationship types. Input can be empty.",
            ),
            Tool(
                name="Legislation_Finder",
                func=self.find_legislation,
                description='Find legislation via vector index hits mapped to parent Legislation. Input JSON like {"q":"corporation tax","k":20}.',
            ),
            Tool(
                name="Legislation_Context_Summary",
                func=self.get_legislation_context,
                description='Get legislation hierarchy summary using vector-index seeds. Input JSON like {"q":"ukpga/2010/4","k":20}.',
            ),
            Tool(
                name="Contextual_Text_Retriever",
                func=self.retrieve_text_with_context,
                description='Retrieve context-rich text using vector index. Input JSON like {"q":"corporate tax","k":20,"limit":10}.',
            ),
            Tool(
                name="Citation_Network_Explorer",
                func=self.citation_reasoning,
                description="Use vector-index seeds to find relevant legislation, then expand outgoing citation links.",
            ),
            Tool(
                name="Supersedes_Network_Explorer",
                func=self.supersedes_chain,
                description='Get outgoing supersedes network. Input JSON like {"q":"Value Added Tax Act"}.',
            ),
            Tool(
                name="Superseded_By_Network_Explorer",
                func=self.superseded_chain,
                description='Get incoming superseded_by network. Input JSON like {"q":"Value Added Tax Act"}.',
            ),
            Tool(
                name="Read_Only_Cypher",
                func=self.read_only_cypher,
                description="Execute ad-hoc read-only Cypher. Input: Cypher query string.",
            ),
            Tool(
                name="Legislation_By_URI",
                func=self.legislation_by_uri,
                description='Deterministic legislation lookup by URI. Input JSON: {"uri":"http://www.legislation.gov.uk/..."}.',
            ),
            Tool(
                name="Citation_Counts",
                func=self.citation_counts,
                description='Get inbound/outbound citation counts and top linked Acts. Input JSON: {"q":"Value Added Tax Act"}.',
            ),
            Tool(
                name="Hierarchy_Path_Resolver",
                func=self.hierarchy_path_resolver,
                description='Resolve full hierarchy context for a node by elementId or uri. Input JSON: {"node_id":"..."} or {"uri":"..."}.',
            ),
            Tool(
                name="Answer_Grounder",
                func=self.answer_grounder,
                description='Return a grounded answer object with evidence/citations. Input JSON: {"answer":"...","evidence":[...]} or {"q":"..."}.',
            ),
        ]


# ---------------------------------------------------------------- agent wiring
def build_llm(config: Config):
    """Build the chat model: Gemini when GOOGLE_API_KEY is set, else OpenAI."""
    if config.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            api_key=config.google_api_key,
            include_thoughts=True,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-5-mini", temperature=0, api_key=config.openai_api_key)


def build_embeddings(model_name: str = DEFAULT_EMBED_MODEL):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name, encode_kwargs={"normalize_embeddings": True}
    )


def build_agent(analysis, config: Optional[Config] = None, embed_model: str = DEFAULT_EMBED_MODEL):
    """Assemble the LangChain agent (LLM + graph tools + semantic retriever)."""
    config = config or load_config()
    from langchain.agents import create_agent
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import Tool, create_retriever_tool
    from langchain_neo4j import GraphCypherQAChain, Neo4jGraph, Neo4jVector

    llm = build_llm(config)
    embeddings = build_embeddings(embed_model)

    graph = Neo4jGraph(
        url=config.neo4j_uri,
        username=config.neo4j_user,
        password=config.neo4j_password,
        database=config.neo4j_database,
    )
    cypher_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        llm=llm,
        cypher_prompt=PromptTemplate(
            input_variables=["schema", "question"], template=CYPHER_PROMPT_TEMPLATE
        ),
        verbose=True,
        allow_dangerous_requests=True,
    )
    text2cypher_tool = Tool(
        name="Text2Cypher_Expert",
        func=cypher_chain.invoke,
        description="Translate complex natural language questions into Cypher when other tools are insufficient.",
    )

    vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=config.neo4j_uri,
        username=config.neo4j_user,
        password=config.neo4j_password,
        index_name=VECTOR_INDEX_NAME,
        node_label="Text",
        text_node_properties=["title", "description", "text"],
        embedding_node_property="text_embedding",
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": config.agent_retrieval_k})
    semantic_tool = create_retriever_tool(
        retriever,
        name="Semantic_Search",
        description="Use this for topic/meaning based retrieval over embedded text.",
    )

    tools = LegislationTools(analysis, embeddings, k=config.agent_retrieval_k).build_tools()
    tools.extend([text2cypher_tool, semantic_tool])
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def _get_analysis(config: Config):
    """Import Neo4jAnalysis from the repo root and connect."""
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from neo4j_analysis import Neo4jAnalysis

    return Neo4jAnalysis(
        config.neo4j_uri, config.neo4j_user, config.neo4j_password, config.neo4j_database
    )


def main(config: Optional[Config] = None, embed_model: str = DEFAULT_EMBED_MODEL) -> None:
    """Embed content nodes, create indexes, and smoke-test the agent."""
    config = (config or load_config()).require_neo4j()
    check_embedding_consistency(embed_model, embed_model)

    analysis = _get_analysis(config)
    try:
        model = load_embedding_model(embed_model)
        run_embeddings(analysis, model)
        label_text_nodes(analysis)
        create_vector_index(analysis)
        create_text_index(analysis)

        agent = build_agent(analysis, config, embed_model=embed_model)
        question = "Find legislation that discusses the nuances of Corporate Tax."
        result = agent.invoke({"messages": [("user", question)]})
        print(result)
    finally:
        analysis.close()


if __name__ == "__main__":
    main()
