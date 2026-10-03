"""Recursive crawler for legislation.gov.uk CLML XML.

Python port of ``crawler.ipynb``. Fetches CLML XML for a seed list of Acts,
parses the structural hierarchy (parts / chapters / sections / paragraphs /
schedules / explanatory notes) plus citations and commentaries, and writes one
JSON document per Act into ``<json_output_dir>/<year>/``.

Fetching goes through `crawl4ai <https://github.com/unclecode/crawl4ai>`_ by
default, using its **HTTP-only** ``AsyncHTTPCrawlerStrategy`` (no browser). The
browser strategy is deliberately *not* used: a headless browser renders XML
through its XML viewer and returns DOM-wrapped HTML, which would corrupt the raw
CLML our parser depends on. A ``requests``-based fetcher is kept as a fallback
(used automatically when crawl4ai is unavailable, or when a session is injected
for offline tests). This makes ``src/crawler.py`` intentionally diverge from
``crawler.ipynb`` (which fetched synchronously with ``requests``).

Changes vs. the notebook (behaviour-preserving unless noted):
  * module-level globals (``LEGISLATION_URL_PREFIX`` etc.) are now constructor
    arguments, so the crawler can be instantiated and unit-tested in isolation;
  * the ``super`` local (which shadowed the builtin) is renamed ``super_info``;
  * a shadowed ``idx`` loop variable in paragraph extraction is renamed
    ``para_idx`` (the notebook reused ``idx`` for both the section and the inner
    paragraph loop);
  * fetching is async + concurrent (one BFS frontier fetched at a time) and the
    network layer is a swappable ``fetcher``; the ``requests`` session remains
    injectable for offline testing via the fallback fetcher.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from bs4 import BeautifulSoup

from .config import Config, load_config

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "legal-legislation-explorer/1.0 (+https://github.com/unclecode/crawl4ai)"


class RequestsFetcher:
    """Fallback fetcher backed by a synchronous ``requests`` session.

    Used when crawl4ai is not installed, or when a session is injected for
    offline tests. Presents the same async ``fetch_many`` interface as
    :class:`Crawl4aiFetcher`; concurrency is achieved by running the blocking
    ``get`` calls in a thread pool, bounded by ``semaphore``.
    """

    def __init__(self, session=None, semaphore: Optional[asyncio.Semaphore] = None) -> None:
        self._session = session
        self._semaphore = semaphore

    async def __aenter__(self) -> "RequestsFetcher":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def fetch_many(self, urls: list) -> dict:
        async def one(url):
            if self._semaphore is not None:
                async with self._semaphore:
                    return url, await self._fetch(url)
            return url, await self._fetch(url)

        pairs = await asyncio.gather(*(one(u) for u in urls))
        return dict(pairs)

    async def _fetch(self, url: str) -> Optional[bytes]:
        def _blocking():
            import requests

            try:
                # The frontier is fetched concurrently across a thread pool. A
                # single ``requests.Session`` is not thread-safe, so the auto
                # path uses a per-call ``requests.get`` (fresh session each
                # time). An injected session is used as-is — the caller owns its
                # thread-safety (tests inject a trivial synchronous fake).
                getter = self._session.get if self._session is not None else requests.get
                response = getter(url)
                response.raise_for_status()
                return response.content
            except requests.exceptions.RequestException:
                return None

        return await asyncio.to_thread(_blocking)


class Crawl4aiFetcher:
    """Default fetcher: crawl4ai's HTTP-only strategy (no browser).

    Holds one open ``AsyncWebCrawler`` for the lifetime of a crawl and fetches a
    whole BFS frontier concurrently. We use ``AsyncHTTPCrawlerStrategy`` so the
    raw CLML XML is returned verbatim in ``result.html`` — the browser strategy
    would render XML through Chromium and corrupt it.

    Concurrency is a semaphore-bounded ``asyncio.gather`` of per-URL ``arun``
    calls rather than ``arun_many``: results key strictly by the input URL
    (legislation.gov.uk issues redirects, so a response URL may not match the
    request), and it avoids known rough edges in ``arun_many`` with the HTTP
    strategy. Because only the HTTP strategy is used, no browser binary
    (``crawl4ai-setup`` / chromium) is required.
    """

    def __init__(
        self,
        semaphore: Optional[asyncio.Semaphore] = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._semaphore = semaphore
        self._user_agent = user_agent
        self._crawler = None
        self._run_config = None

    async def __aenter__(self) -> "Crawl4aiFetcher":
        from crawl4ai import (
            AsyncWebCrawler,
            CacheMode,
            CrawlerRunConfig,
            HTTPCrawlerConfig,
        )

        # ``AsyncHTTPCrawlerStrategy`` is NOT re-exported from the ``crawl4ai``
        # top-level package in the 0.9 line (``from crawl4ai import
        # AsyncHTTPCrawlerStrategy`` raises ImportError) — it must be imported
        # from the ``async_crawler_strategy`` submodule. Verified against
        # crawl4ai 0.9.4; this matches the pinned ``crawl4ai>=0.9,<1`` line.
        from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

        strategy = AsyncHTTPCrawlerStrategy(
            browser_config=HTTPCrawlerConfig(
                method="GET",
                headers={"User-Agent": self._user_agent},
                follow_redirects=True,
                verify_ssl=True,
            )
        )
        # BYPASS crawl4ai's own cache: this crawler keeps its own file cache.
        self._run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        self._crawler = AsyncWebCrawler(crawler_strategy=strategy)
        await self._crawler.__aenter__()
        return self

    async def __aexit__(self, *exc) -> bool:
        if self._crawler is not None:
            await self._crawler.__aexit__(*exc)
            self._crawler = None
        return False

    async def fetch_many(self, urls: list) -> dict:
        async def one(url):
            if self._semaphore is not None:
                async with self._semaphore:
                    res = await self._crawler.arun(url, config=self._run_config)
            else:
                res = await self._crawler.arun(url, config=self._run_config)
            return url, self._to_bytes(res)

        pairs = await asyncio.gather(*(one(u) for u in urls))
        return dict(pairs)

    @staticmethod
    def _to_bytes(res) -> Optional[bytes]:
        if res is None or not getattr(res, "success", False):
            return None
        html = getattr(res, "html", None)
        if not html:
            return None
        return html.encode("utf-8") if isinstance(html, str) else html


class XmlSchemaAccumulator:
    """Accumulates the union of tags/attributes seen across crawled XML docs."""

    def __init__(self) -> None:
        self.schema_tree: dict = {}

    def add_soup(self, soup: BeautifulSoup) -> dict:
        for root_tag in soup.find_all(recursive=False):
            self._traverse(root_tag, self.schema_tree)
        return self.schema_tree

    def _traverse(self, tag, current_level: dict) -> None:
        if tag.name is None:
            return

        if tag.name not in current_level:
            current_level[tag.name] = {"attributes": set(), "children": {}}

        current_level[tag.name]["attributes"].update(tag.attrs.keys())

        for child in tag.find_all(recursive=False):
            self._traverse(child, current_level[tag.name]["children"])

    def print_current_schema(self, node: Optional[dict] = None, depth: int = 0) -> None:
        if node is None:
            node = self.schema_tree
            print("\n--- Current Accumulated XML Schema ---")

        for tag_name, tag_data in node.items():
            indent = "  " * depth
            attrs = ", ".join(sorted(tag_data["attributes"]))
            attr_str = f" (Attributes: {attrs})" if attrs else ""
            print(f"{indent}└── <{tag_name}>{attr_str}")
            self.print_current_schema(tag_data["children"], depth + 1)


class LegislationCrawler:
    """Breadth-first crawler that turns CLML XML into structured JSON."""

    def __init__(
        self,
        url_prefix: str,
        json_output_dir: str = "json_out",
        max_depth: int = 2,
        cache_dir: str = ".cache",
        session=None,
        fetcher=None,
        max_concurrency: int = 5,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.url_prefix = url_prefix
        self.json_output_dir = json_output_dir
        self.max_depth = max_depth
        self.cache_dir = cache_dir
        self.max_concurrency = max_concurrency
        self.user_agent = user_agent
        self._session = session
        self._fetcher = fetcher
        self.visited_urls: set = set()
        self.schema_accumulator = XmlSchemaAccumulator()
        self.last_fetch_status = ""

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.json_output_dir, exist_ok=True)

    def _make_fetcher(self, semaphore: asyncio.Semaphore):
        """Pick the network layer: injected > session > crawl4ai > requests.

        An explicitly injected ``fetcher`` wins (used by tests). Otherwise an
        injected ``session`` forces the requests fallback (keeps the offline
        ``_FakeSession`` path working). Otherwise crawl4ai is used when it
        imports, falling back to requests with a warning if it does not.
        """
        if self._fetcher is not None:
            return self._fetcher
        if self._session is not None:
            return RequestsFetcher(self._session, semaphore)
        try:
            import crawl4ai  # noqa: F401
        except ImportError:
            logger.warning(
                "crawl4ai is not installed; falling back to the requests fetcher. "
                "Install crawl4ai for the default concurrent HTTP crawler."
            )
            return RequestsFetcher(None, semaphore)
        return Crawl4aiFetcher(semaphore=semaphore, user_agent=self.user_agent)

    # ------------------------------------------------------------------ URLs
    def get_safe_filename(self, url: str) -> str:
        clean_url = url.split("://")[-1].replace(self.url_prefix + "/", "")
        return clean_url.replace("/", "_")

    def normalize_url(self, uri: Optional[str]) -> Optional[str]:
        if not uri:
            return None
        clean_uri = uri.replace("/id/", "/").rstrip("/")
        if not clean_uri.endswith("data.xml"):
            return f"{clean_uri}/data.xml"
        return clean_uri

    def clean_url(self, url: str) -> str:
        return url.replace("/id/", "/").rstrip("/")

    # ------------------------------------------------------------- metadata
    def extract_identifier(self, soup: BeautifulSoup) -> dict:
        identifier = {}
        title = soup.find("dc:title")
        identifier["title"] = title.text.strip() if title else None
        description = soup.find("dc:description")
        identifier["description"] = description.text.strip() if description else None
        publisher = soup.find("dc:publisher")
        identifier["publisher"] = publisher.text.strip() if publisher else None
        modified = soup.find("dc:modified")
        identifier["modified"] = modified.text.strip() if modified else None
        identifier_tag = soup.find("dc:identifier")
        identifier["uri"] = identifier_tag.text.strip() if identifier_tag else None
        valid = soup.find("dct:valid")
        identifier["valid_date"] = valid.text.strip() if valid else None
        return identifier

    def extract_super(self, soup: BeautifulSoup) -> dict:
        super_info = {}
        supersedes = soup.find("ukm:Supersedes")
        if supersedes:
            super_info["supersedes"] = self.clean_url(supersedes.get("URI"))
        superseded_by = soup.find("ukm:SupersededBy")
        if superseded_by:
            super_info["superseded_by"] = self.clean_url(superseded_by.get("URI"))
        return super_info

    def extract_metadata(self, soup: BeautifulSoup) -> dict:
        metadata: dict = {}
        metadata_block = soup.find(
            ["ukm:PrimaryMetadata", "ukm:SecondaryMetadata", "ukm:EUMetadata"]
        )
        if not metadata_block:
            return metadata

        year = metadata_block.find("ukm:Year")
        metadata["year"] = year.get("Value") if year else None
        number = metadata_block.find("ukm:Number")
        metadata["number"] = number.get("Value") if number else None
        enactment = metadata_block.find("ukm:EnactmentDate")
        metadata["enactment_date"] = enactment.get("Date") if enactment else None
        status = metadata_block.find("ukm:DocumentStatus")
        metadata["status"] = status.get("Value") if status else None
        isbn = metadata_block.find("ukm:ISBN")
        metadata["isbn"] = isbn.get("Value") if isbn else None
        category = metadata_block.find("ukm:DocumentCategory")
        metadata["category"] = category.get("Value") if category else None
        coming_into_force = metadata_block.find("ukm:ComingIntoForce")
        if coming_into_force:
            date_tag = coming_into_force.find("ukm:DateTime")
            metadata["coming_into_force"] = (
                date_tag.get("Date").strip() if date_tag else None
            )

        unapplied_effects_list = []
        for effect in metadata_block.find_all("ukm:UnappliedEffect"):
            effect_data = {
                "effect_id": effect.get("EffectId"),
                "type": effect.get("Type"),
                "affected_provisions": effect.get("AffectedProvisions"),
                "affecting_provisions": effect.get("AffectingProvisions"),
                "requires_applied": effect.get("RequiresApplied") == "true",
                "notes": effect.get("Notes"),
                "modified_date": effect.get("Modified"),
                "affecting_title": None,
                "in_force_date": None,
                "in_force_qualification": None,
            }
            affecting_title = effect.find("ukm:AffectingTitle")
            if affecting_title:
                effect_data["affecting_title"] = affecting_title.text.strip()
            in_force = effect.find("ukm:InForce")
            if in_force:
                if in_force.get("Date"):
                    effect_data["in_force_date"] = in_force.get("Date")
                elif in_force.get("Prospective") == "true":
                    effect_data["in_force_date"] = "Prospective"
                effect_data["in_force_qualification"] = in_force.get(
                    "Qualification"
                ) or in_force.get("OtherQualification")
            unapplied_effects_list.append(effect_data)

        metadata["unapplied_effects"] = unapplied_effects_list
        return metadata

    # ------------------------------------------------------------- caching
    def _cache_path(self, safe_name: str) -> str:
        return os.path.join(self.cache_dir, safe_name)

    def _read_cache(self, safe_name: str) -> Optional[bytes]:
        path = self._cache_path(safe_name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None

    def _write_cache(self, safe_name: str, raw_bytes: bytes) -> bytes:
        """Prettify and persist fetched XML; return the prettified bytes.

        Prettifying keeps cache files stable/diffable (matches the notebook's
        behaviour). The prettified bytes are what the rest of the pipeline then
        parses, so cached and freshly-fetched docs parse identically.
        """
        pretty = BeautifulSoup(raw_bytes, "xml").prettify(encoding="utf-8")
        with open(self._cache_path(safe_name), "wb") as f:
            f.write(pretty)
        return pretty

    # ---------------------------------------------------- content extraction
    def _extract_commentaries(
        self, soup: BeautifulSoup, current_depth: int, found_citations: set
    ) -> dict:
        commentaries_map: dict = {}
        for comm in soup.find_all("Commentary"):
            comm_id = comm.get("id")
            if not comm_id:
                continue

            full_text = comm.get_text(separator=" ", strip=True)
            citations = []
            for cit in comm.find_all("Citation"):
                cit_uri = cit.get("URI")
                citations.append(
                    {
                        "id": cit.get("id"),
                        "uri": cit_uri,
                        "title": cit.get("Title"),
                        "class": cit.get("Class"),
                        "year": cit.get("Year"),
                        "number": cit.get("Number"),
                        "text": cit.text.strip(),
                    }
                )
                if cit_uri and current_depth < self.max_depth:
                    found_citations.add(cit_uri)

            citation_subrefs = []
            for subref in comm.find_all("CitationSubRef"):
                citation_subrefs.append(
                    {
                        "id": subref.get("id"),
                        "uri": subref.get("URI"),
                        "citation_ref": subref.get("CitationRef"),
                        "section_ref": subref.get("SectionRef"),
                        "text": subref.text.strip(),
                    }
                )

            commentaries_map[comm_id] = {
                "type": comm.get("Type"),
                "text": full_text,
                "citations": citations,
                "citation_subrefs": citation_subrefs,
            }
        return commentaries_map

    def _resolve_commentaries(
        self, element, commentaries_map: dict, exclude_parent: Optional[str] = None
    ) -> list:
        resolved = []
        for cref in element.find_all("CommentaryRef"):
            if exclude_parent and cref.find_parent(exclude_parent) is not None:
                continue
            ref_id = cref.get("Ref")
            if ref_id in commentaries_map:
                resolved.append({"ref_id": ref_id, **commentaries_map[ref_id]})
        return resolved

    def _get_joined_text(self, element) -> str:
        return " ".join(
            t.get_text(separator=" ", strip=True) for t in element.find_all("Text")
        )

    def _extract_body_parts(self, soup: BeautifulSoup, commentaries_map: dict) -> list:
        document_tree: dict = {}
        body_p1s = [p1 for p1 in soup.find_all("P1") if not p1.find_parent("Schedules")]

        for idx, section in enumerate(body_p1s, start=1):
            part = section.find_parent("Part")
            chapter = section.find_parent("Chapter")
            p1group = section.find_parent("P1group")

            part_num = (
                part.find("Number").text.strip()
                if part and part.find("Number")
                else "No Part"
            )
            part_uri = part.get("DocumentURI") if part else None
            part_restrict_start_date = part.get("RestrictStartDate") if part else None
            part_restrict_end_date = part.get("RestrictEndDate") if part else None
            part_status = part.get("Status") if part else None
            part_title = (
                part.find("Title").text.strip() if part and part.find("Title") else None
            )
            chapter_uri = chapter.get("DocumentURI") if chapter else None
            chapter_restrict_start_date = (
                chapter.get("RestrictStartDate") if chapter else None
            )
            chapter_restrict_end_date = (
                chapter.get("RestrictEndDate") if chapter else None
            )
            chapter_status = chapter.get("Status") if chapter else None
            chapter_num = (
                chapter.find("Number").text.strip()
                if chapter and chapter.find("Number")
                else "No Chapter"
            )
            chapter_title = (
                chapter.find("Title").text.strip()
                if chapter and chapter.find("Title")
                else None
            )

            chap_dict_key = chapter_uri or chapter_num

            if part_num not in document_tree:
                document_tree[part_num] = {
                    "title": part_title,
                    "part_uri": part_uri,
                    "restrict_start_date": part_restrict_start_date,
                    "restrict_end_date": part_restrict_end_date,
                    "status": part_status,
                    "chapters": {},
                }

            if chap_dict_key not in document_tree[part_num]["chapters"]:
                document_tree[part_num]["chapters"][chap_dict_key] = {
                    "chapter_uri": chapter_uri,
                    "chapter_number": (
                        chapter_num if chapter_num != "No Chapter" else None
                    ),
                    "restrict_start_date": chapter_restrict_start_date,
                    "restrict_end_date": chapter_restrict_end_date,
                    "status": chapter_status,
                    "title": chapter_title,
                    "sections": [],
                }

            section_num = (
                section.find("Pnumber").text.strip()
                if section.find("Pnumber")
                else None
            )
            section_title = (
                p1group.find("Title").text.strip()
                if p1group and p1group.find("Title")
                else None
            )
            section_restrict_start_date = (
                p1group.get("RestrictStartDate") if p1group else None
            )
            section_restrict_end_date = (
                p1group.get("RestrictEndDate") if p1group else None
            )
            section_restrict_extent = p1group.get("RestrictExtent") if p1group else None
            section_status = p1group.get("Status") if p1group else None

            section_data = {
                "order": idx,
                "section_number": section_num,
                "title": section_title,
                "uri": section.get("DocumentURI") or section.get("id"),
                "restrict_start_date": section_restrict_start_date,
                "restrict_end_date": section_restrict_end_date,
                "restrict_extent": section_restrict_extent,
                "status": section_status,
                "commentaries": self._resolve_commentaries(
                    section, commentaries_map, exclude_parent="P2"
                ),
                "paragraphs": [],
            }

            paragraphs = section.find_all(["P2", "P3", "P4"])
            if not paragraphs:
                section_data["text"] = self._get_joined_text(section)
            else:
                # NOTE: the notebook reused ``idx`` here, shadowing the section
                # loop counter; renamed to ``para_idx`` for clarity.
                for para_idx, para in enumerate(paragraphs, start=1):
                    para_num = (
                        para.find("Pnumber").text.strip()
                        if para.find("Pnumber")
                        else None
                    )
                    section_data["paragraphs"].append(
                        {
                            "order": para_idx,
                            "paragraph_number": para_num,
                            "text": self._get_joined_text(para),
                            "uri": para.get("DocumentURI") or para.get("id"),
                            "restrict_start_date": para.get("RestrictStartDate"),
                            "restrict_end_date": para.get("RestrictEndDate"),
                            "restrict_extent": para.get("RestrictExtent"),
                            "status": para.get("Status"),
                            "commentaries": self._resolve_commentaries(
                                para, commentaries_map
                            ),
                        }
                    )

            document_tree[part_num]["chapters"][chap_dict_key]["sections"].append(
                section_data
            )

        parts_list = []
        for part_idx, (p_num, p_data) in enumerate(document_tree.items(), start=1):
            part_obj = {
                "order": part_idx,
                "part_number": p_num if p_num != "No Part" else None,
                "uri": p_data["part_uri"],
                "restrict_start_date": p_data["restrict_start_date"],
                "restrict_end_date": p_data["restrict_end_date"],
                "status": p_data["status"],
                "title": p_data["title"],
                "chapters": [],
            }
            for chap_idx, (_chap_key, c_data) in enumerate(
                p_data["chapters"].items(), start=1
            ):
                part_obj["chapters"].append(
                    {
                        "order": chap_idx,
                        "chapter_number": c_data["chapter_number"],
                        "uri": c_data["chapter_uri"],
                        "restrict_start_date": c_data["restrict_start_date"],
                        "restrict_end_date": c_data["restrict_end_date"],
                        "status": c_data["status"],
                        "title": c_data["title"],
                        "sections": c_data["sections"],
                    }
                )
            parts_list.append(part_obj)

        return parts_list

    def _extract_schedules(self, soup: BeautifulSoup, commentaries_map: dict) -> list:
        schedules_list: list = []
        schedules_root = soup.find("Schedules")
        if not schedules_root:
            return schedules_list

        for sched_idx, schedule in enumerate(
            schedules_root.find_all("Schedule"), start=1
        ):
            sched_num = (
                schedule.find("Number").text.strip() if schedule.find("Number") else None
            )
            sched_title_node = schedule.find("Title")
            sched_title = (
                sched_title_node.get_text(strip=True) if sched_title_node else None
            )
            sched_ref = (
                schedule.find("Reference").text.strip()
                if schedule.find("Reference")
                else None
            )

            sched_obj = {
                "order": sched_idx,
                "schedule_number": sched_num,
                "title": sched_title,
                "reference": sched_ref,
                "uri": schedule.get("DocumentURI") or schedule.get("id"),
                "paragraphs": [],
            }

            for p1_idx, p1 in enumerate(schedule.find_all("P1"), start=1):
                p1_num = p1.find("Pnumber").text.strip() if p1.find("Pnumber") else None
                pblock = p1.find_parent("Pblock")
                pblock_title = (
                    pblock.find("Title").get_text(strip=True)
                    if pblock and pblock.find("Title")
                    else None
                )

                p1_data = {
                    "order": p1_idx,
                    "paragraph_number": p1_num,
                    "crossheading": pblock_title,
                    "uri": p1.get("DocumentURI") or p1.get("id"),
                    "commentaries": self._resolve_commentaries(
                        p1, commentaries_map, exclude_parent="P2"
                    ),
                    "subparagraphs": [],
                }

                p2s = p1.find_all("P2")
                if not p2s:
                    p1_data["text"] = self._get_joined_text(p1)
                else:
                    for p2_idx, p2 in enumerate(p2s, start=1):
                        p2_num = (
                            p2.find("Pnumber").text.strip()
                            if p2.find("Pnumber")
                            else None
                        )
                        p1_data["subparagraphs"].append(
                            {
                                "order": p2_idx,
                                "subparagraph_number": p2_num,
                                "text": self._get_joined_text(p2),
                                "uri": p2.get("DocumentURI") or p2.get("id"),
                                "commentaries": self._resolve_commentaries(
                                    p2, commentaries_map
                                ),
                            }
                        )

                sched_obj["paragraphs"].append(p1_data)
            schedules_list.append(sched_obj)
        return schedules_list

    def _extract_explanatory_notes(
        self, soup: BeautifulSoup, current_depth: int, found_citations: set
    ) -> Optional[dict]:
        notes_root = soup.find("ExplanatoryNotes")
        if not notes_root:
            return None

        notes_uri = notes_root.get("DocumentURI") or notes_root.get("IdURI")

        paragraphs = []
        for p in notes_root.find_all("P"):
            p_text = self._get_joined_text(p)
            citations = []
            for cit in p.find_all("Citation"):
                cit_uri = cit.get("URI")
                citations.append(
                    {
                        "id": cit.get("id"),
                        "uri": cit_uri,
                        "title": cit.get("Title"),
                        "class": cit.get("Class"),
                        "year": cit.get("Year"),
                        "number": cit.get("Number"),
                        "text": cit.text.strip(),
                    }
                )
                if cit_uri and current_depth < self.max_depth:
                    found_citations.add(cit_uri)
            paragraphs.append({"text": p_text, "citations": citations})

        return {"uri": notes_uri, "paragraphs": paragraphs}

    # ---------------------------------------------------- parse & persist
    def _parse_document(
        self, soup: BeautifulSoup, current_depth: int, base_url: str
    ) -> tuple:
        """Turn a parsed document into (json_doc, discovered_links).

        Pure and I/O-free so it is trivially unit-testable. ``discovered_links``
        is a list of ``(url, depth)`` to enqueue: citations (already gated to
        ``current_depth < max_depth`` inside the extractors, hence empty at the
        depth limit) at ``current_depth + 1``, and supersedes / superseded-by
        links always at ``current_depth`` so chains stay complete.
        """
        identifier = self.extract_identifier(soup)
        metadata = self.extract_metadata(soup)
        super_info = self.extract_super(soup)

        found_citations: set = set()
        commentaries_map = self._extract_commentaries(
            soup, current_depth, found_citations
        )

        document = {
            "legislation_url": base_url,
            "identifier": identifier,
            "super": super_info,
            "metadata": metadata,
            "parts": self._extract_body_parts(soup, commentaries_map),
            "schedules": self._extract_schedules(soup, commentaries_map),
            "explanatory_notes": self._extract_explanatory_notes(
                soup, current_depth, found_citations
            ),
        }

        discovered: list = [(cit_url, current_depth + 1) for cit_url in found_citations]
        if super_info.get("supersedes"):
            discovered.append((super_info["supersedes"], current_depth))
        if super_info.get("superseded_by"):
            discovered.append((super_info["superseded_by"], current_depth))

        return document, discovered

    def _persist(self, document: dict, safe_name: str) -> str:
        """Write ``document`` under ``<json_output_dir>/<year>/`` and return the path."""
        doc_year = document.get("metadata", {}).get("year") or "unknown_year"
        year_dir = os.path.join(self.json_output_dir, str(doc_year))
        os.makedirs(year_dir, exist_ok=True)
        json_filepath = os.path.join(year_dir, safe_name.replace(".xml", ".json"))
        with open(json_filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(document, indent=4))
        return json_filepath

    # ----------------------------------------------------------- crawl loop
    def _make_pbar(self):
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            return None
        return tqdm(desc="Crawling Legislation", unit="docs")

    async def _process_wave(self, fetcher, frontier: list, pbar) -> list:
        """Process one BFS frontier concurrently; return the next frontier.

        The whole frontier of uncached URLs is fetched in one concurrent batch.
        Dedup happens here (not at enqueue time): ``visited_urls`` is marked as
        each URL enters the wave, so repeats within or across waves are dropped.
        Note depth values and wave index are decoupled — supersedes links keep
        their parent's depth; termination is guaranteed by ``visited_urls``.
        """
        # 1. normalize + dedup the frontier into concrete work items
        entries = []  # (xml_url, base, depth, safe_name)
        for target_url, depth in frontier:
            xml_url = self.normalize_url(target_url)
            if not xml_url:
                continue
            base = xml_url.replace("/data.xml", "")
            if base in self.visited_urls:
                continue
            self.visited_urls.add(base)
            entries.append((xml_url, base, depth, self.get_safe_filename(xml_url)))

        if not entries:
            return []

        # 2. split cached vs to-fetch. raw_by_xml maps xml_url -> (bytes, from_network)
        raw_by_xml: dict = {}
        to_fetch: list = []
        for xml_url, _base, _depth, safe_name in entries:
            cached = self._read_cache(safe_name)
            if cached is not None:
                raw_by_xml[xml_url] = (cached, False)
            else:
                to_fetch.append(xml_url)

        # 3. concurrent fetch of the uncached URLs
        if to_fetch:
            self.last_fetch_status = f"Fetched {len(to_fetch)} (cached {len(entries) - len(to_fetch)})"
            fetched = await fetcher.fetch_many(to_fetch)
            for xml_url in to_fetch:
                raw = fetched.get(xml_url)
                if raw is not None:
                    raw_by_xml[xml_url] = (raw, True)
        else:
            self.last_fetch_status = f"Cached {len(entries)}"

        # 4. parse + persist each doc; collect discovered links for the next wave.
        # A malformed document is logged and skipped rather than aborting the
        # whole crawl (one bad doc would otherwise kill the entire wave).
        next_frontier: list = []
        for xml_url, base, depth, safe_name in entries:
            got = raw_by_xml.get(xml_url)
            if got is None:
                continue  # fetch failed; skip
            try:
                raw, from_network = got
                if from_network:
                    raw = self._write_cache(safe_name, raw)
                soup = BeautifulSoup(raw, "xml")
                self.schema_accumulator.add_soup(soup)
                document, discovered = self._parse_document(soup, depth, base)
                self._persist(document, safe_name)
                next_frontier.extend(discovered)
            except Exception:  # noqa: BLE001 - one bad doc must not stop the crawl
                logger.exception("Failed to process %s; skipping.", base)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.set_postfix(
                {"frontier": len(next_frontier), "action": self.last_fetch_status}
            )
        return next_frontier

    async def crawl(self, start_url: str) -> None:
        """Breadth-first crawl from ``start_url``, one concurrent wave at a time."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        fetcher = self._make_fetcher(semaphore)
        pbar = self._make_pbar()
        try:
            async with fetcher:
                frontier = [(start_url, 0)]
                while frontier:
                    frontier = await self._process_wave(fetcher, frontier, pbar)
        finally:
            if pbar is not None:
                pbar.close()


async def _run(config: Config) -> None:
    crawler = LegislationCrawler(
        url_prefix=config.legislation_url_prefix,
        json_output_dir=config.json_output_dir,
        max_depth=config.depth_limit,
    )

    with open(config.legislation_uri_list_file, "r", encoding="utf-8") as f:
        legislation_uris = [line.strip() for line in f if line.strip()]

    for uri in legislation_uris:
        await crawler.crawl(f"{config.legislation_url_prefix}{uri}")

    crawler.schema_accumulator.print_current_schema()


def main(config: Optional[Config] = None) -> None:
    """Crawl every seed URI listed in ``LEGISLATION_URI_LIST_FILE``."""
    config = config or load_config()
    if not config.legislation_url_prefix or not config.legislation_uri_list_file:
        raise RuntimeError(
            "Set LEGISLATION_URL_PREFIX and LEGISLATION_URI_LIST_FILE to crawl."
        )
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
