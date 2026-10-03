"""Tests for src.crawler (the CLML parsing core + async crawl loop)."""

import asyncio
import json

import pytest

from src.crawler import (
    Crawl4aiFetcher,
    LegislationCrawler,
    RequestsFetcher,
    XmlSchemaAccumulator,
)


# --------------------------------------------------------------- URL helpers
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://www.legislation.gov.uk/id/ukpga/2010/4", "http://www.legislation.gov.uk/ukpga/2010/4/data.xml"),
        ("http://www.legislation.gov.uk/ukpga/2010/4/", "http://www.legislation.gov.uk/ukpga/2010/4/data.xml"),
        ("http://www.legislation.gov.uk/ukpga/2010/4/data.xml", "http://www.legislation.gov.uk/ukpga/2010/4/data.xml"),
        (None, None),
        ("", None),
    ],
)
def test_normalize_url(crawler, raw, expected):
    assert crawler.normalize_url(raw) == expected


def test_clean_url_strips_id_and_trailing_slash(crawler):
    assert crawler.clean_url("http://x/id/ukpga/1988/1/") == "http://x/ukpga/1988/1"


def test_get_safe_filename_replaces_slashes(crawler):
    name = crawler.get_safe_filename(
        "http://www.legislation.gov.uk/ukpga/2010/4/data.xml"
    )
    assert "/" not in name
    assert name.endswith("data.xml")


# ---------------------------------------------------------------- metadata
def test_extract_identifier(crawler, sample_soup):
    ident = crawler.extract_identifier(sample_soup)
    assert ident["title"] == "Corporation Tax Act 2010"
    assert ident["uri"] == "http://www.legislation.gov.uk/ukpga/2010/4"
    assert ident["valid_date"] == "2020-01-01"
    assert ident["publisher"] == "Statute Law Database"


def test_extract_metadata_core_fields(crawler, sample_soup):
    meta = crawler.extract_metadata(sample_soup)
    assert meta["year"] == "2010"
    assert meta["number"] == "4"
    assert meta["enactment_date"] == "2010-03-03"
    assert meta["status"] == "final"
    assert meta["category"] == "primary"
    assert meta["coming_into_force"] == "2010-04-01"


def test_extract_metadata_unapplied_effects(crawler, sample_soup):
    meta = crawler.extract_metadata(sample_soup)
    effects = meta["unapplied_effects"]
    assert len(effects) == 1
    eff = effects[0]
    assert eff["effect_id"] == "E1"
    assert eff["requires_applied"] is True
    assert eff["affecting_title"] == "Finance Act 2023"
    assert eff["in_force_date"] == "2023-04-01"
    assert eff["in_force_qualification"] == "wholly"


def test_extract_metadata_empty_when_no_block(crawler):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup("<Legislation></Legislation>", "xml")
    assert crawler.extract_metadata(soup) == {}


def test_extract_super(crawler, sample_soup):
    super_info = crawler.extract_super(sample_soup)
    assert super_info["supersedes"] == "http://www.legislation.gov.uk/ukpga/1988/1"
    assert super_info["superseded_by"] == "http://www.legislation.gov.uk/ukpga/2020/14"


# ------------------------------------------------------------ commentaries
def test_extract_commentaries_collects_citations_and_subrefs(crawler, sample_soup):
    found = set()
    cmap = crawler._extract_commentaries(sample_soup, current_depth=0, found_citations=found)
    assert "c1" in cmap
    c1 = cmap["c1"]
    assert c1["type"] == "C"
    assert len(c1["citations"]) == 1
    assert c1["citations"][0]["uri"] == "http://www.legislation.gov.uk/id/ukpga/1988/1"
    assert len(c1["citation_subrefs"]) == 1
    assert c1["citation_subrefs"][0]["citation_ref"] == "cit1"
    # depth 0 < max_depth 2 -> citation queued for crawling
    assert "http://www.legislation.gov.uk/id/ukpga/1988/1" in found


def test_commentaries_not_followed_past_max_depth(crawler, sample_soup):
    found = set()
    crawler._extract_commentaries(sample_soup, current_depth=2, found_citations=found)
    assert found == set()


# ---------------------------------------------------------------- body tree
def test_extract_body_parts_hierarchy(crawler, sample_soup):
    cmap = crawler._extract_commentaries(sample_soup, 0, set())
    parts = crawler._extract_body_parts(sample_soup, cmap)
    assert len(parts) == 1
    part = parts[0]
    assert part["order"] == 1
    assert part["part_number"] == "Part 1"
    assert part["title"] == "Calculation of liability"
    assert part["restrict_start_date"] == "2010-04-01"

    chapter = part["chapters"][0]
    assert chapter["chapter_number"] == "Chapter 1"
    assert chapter["title"] == "Currency"

    section = chapter["sections"][0]
    assert section["order"] == 1
    assert section["section_number"] == "1"
    assert section["title"] == "Period for which tax is charged"

    para = section["paragraphs"][0]
    assert para["order"] == 1  # inner loop counter fix (was shadowed `idx`)
    assert para["paragraph_number"] == "2"
    assert "financial years" in para["text"]
    # commentary resolves at the paragraph (P2) level
    assert [c["ref_id"] for c in para["commentaries"]] == ["c1"]


# ----------------------------------------------------------------- schedules
def test_extract_schedules(crawler, sample_soup):
    cmap = crawler._extract_commentaries(sample_soup, 0, set())
    scheds = crawler._extract_schedules(sample_soup, cmap)
    assert len(scheds) == 1
    sched = scheds[0]
    assert sched["schedule_number"] == "Schedule 1"
    assert sched["reference"] == "Section 1"
    p1 = sched["paragraphs"][0]
    assert p1["paragraph_number"] == "1"
    assert p1["crossheading"] == "Amendments"
    sub = p1["subparagraphs"][0]
    assert sub["subparagraph_number"] == "2"
    assert "consequential amendment" in sub["text"]


# ----------------------------------------------------------- explanatory notes
def test_extract_explanatory_notes(crawler, sample_soup):
    found = set()
    notes = crawler._extract_explanatory_notes(sample_soup, 0, found)
    assert notes["uri"] == "http://www.legislation.gov.uk/ukpga/2010/4/notes"
    assert len(notes["paragraphs"]) == 1
    assert "explain" in notes["paragraphs"][0]["text"]
    assert len(notes["paragraphs"][0]["citations"]) == 1
    assert "http://www.legislation.gov.uk/id/ukpga/2007/3" in found


def test_explanatory_notes_none_when_absent(crawler):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup("<Legislation></Legislation>", "xml")
    assert crawler._extract_explanatory_notes(soup, 0, set()) is None


# --------------------------------------------------------- parse_document
def test_parse_document_builds_doc_and_discovers_links(crawler, sample_soup):
    doc, discovered = crawler._parse_document(
        sample_soup, 0, "http://www.legislation.gov.uk/ukpga/2010/4"
    )
    assert doc["legislation_url"] == "http://www.legislation.gov.uk/ukpga/2010/4"
    assert doc["identifier"]["title"] == "Corporation Tax Act 2010"
    assert doc["parts"] and doc["schedules"] and doc["explanatory_notes"]

    links = dict(discovered)  # url -> depth
    # supersedes / superseded_by are always discovered at the current depth
    assert links["http://www.legislation.gov.uk/ukpga/1988/1"] == 0
    assert links["http://www.legislation.gov.uk/ukpga/2020/14"] == 0
    # citations are discovered at depth + 1 (and gated by max_depth inside extractors)
    assert links["http://www.legislation.gov.uk/id/ukpga/2007/3"] == 1


def test_parse_document_no_citations_at_max_depth(crawler, sample_soup):
    # at the depth limit the extractors collect no citations, so only the
    # always-followed supersedes links are discovered
    _doc, discovered = crawler._parse_document(
        sample_soup, crawler.max_depth, "http://www.legislation.gov.uk/ukpga/2010/4"
    )
    links = dict(discovered)
    assert "http://www.legislation.gov.uk/id/ukpga/2007/3" not in links
    assert links["http://www.legislation.gov.uk/ukpga/1988/1"] == crawler.max_depth


# ------------------------------------------------------------- fetchers
class _FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


class _FakeSession:
    """Synchronous fake for the requests fallback path."""

    def __init__(self, content):
        self._content = content
        self.requested = []

    def get(self, url):
        self.requested.append(url)
        return _FakeResponse(self._content)


class _FakeFetcher:
    """Async fake fetcher: returns scripted bytes per URL, records requests."""

    def __init__(self, content_by_url=None, default=None):
        self.content_by_url = content_by_url or {}
        self.default = default
        self.requested = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch_many(self, urls):
        out = {}
        for url in urls:
            self.requested.append(url)
            out[url] = self.content_by_url.get(url, self.default)
        return out


def test_requests_fetcher_wraps_session_concurrently(sample_xml_bytes):
    session = _FakeSession(sample_xml_bytes)
    fetcher = RequestsFetcher(session)
    urls = ["http://x/a/data.xml", "http://x/b/data.xml"]
    result = asyncio.run(fetcher.fetch_many(urls))
    assert result["http://x/a/data.xml"] == sample_xml_bytes
    assert result["http://x/b/data.xml"] == sample_xml_bytes
    assert set(session.requested) == set(urls)


def test_fetcher_selection(tmp_path):
    sem = asyncio.Semaphore(1)

    # an injected fetcher wins outright
    fake = _FakeFetcher()
    c_fetcher = LegislationCrawler(
        url_prefix="http://x",
        json_output_dir=str(tmp_path / "j1"),
        cache_dir=str(tmp_path / "c1"),
        fetcher=fake,
    )
    assert c_fetcher._make_fetcher(sem) is fake

    # an injected session forces the requests fallback
    c_session = LegislationCrawler(
        url_prefix="http://x",
        json_output_dir=str(tmp_path / "j2"),
        cache_dir=str(tmp_path / "c2"),
        session=_FakeSession(b"<x/>"),
    )
    assert isinstance(c_session._make_fetcher(sem), RequestsFetcher)

    # default: crawl4ai when importable, else the requests fallback — either way
    # a working async fetcher
    c_default = LegislationCrawler(
        url_prefix="http://x",
        json_output_dir=str(tmp_path / "j3"),
        cache_dir=str(tmp_path / "c3"),
    )
    default_fetcher = c_default._make_fetcher(sem)
    assert isinstance(default_fetcher, (RequestsFetcher, Crawl4aiFetcher))


# ------------------------------------------------------------- crawl loop
def _make_crawler(tmp_path, fetcher, **kw):
    return LegislationCrawler(
        url_prefix="http://www.legislation.gov.uk",
        json_output_dir=str(tmp_path / "json_out"),
        max_depth=2,
        cache_dir=str(tmp_path / ".cache"),
        fetcher=fetcher,
        **kw,
    )


def test_crawl_writes_json_and_expands_frontier(tmp_path, sample_xml_bytes):
    start = "http://www.legislation.gov.uk/ukpga/2010/4"
    xml_url = "http://www.legislation.gov.uk/ukpga/2010/4/data.xml"
    fetcher = _FakeFetcher({xml_url: sample_xml_bytes})
    crawler = _make_crawler(tmp_path, fetcher)

    asyncio.run(crawler.crawl(start))

    # exactly one document written (only the start URL has fake content), under
    # its year directory
    written = list((tmp_path / "json_out" / "2010").glob("*.json"))
    assert len(written) == 1
    doc = json.loads(written[0].read_text(encoding="utf-8"))
    assert doc["legislation_url"] == start
    assert doc["identifier"]["title"] == "Corporation Tax Act 2010"
    assert doc["parts"] and doc["schedules"] and doc["explanatory_notes"]

    # discovered supersedes / superseded_by links entered a later wave (attempted)
    assert "http://www.legislation.gov.uk/ukpga/1988/1" in crawler.visited_urls
    assert "http://www.legislation.gov.uk/ukpga/2020/14" in crawler.visited_urls
    # the start doc was fetched exactly once
    assert fetcher.requested.count(xml_url) == 1


def test_crawl_caches_and_does_not_refetch(tmp_path, sample_xml_bytes):
    start = "http://www.legislation.gov.uk/ukpga/2010/4"
    xml_url = "http://www.legislation.gov.uk/ukpga/2010/4/data.xml"
    fetcher = _FakeFetcher({xml_url: sample_xml_bytes})
    crawler = _make_crawler(tmp_path, fetcher)

    asyncio.run(crawler.crawl(start))
    # a prettified cache file now exists for the start doc
    cache_files = list((tmp_path / ".cache").glob("*"))
    assert cache_files

    # a fresh crawler (new visited set) sharing the cache dir serves the start
    # doc from the file cache instead of refetching it (the uncached supersedes
    # links are still attempted, which is expected)
    fetcher2 = _FakeFetcher({xml_url: sample_xml_bytes})
    crawler2 = _make_crawler(tmp_path, fetcher2)
    asyncio.run(crawler2.crawl(start))
    assert xml_url not in fetcher2.requested


def test_crawl_skips_already_visited(tmp_path, sample_xml_bytes):
    start = "http://www.legislation.gov.uk/ukpga/2010/4"
    xml_url = "http://www.legislation.gov.uk/ukpga/2010/4/data.xml"
    fetcher = _FakeFetcher({xml_url: sample_xml_bytes})
    crawler = _make_crawler(tmp_path, fetcher)
    crawler.visited_urls.add("http://www.legislation.gov.uk/ukpga/2010/4")

    asyncio.run(crawler.crawl(start))
    assert fetcher.requested == []  # nothing fetched
    assert list((tmp_path / "json_out").rglob("*.json")) == []


# a <ukm:Supersedes> with no URI attribute crashes extract_super (clean_url(None))
_MALFORMED_XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>'
    b'<Legislation xmlns:ukm="http://www.legislation.gov.uk/namespaces/metadata">'
    b"<ukm:Supersedes/></Legislation>"
)


def test_process_wave_skips_malformed_document(tmp_path, sample_xml_bytes):
    good_xml = "http://www.legislation.gov.uk/ukpga/2010/4/data.xml"
    bad_xml = "http://www.legislation.gov.uk/ukpga/2010/5/data.xml"
    fetcher = _FakeFetcher({good_xml: sample_xml_bytes, bad_xml: _MALFORMED_XML})
    crawler = _make_crawler(tmp_path, fetcher)
    frontier = [
        ("http://www.legislation.gov.uk/ukpga/2010/4", 0),
        ("http://www.legislation.gov.uk/ukpga/2010/5", 0),
    ]

    async def run():
        async with fetcher:
            return await crawler._process_wave(fetcher, frontier, None)

    asyncio.run(run())

    # the malformed doc is skipped (no crash); the good doc is still written
    written = list((tmp_path / "json_out").rglob("*.json"))
    assert len(written) == 1
    assert written[0].parent.name == "2010"


# -------------------------------------------------------------- schema accumulator
def test_schema_accumulator_records_tags(sample_soup):
    acc = XmlSchemaAccumulator()
    tree = acc.add_soup(sample_soup)
    assert "Legislation" in tree
    # attributes are accumulated as a set
    assert isinstance(tree["Legislation"]["attributes"], set)
