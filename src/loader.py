"""Load crawled JSON into Neo4j with PySpark.

Python equivalent of ``loader.ipynb``. Reads the per-Act JSON produced by the
crawler and MERGEs the full node/relationship hierarchy into Neo4j using the
Neo4j Spark connector.

PySpark is imported lazily / defensively so this module (and its unit tests)
can be imported in an environment without Spark installed; the SQL functions
resolve to ``None`` and are only exercised when an actual load runs.
"""

from __future__ import annotations

from typing import List, Optional

from .config import Config, load_config

try:  # PySpark is heavy and optional at import time.
    from pyspark.sql.functions import (
        coalesce,
        col,
        concat,
        explode_outer,
        lit,
        md5,
        regexp_replace,
        to_date,
    )

    _HAS_PYSPARK = True
except ImportError:  # pragma: no cover - exercised only without pyspark
    coalesce = col = concat = explode_outer = lit = md5 = regexp_replace = to_date = None
    _HAS_PYSPARK = False


# Unique-constraint DDL applied before the Spark load so MERGE is fast and
# cannot create duplicates. Exposed as a constant for inspection/testing.
CONSTRAINTS: List[str] = [
    "CREATE CONSTRAINT leg_uri_unique IF NOT EXISTS FOR (l:Legislation) REQUIRE l.uri IS UNIQUE;",
    "CREATE CONSTRAINT part_id_unique IF NOT EXISTS FOR (p:Part) REQUIRE p.id IS UNIQUE;",
    "CREATE CONSTRAINT chap_id_unique IF NOT EXISTS FOR (c:Chapter) REQUIRE c.id IS UNIQUE;",
    "CREATE CONSTRAINT sec_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE;",
    "CREATE CONSTRAINT para_id_unique IF NOT EXISTS FOR (pa:Paragraph) REQUIRE pa.id IS UNIQUE;",
    "CREATE CONSTRAINT sched_id_unique IF NOT EXISTS FOR (s:Schedule) REQUIRE s.id IS UNIQUE;",
    "CREATE CONSTRAINT sched_para_id_unique IF NOT EXISTS FOR (p:ScheduleParagraph) REQUIRE p.id IS UNIQUE;",
    "CREATE CONSTRAINT sched_subpara_id_unique IF NOT EXISTS FOR (sp:ScheduleSubparagraph) REQUIRE sp.id IS UNIQUE;",
    "CREATE CONSTRAINT com_id_unique IF NOT EXISTS FOR (com:Commentary) REQUIRE com.id IS UNIQUE;",
    "CREATE CONSTRAINT cit_id_unique IF NOT EXISTS FOR (cit:Citation) REQUIRE cit.id IS UNIQUE;",
    "CREATE CONSTRAINT sub_id_unique IF NOT EXISTS FOR (sub:CitationSubRef) REQUIRE sub.id IS UNIQUE;",
    "CREATE CONSTRAINT en_id_unique IF NOT EXISTS FOR (en:ExplanatoryNotes) REQUIRE en.id IS UNIQUE;",
    "CREATE CONSTRAINT ep_id_unique IF NOT EXISTS FOR (ep:ExplanatoryNotesParagraph) REQUIRE ep.id IS UNIQUE;",
]


class LegislationGraphLoader:
    """Transforms the crawler's JSON into graph nodes/edges via Spark."""

    def __init__(self, uri, user, password, json_output_dir, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.json_output_dir = json_output_dir
        self.database = database

    def _write_to_neo4j(self, df, query):
        if df is not None:
            df.write.format("org.neo4j.spark.DataSource").mode("Append").option(
                "query", query
            ).save()

    def _safe_extract(self, struct_col, expected_fields, actual_fields, prefix=""):
        return [
            (
                col(f"{struct_col}.{f}").alias(f"{prefix}{f}")
                if f in actual_fields
                else lit(None).alias(f"{prefix}{f}")
            )
            for f in expected_fields
        ]

    def _safe_legis_extract(self, df, struct_col, expected_fields):
        if struct_col not in df.columns:
            return [
                lit(None).cast("date" if is_date else "string").alias(alias)
                for _f, alias, is_date in expected_fields
            ]

        actual_fields = df.schema[struct_col].dataType.fieldNames()
        exprs = []
        for f, alias, is_date in expected_fields:
            if f in actual_fields:
                c = col(f"{struct_col}.{f}")
                if is_date:
                    c = to_date(c, "yyyy-MM-dd")
                exprs.append(c.alias(alias))
            else:
                exprs.append(lit(None).cast("date" if is_date else "string").alias(alias))
        return exprs

    def _write_legislation_nodes(self, raw_df):
        print("Writing Legislation Nodes...")
        id_fields = [
            ("title", "title", False),
            ("description", "description", False),
            ("modified", "modified_date", True),
            ("valid_date", "valid_date", True),
        ]
        meta_fields = [
            ("enactment_date", "enactment_date", True),
            ("status", "status", False),
            ("category", "category", False),
            ("coming_into_force", "coming_into_force", True),
        ]

        select_exprs = [col("legislation_url").alias("uri")]
        select_exprs.extend(self._safe_legis_extract(raw_df, "identifier", id_fields))
        select_exprs.extend(self._safe_legis_extract(raw_df, "metadata", meta_fields))

        legis_df = raw_df.select(*select_exprs).dropDuplicates(["uri"])

        query = """
            UNWIND event AS row
            MERGE (l:Legislation {uri: row.uri})
            SET l.title = row.title,
                l.description = row.description,
                l.modified_date = row.modified_date,
                l.valid_date = row.valid_date,
                l.enactment_date = row.enactment_date,
                l.status = row.status,
                l.category = row.category,
                l.coming_into_force = row.coming_into_force
        """
        self._write_to_neo4j(legis_df, query)

    def _write_part_nodes(self, raw_df):
        print("Writing Part Nodes...")
        parts_df = (
            raw_df.select(
                col("legislation_url").alias("legis_uri"),
                explode_outer("parts").alias("part"),
            )
            .filter(col("part").isNotNull())
            .withColumn(
                "part_id",
                concat(
                    col("legis_uri"),
                    lit("#part_"),
                    coalesce(col("part.part_number"), md5(col("part").cast("string"))),
                ),
            )
        )

        query = """
            UNWIND event AS row
            MATCH (l:Legislation {uri: row.legis_uri})
            MERGE (p:Part {id: row.part_id})
            SET p.number = row.`part.part_number`,
                p.order = row.`part.order`,
                p.title = row.`part.title`,
                p.uri = row.`part.uri`,
                p.status = row.`part.status`,
                p.restrict_start_date = date(row.`part.restrict_start_date`),
                p.restrict_end_date = date(row.`part.restrict_end_date`)
            MERGE (l)-[:HAS_PART]->(p)
        """
        self._write_to_neo4j(parts_df, query)
        return parts_df

    def _write_chapter_nodes(self, parts_df):
        print("Writing Chapter Nodes...")
        chapters_df = (
            parts_df.select(
                col("legis_uri"),
                col("part_id"),
                explode_outer("part.chapters").alias("chapter"),
            )
            .filter(col("chapter").isNotNull())
            .withColumn(
                "chapter_id",
                coalesce(
                    col("chapter.uri"),
                    concat(
                        col("part_id"),
                        lit("#chapter_"),
                        coalesce(
                            col("chapter.chapter_number"),
                            md5(col("chapter").cast("string")),
                        ),
                    ),
                ),
            )
        )

        query = """
            UNWIND event AS row
            MATCH (p:Part {id: row.part_id})
            MERGE (c:Chapter {id: row.chapter_id})
            SET c.number = row.`chapter.chapter_number`,
                c.order = row.`chapter.order`,
                c.title = row.`chapter.title`,
                c.uri = row.`chapter.uri`,
                c.status = row.`chapter.status`,
                c.restrict_start_date = date(row.`chapter.restrict_start_date`),
                c.restrict_end_date = date(row.`chapter.restrict_end_date`)
            MERGE (p)-[:HAS_CHAPTER]->(c)
        """
        self._write_to_neo4j(chapters_df, query)
        return chapters_df

    def _write_section_nodes(self, chapters_df):
        print("Writing Section Nodes...")
        sections_df = (
            chapters_df.select(
                col("legis_uri"),
                col("chapter_id"),
                explode_outer("chapter.sections").alias("section"),
            )
            .filter(col("section").isNotNull())
            .withColumn(
                "sec_id",
                coalesce(
                    col("section.uri"),
                    concat(
                        col("chapter_id"),
                        lit("#sec_"),
                        coalesce(
                            col("section.section_number"),
                            md5(col("section").cast("string")),
                        ),
                    ),
                ),
            )
        )

        query = """
            UNWIND event AS row
            MATCH (c:Chapter {id: row.chapter_id})
            MERGE (s:Section {id: row.sec_id})
            SET s.number = row.`section.section_number`,
                s.order = row.`section.order`,
                s.title = row.`section.title`,
                s.uri = row.`section.uri`,
                s.restrict_extent = row.`section.restrict_extent`,
                s.restrict_start_date = date(row.`section.restrict_start_date`),
                s.restrict_end_date = date(row.`section.restrict_end_date`),
                s.status = row.`section.status`
            MERGE (c)-[:HAS_SECTION]->(s)
        """
        self._write_to_neo4j(sections_df, query)
        return sections_df

    def _write_paragraph_nodes(self, sections_df):
        print("Writing Paragraph Nodes...")
        paragraphs_df = (
            sections_df.select(
                col("legis_uri"),
                col("sec_id"),
                explode_outer("section.paragraphs").alias("paragraph"),
            )
            .filter(col("paragraph").isNotNull())
            .withColumn(
                "para_id",
                coalesce(
                    col("paragraph.uri"),
                    concat(
                        col("sec_id"),
                        lit("#para_"),
                        coalesce(
                            col("paragraph.paragraph_number"),
                            md5(col("paragraph").cast("string")),
                        ),
                    ),
                ),
            )
        )

        query = """
            UNWIND event AS row
            MATCH (s:Section {id: row.sec_id})
            MERGE (pa:Paragraph {id: row.para_id})
            SET pa.number = row.`paragraph.paragraph_number`,
                pa.order = row.`paragraph.order`,
                pa.text = row.`paragraph.text`,
                pa.uri = row.`paragraph.uri`,
                pa.restrict_extent = row.`paragraph.restrict_extent`,
                pa.restrict_start_date = date(row.`paragraph.restrict_start_date`),
                pa.restrict_end_date = date(row.`paragraph.restrict_end_date`),
                pa.status = row.`paragraph.status`
            MERGE (s)-[:HAS_PARAGRAPH]->(pa)
        """
        self._write_to_neo4j(paragraphs_df, query)
        return paragraphs_df

    def _write_schedules_nodes(self, raw_df):
        if (
            "schedules" not in raw_df.columns
            or raw_df.schema["schedules"].dataType.simpleString() == "array<string>"
        ):
            return None, None, None

        print("Writing Schedule Nodes...")
        schedules_df = (
            raw_df.select(
                col("legislation_url").alias("legis_uri"),
                explode_outer("schedules").alias("schedule"),
            )
            .filter(col("schedule").isNotNull())
            .withColumn(
                "sched_id",
                coalesce(
                    col("schedule.uri"),
                    concat(
                        col("legis_uri"),
                        lit("#sched_"),
                        coalesce(
                            col("schedule.schedule_number"),
                            md5(col("schedule").cast("string")),
                        ),
                    ),
                ),
            )
        )

        query_sched = """
            UNWIND event AS row
            MATCH (l:Legislation {uri: row.legis_uri})
            MERGE (sc:Schedule {id: row.sched_id})
            SET sc.number = row.`schedule.schedule_number`,
                sc.order = row.`schedule.order`,
                sc.title = row.`schedule.title`,
                sc.reference = row.`schedule.reference`,
                sc.uri = row.`schedule.uri`
            MERGE (l)-[:HAS_SCHEDULE]->(sc)
        """
        self._write_to_neo4j(schedules_df, query_sched)

        print("Writing Schedule Paragraph Nodes...")
        sched_paras_df = (
            schedules_df.select(
                col("legis_uri"),
                col("sched_id"),
                explode_outer("schedule.paragraphs").alias("paragraph"),
            )
            .filter(col("paragraph").isNotNull())
            .withColumn(
                "para_id",
                coalesce(
                    col("paragraph.uri"),
                    concat(
                        col("sched_id"),
                        lit("#spara_"),
                        coalesce(
                            col("paragraph.paragraph_number"),
                            md5(col("paragraph").cast("string")),
                        ),
                    ),
                ),
            )
        )

        query_para = """
            UNWIND event AS row
            MATCH (sc:Schedule {id: row.sched_id})
            MERGE (p:ScheduleParagraph {id: row.para_id})
            SET p.number = row.`paragraph.paragraph_number`,
                p.order = row.`paragraph.order`,
                p.crossheading = row.`paragraph.crossheading`,
                p.text = row.`paragraph.text`,
                p.uri = row.`paragraph.uri`
            MERGE (sc)-[:HAS_PARAGRAPH]->(p)
        """
        self._write_to_neo4j(sched_paras_df, query_para)

        sched_para_comm_df = sched_paras_df.select(
            col("legis_uri"),
            col("para_id").alias("parent_id"),
            explode_outer("paragraph.commentaries").alias("commentary"),
        ).filter(col("commentary").isNotNull())

        sched_subpara_comm_df = None
        if "subparagraphs" in sched_paras_df.schema["paragraph"].dataType.fieldNames():
            print("Writing Schedule Sub-paragraph Nodes...")
            sched_subparas_df = (
                sched_paras_df.select(
                    col("legis_uri"),
                    col("para_id"),
                    explode_outer("paragraph.subparagraphs").alias("subparagraph"),
                )
                .filter(col("subparagraph").isNotNull())
                .withColumn(
                    "subpara_id",
                    coalesce(
                        col("subparagraph.uri"),
                        concat(
                            col("para_id"),
                            lit("#ssub_"),
                            coalesce(
                                col("subparagraph.subparagraph_number"),
                                md5(col("subparagraph").cast("string")),
                            ),
                        ),
                    ),
                )
            )

            query_sub = """
                UNWIND event AS row
                MATCH (p:ScheduleParagraph {id: row.para_id})
                MERGE (sp:ScheduleSubparagraph {id: row.subpara_id})
                SET sp.number = row.`subparagraph.subparagraph_number`,
                    sp.order = row.`subparagraph.order`,
                    sp.text = row.`subparagraph.text`,
                    sp.uri = row.`subparagraph.uri`
                MERGE (p)-[:HAS_SUBPARAGRAPH]->(sp)
            """
            self._write_to_neo4j(sched_subparas_df, query_sub)

            sched_subpara_comm_df = sched_subparas_df.select(
                col("legis_uri"),
                col("subpara_id").alias("parent_id"),
                explode_outer("subparagraph.commentaries").alias("commentary"),
            ).filter(col("commentary").isNotNull())

        return sched_paras_df, sched_para_comm_df, sched_subpara_comm_df

    def _write_single_commentary(self, df, parent_label):
        if df is not None:
            actual_fields = df.schema["commentary"].dataType.fieldNames()
            safe_cols = [
                col("legis_uri"),
                col("parent_id"),
                col("commentary.ref_id").alias("ref_id"),
            ] + self._safe_extract("commentary", ["type", "text"], actual_fields)

            flat_df = (
                df.select(*safe_cols)
                .filter(col("ref_id").isNotNull())
                .dropDuplicates(["parent_id", "ref_id"])
                .repartition(10, "parent_id")
            )

            query = f"""
                UNWIND event AS row
                WITH row WHERE row.ref_id IS NOT NULL AND row.legis_uri IS NOT NULL
                MATCH (parent:{parent_label} {{id: row.parent_id}})
                MERGE (com:Commentary {{id: row.legis_uri + "#" + row.ref_id}})
                SET com.type = row.type,
                    com.text = row.text
                MERGE (parent)-[:HAS_COMMENTARY]->(com)
            """
            self._write_to_neo4j(flat_df, query)

    def _write_commentaries(self, para_comm_df, sched_para_comm_df, sched_subpara_comm_df):
        print("Writing Commentary Nodes...")
        self._write_single_commentary(para_comm_df, "Paragraph")
        self._write_single_commentary(sched_para_comm_df, "ScheduleParagraph")
        self._write_single_commentary(sched_subpara_comm_df, "ScheduleSubparagraph")

    def _write_citations(self, all_comms):
        if "citations" not in all_comms.schema["commentary"].dataType.fieldNames():
            return

        print("Writing Citation Nodes... (Pass 1: Nodes & Target Acts)")
        citations_df = (
            all_comms.select(
                col("legis_uri"),
                col("commentary.ref_id").alias("comm_id"),
                explode_outer("commentary.citations").alias("citation"),
            )
            .filter(col("citation").isNotNull())
            .filter(col("citation.uri").isNotNull())
        )

        expected_fields = ["id", "uri", "title", "year", "class", "text"]
        actual_fields = citations_df.schema["citation"].dataType.fieldNames()
        safe_cols = [col("legis_uri"), col("comm_id")] + self._safe_extract(
            "citation", expected_fields, actual_fields, "cit_"
        )

        citations_flat = (
            citations_df.select(*safe_cols)
            .filter(col("cit_id").isNotNull())
            .withColumn("norm_uri", regexp_replace(col("cit_uri"), r"/id/", "/"))
        )

        pass_1_df = citations_flat.dropDuplicates(["legis_uri", "cit_id"]).repartition(
            10, "norm_uri"
        )

        query_1 = """
            UNWIND event AS row
            WITH row WHERE row.cit_id IS NOT NULL AND row.legis_uri IS NOT NULL
            MERGE (cit:Citation {id: row.legis_uri + "#" + row.cit_id})
            SET cit.uri = row.cit_uri,
                cit.title = row.cit_title,
                cit.year = row.cit_year,
                cit.class = row.cit_class,
                cit.text = row.cit_text
            WITH cit, row WHERE row.norm_uri IS NOT NULL
            MATCH (leg:Legislation {uri: row.norm_uri})
            MERGE (cit)-[:CITES]->(leg)
        """
        self._write_to_neo4j(pass_1_df, query_1)

        print("Writing Citation Nodes... (Pass 2: Parent Commentary Links)")
        pass_2_df = citations_flat.dropDuplicates(["comm_id", "cit_id"]).repartition(
            10, "legis_uri", "comm_id"
        )

        query_2 = """
            UNWIND event AS row
            WITH row WHERE row.comm_id IS NOT NULL AND row.cit_id IS NOT NULL AND row.legis_uri IS NOT NULL
            MATCH (com:Commentary {id: row.legis_uri + "#" + row.comm_id})
            MATCH (cit:Citation {id: row.legis_uri + "#" + row.cit_id})
            MERGE (com)-[:HAS_CITATION]->(cit)
        """
        self._write_to_neo4j(pass_2_df, query_2)

    def _write_citation_subrefs(self, all_comms):
        if "citation_subrefs" not in all_comms.schema["commentary"].dataType.fieldNames():
            return

        print("Writing Citation SubRefs... (Pass 1: SubRef Nodes & Target Acts)")
        subrefs_df = (
            all_comms.select(
                col("legis_uri"),
                col("commentary.ref_id").alias("comm_id"),
                explode_outer("commentary.citation_subrefs").alias("subref"),
            )
            .filter(col("subref").isNotNull())
            .filter(col("subref.uri").isNotNull())
        )

        expected_fields = ["id", "citation_ref", "uri", "section_ref", "text"]
        actual_fields = subrefs_df.schema["subref"].dataType.fieldNames()
        safe_cols = [col("legis_uri"), col("comm_id")] + self._safe_extract(
            "subref", expected_fields, actual_fields, "sub_"
        )

        subrefs_base = (
            subrefs_df.select(*safe_cols)
            .filter(col("sub_id").isNotNull())
            .withColumn(
                "base_uri",
                regexp_replace(
                    col("sub_uri"),
                    r"(http://www\.legislation\.gov\.uk)/id/([^/]+/[0-9]+/[0-9]+).*",
                    "$1/$2",
                ),
            )
        )

        pass_1_df = subrefs_base.dropDuplicates(["legis_uri", "sub_id"]).repartition(
            10, "base_uri"
        )

        query_1 = """
            UNWIND event AS row
            WITH row WHERE row.sub_id IS NOT NULL AND row.legis_uri IS NOT NULL
            MERGE (sub:CitationSubRef {id: row.legis_uri + "#" + row.sub_id})
            SET sub.uri = row.sub_uri,
                sub.section_ref = row.sub_section_ref,
                sub.text = row.sub_text
            WITH sub, row WHERE row.base_uri IS NOT NULL
            MATCH (leg:Legislation {uri: row.base_uri})
            MERGE (sub)-[:REFERENCES]->(leg)
        """
        self._write_to_neo4j(pass_1_df, query_1)

        print("Writing Citation SubRefs... (Pass 2: Parent Relationships)")
        pass_2_df = (
            subrefs_base.dropDuplicates(["comm_id", "sub_id"])
            .withColumn("actual_parent", coalesce(col("sub_citation_ref"), col("comm_id")))
            .repartition(10, "legis_uri", "actual_parent")
        )

        query_2 = """
            UNWIND event AS row
            WITH row WHERE row.comm_id IS NOT NULL AND row.sub_id IS NOT NULL AND row.legis_uri IS NOT NULL
            MATCH (sub:CitationSubRef {id: row.legis_uri + "#" + row.sub_id})
            MATCH (com:Commentary {id: row.legis_uri + "#" + row.comm_id})
            OPTIONAL MATCH (cit:Citation {id: row.legis_uri + "#" + row.sub_citation_ref})
            FOREACH (_ IN CASE WHEN cit IS NOT NULL THEN [1] ELSE [] END | MERGE (cit)-[:HAS_SUBREF]->(sub))
            FOREACH (_ IN CASE WHEN cit IS NULL THEN [1] ELSE [] END | MERGE (com)-[:HAS_SUBREF]->(sub))
        """
        self._write_to_neo4j(pass_2_df, query_2)

    def _write_super_relationships(self, raw_df):
        if "super" not in raw_df.columns:
            return

        print("Writing Super Relationships...")
        actual_fields = raw_df.schema["super"].dataType.fieldNames()
        safe_cols = [col("legislation_url").alias("legis_uri")] + self._safe_extract(
            "super", ["supersedes", "superseded_by"], actual_fields
        )

        super_df = raw_df.select(*safe_cols)

        query = """
            UNWIND event AS row
            WITH row WHERE row.legis_uri IS NOT NULL
            MATCH (l:Legislation {uri: row.legis_uri})
            FOREACH (_ IN CASE WHEN row.supersedes IS NOT NULL THEN [1] ELSE [] END |
                MERGE (target:Legislation {uri: row.supersedes})
                MERGE (l)-[:SUPERSEDES]->(target)
            )
            FOREACH (_ IN CASE WHEN row.superseded_by IS NOT NULL THEN [1] ELSE [] END |
                MERGE (target:Legislation {uri: row.superseded_by})
                MERGE (l)-[:SUPERSEDED_BY]->(target)
            )
        """
        self._write_to_neo4j(super_df, query)

    def _write_explanatory_notes_nodes(self, raw_df):
        if "explanatory_notes" not in raw_df.columns:
            return None

        print("Writing Explanatory Notes Nodes...")
        notes_base_df = (
            raw_df.select(
                col("legislation_url").alias("legis_uri"), col("explanatory_notes")
            )
            .filter(col("explanatory_notes").isNotNull())
            .withColumn(
                "notes_id",
                coalesce(
                    col("explanatory_notes.uri"),
                    concat(
                        col("legis_uri"),
                        lit("#en_"),
                        md5(col("explanatory_notes").cast("string")),
                    ),
                ),
            )
        )

        query_base = """
            UNWIND event AS row
            MATCH (l:Legislation {uri: row.legis_uri})
            MERGE (en:ExplanatoryNotes {id: row.notes_id})
            SET en.uri = row.`explanatory_notes.uri`
            MERGE (l)-[:HAS_EXPLANATORY_NOTES]->(en)
        """
        self._write_to_neo4j(notes_base_df, query_base)

        notes_paras_df = (
            notes_base_df.select(
                col("notes_id"),
                col("legis_uri"),
                explode_outer("explanatory_notes.paragraphs").alias("paragraph"),
            )
            .filter(col("paragraph").isNotNull())
            .withColumn(
                "para_id",
                concat(
                    col("notes_id"),
                    lit("#enp_"),
                    md5(col("paragraph.text").cast("string")),
                ),
            )
        )

        query_paras = """
            UNWIND event AS row
            MATCH (en:ExplanatoryNotes {id: row.notes_id})
            MERGE (p:ExplanatoryNotesParagraph {id: row.para_id})
            SET p.text = row.`paragraph.text`,
                p.uri = row.`paragraph.uri`
            MERGE (en)-[:HAS_PARAGRAPH]->(p)
        """
        self._write_to_neo4j(notes_paras_df, query_paras)
        return notes_paras_df

    def _write_explanatory_notes_citations(self, notes_paras_df):
        if (
            notes_paras_df is None
            or "citations" not in notes_paras_df.schema["paragraph"].dataType.fieldNames()
        ):
            return

        print("Writing Explanatory Notes Citation Nodes (Pass 1: Creation & Parent Links)...")
        citations_df = notes_paras_df.select(
            col("legis_uri"),
            col("para_id"),
            explode_outer("paragraph.citations").alias("citation"),
        ).filter(col("citation").isNotNull())

        expected_fields = ["id", "uri", "title", "year", "class", "text"]
        actual_fields = citations_df.schema["citation"].dataType.fieldNames()
        safe_cols = [col("legis_uri"), col("para_id")] + self._safe_extract(
            "citation", expected_fields, actual_fields, "cit_"
        )

        citations_base = (
            citations_df.select(*safe_cols)
            .filter(col("cit_id").isNotNull())
            .dropDuplicates(["para_id", "cit_id"])
            .withColumn("norm_uri", regexp_replace(col("cit_uri"), r"/id/", "/"))
        )

        pass_1_df = citations_base.repartition(10, "legis_uri", "para_id")

        query_1 = """
            UNWIND event AS row
            WITH row WHERE row.para_id IS NOT NULL AND row.cit_id IS NOT NULL AND row.legis_uri IS NOT NULL
            MATCH (p:ExplanatoryNotesParagraph {id: row.para_id})
            MERGE (cit:Citation {id: row.legis_uri + "#" + row.cit_id})
            SET cit.uri = row.cit_uri,
                cit.title = row.cit_title,
                cit.year = row.cit_year,
                cit.class = row.cit_class,
                cit.text = row.cit_text
            MERGE (p)-[:HAS_CITATION]->(cit)
        """
        self._write_to_neo4j(pass_1_df, query_1)

        print("Writing Explanatory Notes Citation Links (Pass 2: Target Legislation)...")
        pass_2_df = citations_base.filter(col("norm_uri").isNotNull()).repartition(
            10, "norm_uri"
        )

        query_2 = """
            UNWIND event AS row
            WITH row WHERE row.cit_id IS NOT NULL AND row.legis_uri IS NOT NULL AND row.norm_uri IS NOT NULL
            MATCH (cit:Citation {id: row.legis_uri + "#" + row.cit_id})
            MATCH (leg:Legislation {uri: row.norm_uri})
            MERGE (cit)-[:CITES]->(leg)
        """
        self._write_to_neo4j(pass_2_df, query_2)

    def _build_spark(self):
        from pyspark.sql import SparkSession

        return (
            SparkSession.builder.appName("Legislation Full Graph Builder")
            .config(
                "spark.jars.packages",
                "org.neo4j:neo4j-connector-apache-spark_2.12:5.3.2_for_spark_3",
            )
            .config("neo4j.url", self.uri)
            .config("neo4j.authentication.basic.username", self.user)
            .config("neo4j.authentication.basic.password", self.password)
            .getOrCreate()
        )

    def load_full_hierarchy_to_neo4j(self, json_dir: Optional[str] = None):
        if not _HAS_PYSPARK:
            raise RuntimeError("PySpark is required to run the loader.")
        if json_dir is None:
            json_dir = f"{self.json_output_dir}/*/*.json"

        spark = self._build_spark()

        raw_df = (
            spark.read.option("multiline", "true")
            .option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .option("recursiveFileLookup", "true")
            .option("pathGlobFilter", "*.json")
            .json(json_dir)
        )

        if "_corrupt_record" in raw_df.columns:
            raw_df = raw_df.filter(col("_corrupt_record").isNull()).drop("_corrupt_record")

        raw_df = raw_df.filter(
            col("legislation_url").isNotNull() & (col("legislation_url") != "")
        )

        self._write_legislation_nodes(raw_df)
        self._write_super_relationships(raw_df)
        parts_df = self._write_part_nodes(raw_df)
        chapters_df = self._write_chapter_nodes(parts_df)
        sections_df = self._write_section_nodes(chapters_df)
        paragraphs_df = self._write_paragraph_nodes(sections_df)

        sched_paras_df, sched_para_comm_df, sched_subpara_comm_df = (
            self._write_schedules_nodes(raw_df)
        )
        notes_paras_df = self._write_explanatory_notes_nodes(raw_df)
        self._write_explanatory_notes_citations(notes_paras_df)

        sec_comm_df = sections_df.select(
            col("legis_uri"),
            col("sec_id").alias("parent_id"),
            explode_outer("section.commentaries").alias("commentary"),
        ).filter(col("commentary").isNotNull())

        para_comm_df = paragraphs_df.select(
            col("legis_uri"),
            col("para_id").alias("parent_id"),
            explode_outer("paragraph.commentaries").alias("commentary"),
        ).filter(col("commentary").isNotNull())

        self._write_commentaries(para_comm_df, sched_para_comm_df, sched_subpara_comm_df)

        all_comms = sec_comm_df.select("legis_uri", "commentary").unionByName(
            para_comm_df.select("legis_uri", "commentary"), allowMissingColumns=True
        )
        if sched_para_comm_df is not None:
            all_comms = all_comms.unionByName(
                sched_para_comm_df.select("legis_uri", "commentary"),
                allowMissingColumns=True,
            )
        if sched_subpara_comm_df is not None:
            all_comms = all_comms.unionByName(
                sched_subpara_comm_df.select("legis_uri", "commentary"),
                allowMissingColumns=True,
            )

        self._write_citations(all_comms)
        self._write_citation_subrefs(all_comms)

        print("Graph load complete!")


def setup_neo4j_constraints(uri, user, password, database, constraints: Optional[List[str]] = None) -> None:
    """Ensure unique constraints/indexes exist before the Spark load."""
    print("Setting up Neo4j constraints...")
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            for query in constraints or CONSTRAINTS:
                session.run(query)
    finally:
        driver.close()
    print("Constraints successfully applied.\n")


def main(config: Optional[Config] = None) -> None:
    config = (config or load_config()).require_neo4j()
    setup_neo4j_constraints(
        config.neo4j_uri, config.neo4j_user, config.neo4j_password, config.neo4j_database
    )
    loader = LegislationGraphLoader(
        config.neo4j_uri,
        config.neo4j_user,
        config.neo4j_password,
        config.json_output_dir,
        database=config.neo4j_database,
    )
    loader.load_full_hierarchy_to_neo4j()


if __name__ == "__main__":
    main()
