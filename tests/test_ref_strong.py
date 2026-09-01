import json
import unittest

from ingest import (
    REF_EVIDENCE_VERSION,
    _build_current_edges,
    _enrich_ref_with_evidence,
    _get_or_create_paper_for_ref,
    _migrate_papers_cache,
    _migrate_refs_cache,
    _parse_reference,
    _prune_orphan_papers,
    _resolve_ref,
)
from reconcile import make_evidence, reconcile_document


def _paper(paper_id, *, doi=None, arxiv=None):
    return {
        "id": paper_id,
        "doi": doi,
        "arxiv": arxiv,
        "title": "",
        "title_norm": "",
        "authors": [],
        "authors_norm": [],
        "documents": [],
    }


class RefStrongIdTests(unittest.TestCase):
    def test_existing_arxiv_paper_with_doi_doi_only_ref_same_paper(self):
        # existing paper has arxiv:1905.03277 and doi:10.1145/3306346.3323024 (like Handheld)
        papers = {}
        # create paper via arxiv + doi as front-page would
        p_arxiv, _ = reconcile_document(
            papers,
            make_evidence(
                title="Handheld Multi-Frame Super-Resolution",
                authors=["Bartlomiej Wronski", "Ignacio Garcia-Dorado"],
                arxiv="1905.03277v2",
                doi="10.1145/3306346.3323024",
            ),
            "a" * 64,
        )
        self.assertEqual(p_arxiv, "arxiv:1905.03277")
        self.assertEqual(papers[p_arxiv]["doi"], "10.1145/3306346.3323024")
        # DOI-only ref should reuse same paper, not create doi:10...
        ref = {"doi": "10.1145/3306346.3323024", "raw": "Wronski et al. 2019. Handheld ... doi:10.1145/3306346.3323024"}
        # ensure parsing would give doi
        self.assertEqual(_parse_reference(ref["raw"]).get("doi"), "10.1145/3306346.3323024")
        pid, via = _get_or_create_paper_for_ref(papers, ref)
        self.assertEqual(pid, p_arxiv)
        self.assertEqual(via, "doi")
        self.assertEqual(len(papers), 1, "should not create second paper")

    def test_same_paper_cited_once_by_arxiv_and_once_by_doi_one_node(self):
        papers = {}
        # existing paper with both identifiers
        p0, _ = reconcile_document(
            papers,
            make_evidence(
                title="Handheld Multi-Frame Super-Resolution",
                authors=["Bartlomiej Wronski", "Ignacio Garcia-Dorado"],
                arxiv="1905.03277",
                doi="10.1145/3306346.3323024",
            ),
            "a" * 64,
        )
        self.assertEqual(p0, "arxiv:1905.03277")
        # two refs: one arXiv-only, one DOI-only, both for same paper
        ref_arxiv = {"arxiv": "1905.03277", "arxiv_version": "v2", "raw": "Wronski et al. arXiv:1905.03277v2"}
        ref_doi = {"doi": "10.1145/3306346.3323024", "raw": "Wronski et al. doi:10.1145/3306346.3323024"}
        pid1, via1 = _get_or_create_paper_for_ref(papers, ref_arxiv)
        pid2, via2 = _get_or_create_paper_for_ref(papers, ref_doi)
        self.assertEqual(pid1, p0)
        self.assertEqual(pid2, p0)
        self.assertEqual(pid1, pid2)
        self.assertEqual(via1, "arxiv")
        self.assertEqual(via2, "doi")
        # only one paper node, not two
        self.assertEqual(len([k for k in papers if k.startswith("arxiv:") or k.startswith("doi:")]), 1)
        # simulate edge dedup: same source cites same target twice via different refs -> one edge
        source = "arxiv:9999.00000"
        papers[source] = {"id": source, "arxiv": "9999.00000", "title": "", "title_norm": "", "authors": [], "authors_norm": [], "documents": []}
        edge_set = set()
        for ref_pid in [pid1, pid2]:
            edge_set.add((source, ref_pid))
        self.assertEqual(len(edge_set), 1, "same paper cited via arXiv and DOI should dedup to one edge")

    def test_both_identifiers_in_one_ref_no_second_identity(self):
        papers = {}
        ref_both = {"doi": "10.1234/abc.1", "arxiv": "1234.56789", "arxiv_version": "v1", "raw": "Paper arXiv:1234.56789 doi:10.1234/abc.1"}
        pid, via = _get_or_create_paper_for_ref(papers, ref_both)
        # should create one paper, not two
        self.assertIsNotNone(pid)
        self.assertEqual(len(papers), 1)
        # via should be doi (since doi present, we prefer doi for via? but either is okay, check it is one of them)
        self.assertIn(via, ("doi", "arxiv"))
        # paper should store both identifiers for future matching
        paper = papers[pid]
        self.assertEqual(paper.get("doi"), "10.1234/abc.1")
        self.assertEqual(paper.get("arxiv"), "1234.56789")

    def test_both_identifiers_point_to_different_papers_conflict(self):
        papers = {}
        # two existing papers with different strong IDs
        p1, _ = reconcile_document(papers, make_evidence(title="Paper One", authors=["Alice"], doi="10.1000/one"), "a"*64)
        p2, _ = reconcile_document(papers, make_evidence(title="Paper Two", authors=["Bob"], arxiv="1234.56789v1"), "b"*64)
        self.assertNotEqual(p1, p2)
        # ref containing both identifiers, each pointing to different papers -> conflict
        ref_both = {"doi": "10.1000/one", "arxiv": "1234.56789", "raw": "conflict ref doi 10.1000/one arxiv 1234.56789"}
        pid, via = _get_or_create_paper_for_ref(papers, ref_both)
        self.assertIsNone(pid)
        self.assertIsNone(via)
        # no new paper created for conflict
        self.assertEqual(len(papers), 2)

    def test_doi_only_paper_then_both_then_arxiv_only_one_node(self):
        # DOI-only paper -> ref with DOI+arXiv enriches -> arXiv-only ref finds same paper
        papers = {}
        p_doi, _ = reconcile_document(
            papers, make_evidence(title="Paper X", authors=["Alice"], doi="10.1234/x"), "a" * 64
        )
        self.assertEqual(p_doi, "doi:10.1234/x")
        self.assertIsNone(papers[p_doi].get("arxiv"))

        ref_both = {"doi": "10.1234/x", "arxiv": "2401.12345", "arxiv_version": "v1", "raw": "Paper doi:10.1234/x arXiv:2401.12345"}
        pid1, via1 = _get_or_create_paper_for_ref(papers, ref_both)
        self.assertEqual(pid1, p_doi)
        # enriched with arXiv
        self.assertEqual(papers[p_doi].get("arxiv"), "2401.12345")
        self.assertEqual(len(papers), 1)

        ref_arxiv_only = {"arxiv": "2401.12345", "raw": "Paper arXiv:2401.12345"}
        pid2, via2 = _get_or_create_paper_for_ref(papers, ref_arxiv_only)
        self.assertEqual(pid2, p_doi)
        self.assertEqual(pid2, pid1)
        self.assertEqual(len(papers), 1)
        # both identifiers stored on single paper
        self.assertEqual(papers[p_doi]["doi"], "10.1234/x")
        self.assertEqual(papers[p_doi]["arxiv"], "2401.12345")
        self.assertNotIn("arxiv_version", papers[p_doi])
        # edges dedup
        source = "arxiv:9999.00000"
        papers[source] = {"id": source, "arxiv": "9999.00000", "title": "", "title_norm": "", "authors": [], "authors_norm": [], "documents": []}
        edge_set = {(source, pid1), (source, pid2)}
        self.assertEqual(len(edge_set), 1)

    def test_arxiv_only_paper_then_both_then_doi_only_one_node(self):
        # arXiv-only paper -> ref with arXiv+DOI enriches -> DOI-only ref finds same paper
        papers = {}
        p_arxiv, _ = reconcile_document(
            papers, make_evidence(title="Paper Y", authors=["Bob"], arxiv="2401.12345v1"), "a" * 64
        )
        self.assertEqual(p_arxiv, "arxiv:2401.12345")
        self.assertIsNone(papers[p_arxiv].get("doi"))

        ref_both = {"doi": "10.1234/x", "arxiv": "2401.12345", "raw": "Paper arXiv:2401.12345 doi:10.1234/x"}
        pid1, via1 = _get_or_create_paper_for_ref(papers, ref_both)
        self.assertEqual(pid1, p_arxiv)
        self.assertEqual(papers[p_arxiv].get("doi"), "10.1234/x")
        self.assertEqual(len(papers), 1)

        ref_doi_only = {"doi": "10.1234/x", "raw": "Paper doi:10.1234/x"}
        pid2, via2 = _get_or_create_paper_for_ref(papers, ref_doi_only)
        self.assertEqual(pid2, p_arxiv)
        self.assertEqual(pid2, pid1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[p_arxiv]["doi"], "10.1234/x")
        self.assertEqual(papers[p_arxiv]["arxiv"], "2401.12345")
        self.assertNotIn("arxiv_version", papers[p_arxiv])
        source = "arxiv:9999.00001"
        papers[source] = {"id": source, "arxiv": "9999.00001", "title": "", "title_norm": "", "authors": [], "authors_norm": [], "documents": []}
        edge_set = {(source, pid1), (source, pid2)}
        self.assertEqual(len(edge_set), 1)

    def test_conflicting_enrichment_does_not_overwrite(self):
        papers = {}
        p, _ = reconcile_document(
            papers, make_evidence(title="Paper Z", authors=["Carol"], arxiv="2401.12345v1"), "a" * 64
        )
        # add doi to paper
        papers[p]["doi"] = "10.1234/x"
        # ref has same arXiv but different DOI -> should conflict, not overwrite
        ref_conflict = {"doi": "10.9999/other", "arxiv": "2401.12345", "raw": "conflict arXiv 2401.12345 doi 10.9999/other"}
        pid, via = _get_or_create_paper_for_ref(papers, ref_conflict)
        self.assertIsNone(pid)
        self.assertEqual(papers[p]["doi"], "10.1234/x", "should not overwrite existing DOI")
        self.assertEqual(len(papers), 1)

    def test_arxiv_versions_stay_on_refs_and_share_one_paper(self):
        papers = {}
        refs = [_parse_reference(f"Paper arXiv:2401.12345v{version}") for version in (1, 2, 3)]

        resolved = []
        for ref, version in zip(refs, (1, 2, 3)):
            pid, via = _get_or_create_paper_for_ref(papers, ref)
            resolved.append(pid)
            self.assertEqual(via, "arxiv")
            self.assertEqual(ref["arxiv"], "2401.12345")
            self.assertEqual(ref["arxiv_version"], f"v{version}")

        self.assertEqual(resolved, ["arxiv:2401.12345"] * 3)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers["arxiv:2401.12345"]["arxiv"], "2401.12345")
        self.assertNotIn("arxiv_version", papers["arxiv:2401.12345"])

    def test_paper_version_migration_removes_only_legacy_paper_field(self):
        papers = {
            "arxiv:2401.12345": {
                "id": "arxiv:2401.12345",
                "arxiv": "2401.12345",
                "arxiv_version": "v2",
                "title": "Keep this",
                "documents": ["a" * 64],
            }
        }

        self.assertTrue(_migrate_papers_cache(papers))
        self.assertNotIn("arxiv_version", papers["arxiv:2401.12345"])
        self.assertEqual(papers["arxiv:2401.12345"]["title"], "Keep this")
        self.assertEqual(papers["arxiv:2401.12345"]["documents"], ["a" * 64])
        self.assertFalse(_migrate_papers_cache(papers))


class RefEvidenceTests(unittest.TestCase):
    def test_author_year_extracts_ordered_authors_and_punctuated_title(self):
        ref = _parse_reference(
            "Smith, J., Doe, A. (2020). A study of GPT-4: R&D in 3D. Journal of Tests, 12."
        )

        self.assertEqual(ref["authors"], ["Smith, J.", "Doe, A."])
        self.assertEqual(ref["title"], "A study of GPT-4: R&D in 3D")
        self.assertEqual(ref["title_norm"], "a study of gpt-4 r&d in 3d")
        self.assertTrue(ref["authors_complete"])

    def test_full_name_author_list_extracts_title(self):
        ref = _parse_reference(
            "John Smith and Alice Doe. A title: GPT-4, R&D, 3D. In Proceedings of Tests, 2021."
        )

        self.assertEqual(ref["authors"], ["John Smith", "Alice Doe"])
        self.assertEqual(ref["title"], "A title: GPT-4, R&D, 3D")

    def test_quoted_title_extracts_authors_before_quote(self):
        ref = _parse_reference('John Smith and Alice Doe. "Quoted title: GPT-4". In Tests, 2020.')

        self.assertEqual(ref["authors"], ["John Smith", "Alice Doe"])
        self.assertEqual(ref["title"], "Quoted title: GPT-4")

    def test_et_al_marks_author_evidence_incomplete(self):
        ref = _parse_reference("Smith, J., Doe, A., et al. A title. In Journal, 2020.")

        self.assertEqual(ref["authors"], ["Smith, J.", "Doe, A."])
        self.assertFalse(ref["authors_complete"])
        self.assertEqual(ref["title"], "A title")

    def test_organization_prefix_does_not_become_a_person_author(self):
        for raw, title in (
            (
                "Adobe. 2012. Digital Negative (DNG) Specification. In Tests, 2012.",
                "Digital Negative (DNG) Specification",
            ),
            (
                "National Institute of Standards and Technology. 2020. A reference title. In Tests, 2020.",
                "A reference title",
            ),
        ):
            with self.subTest(raw=raw):
                ref = _parse_reference(raw)
                self.assertNotIn("authors", ref)
                self.assertEqual(ref["title"], title)

    def test_ambiguous_prose_is_left_unparsed(self):
        ref = _parse_reference("This is prose about Smith, J. and Doe, A. with no citation title.")

        self.assertNotIn("title", ref)
        self.assertNotIn("authors", ref)

    def test_title_stops_before_url_and_venue_only_references_are_skipped(self):
        with_url = _parse_reference(
            "S. Boyd, L. Xiao, and A. Mutapcic. Subgradient methods.https://example.test/subgrad.pdf, 2003."
        )
        venue_only = _parse_reference(
            "Donnelly, J. and Roegiest, A., 2019. European Conference on Information Retrieval, pp. 795--802."
        )

        self.assertEqual(with_url["title"], "Subgradient methods")
        self.assertNotIn("http", with_url["title"])
        self.assertNotIn("title", venue_only)

    def test_truncated_title_is_not_emitted(self):
        ref = _parse_reference("Lisa Torrey and Jude Shavlik. 2010. Transfer learn-")

        self.assertNotIn("title", ref)
        self.assertEqual(ref["authors"], ["Lisa Torrey", "Jude Shavlik"])
        self.assertEqual(ref["year"], "2010")

    def test_glued_final_author_is_not_treated_as_title(self):
        expected_authors = ["Fang, J.", "Deng, X.", "Chen, H.", "Zhang, N"]
        for separator in (";andZhang, N.", ";and Zhang, N."):
            raw = f"Fang, J.; Deng, X.; Chen, H.{separator} 2026. LightMem: Lightweight and Efficient Memory-"
            with self.subTest(separator=separator):
                ref = _parse_reference(raw)
                self.assertNotIn("title", ref)
                self.assertEqual(ref["authors"], expected_authors)
                self.assertNotIn("andZhang", " ".join(ref["authors"]))

    def test_no_space_venue_tail_stops_at_title(self):
        cases = (
            (
                "Navindra Persaud and Alan Cowey. Blindsight is unlike normal conscious vision: evidence from an exclusion task.Consciousness and cognition, 17(3):1050–1055, 2008.",
                "Blindsight is unlike normal conscious vision: evidence from an exclusion task",
            ),
            (
                "Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition.Neural Computation, 1(4):541–551, 1989.",
                "Backpropagation applied to handwritten zip code recognition",
            ),
            (
                "Stanislas Dehaene and Jean-Pierre Changeux. Experimental and theoretical approaches to conscious processing.Neuron, 70(2):200–227, 2011.",
                "Experimental and theoretical approaches to conscious processing",
            ),
        )
        for raw, expected_title in cases:
            with self.subTest(raw=raw):
                ref = _parse_reference(raw)
                self.assertEqual(ref["title"], expected_title)

    def test_enrichment_preserves_resolution_when_strong_evidence_is_unchanged(self):
        raw = "John Smith and Alice Doe. A title: GPT-4, R&D, 3D. arXiv:2101.12345v2, 2021. doi:10.1234/parsed"
        ref = {
            "index": 7,
            "raw": raw,
            "doi": "https://doi.org/10.1234/parsed",
            "arxiv": "2101.12345",
            "arxiv_version": "v9",
            "paper_id": "arxiv:2101.12345",
            "status": "resolved",
            "resolved_via": "arxiv",
        }

        self.assertTrue(_enrich_ref_with_evidence(ref))
        self.assertEqual(ref["raw"], raw)
        self.assertEqual(ref["index"], 7)
        self.assertEqual(ref["doi"], "10.1234/parsed")
        self.assertEqual(ref["arxiv"], "2101.12345")
        self.assertEqual(ref["arxiv_version"], "v2")
        self.assertEqual(ref["paper_id"], "arxiv:2101.12345")
        self.assertEqual(ref["status"], "resolved")
        self.assertEqual(ref["resolved_via"], "arxiv")
        self.assertEqual(ref["evidence_version"], REF_EVIDENCE_VERSION)
        self.assertFalse(_enrich_ref_with_evidence(ref))

    def test_changed_arxiv_invalidates_resolution_and_rebuilds_only_current_edge(self):
        raw = "Some paper arXiv:2222.22222"
        cache = {
            "doc": {
                "refs": [{
                    "index": 0,
                    "raw": raw,
                    "arxiv": "1111.11111",
                    "paper_id": "arxiv:1111.11111",
                    "status": "resolved",
                    "resolved_via": "arxiv",
                    "evidence_version": REF_EVIDENCE_VERSION - 1,
                }],
            }
        }
        source = "arxiv:0000.00000"
        papers = {
            source: _paper(source, arxiv="0000.00000"),
            "arxiv:1111.11111": _paper("arxiv:1111.11111", arxiv="1111.11111"),
            "arxiv:2222.22222": _paper("arxiv:2222.22222", arxiv="2222.22222"),
        }
        manifest = {"doc": {"paper_id": source}}

        self.assertTrue(_migrate_refs_cache(cache))
        ref = cache["doc"]["refs"][0]
        self.assertNotIn("paper_id", ref)
        self.assertNotIn("status", ref)
        self.assertNotIn("resolved_via", ref)

        paper_id, via, changed = _resolve_ref(papers, ref)
        self.assertTrue(changed)
        self.assertEqual((paper_id, via), ("arxiv:2222.22222", "arxiv"))
        self.assertEqual(_prune_orphan_papers(papers, cache, manifest), ["arxiv:1111.11111"])
        self.assertNotIn("arxiv:1111.11111", papers)
        self.assertIn("arxiv:2222.22222", papers)
        self.assertEqual(_build_current_edges(cache, manifest, papers), {(source, "arxiv:2222.22222")})

    def test_equivalent_doi_normalization_preserves_resolution(self):
        raw = "A paper. doi:10.1234/ABC."
        ref = {
            "index": 0,
            "raw": raw,
            "doi": "https://doi.org/10.1234/ABC",
            "paper_id": "doi:10.1234/abc",
            "status": "resolved",
            "resolved_via": "doi",
            "evidence_version": REF_EVIDENCE_VERSION - 1,
        }
        cache = {"doc": {"refs": [ref]}}
        papers = {ref["paper_id"]: _paper(ref["paper_id"], doi="10.1234/abc")}

        self.assertTrue(_migrate_refs_cache(cache))
        paper_id, via, changed = _resolve_ref(papers, ref)
        self.assertFalse(changed)
        self.assertEqual((paper_id, via), ("doi:10.1234/abc", "doi"))
        self.assertEqual(ref["doi"], "10.1234/abc")

    def test_title_author_year_refresh_preserves_strong_resolution(self):
        raw = "John Smith and Alice Doe. New title. doi:10.1234/same. 2021."
        ref = {
            "index": 0,
            "raw": raw,
            "doi": "10.1234/SAME",
            "year": "1900",
            "title": "Stale title",
            "authors": ["Stale Author"],
            "paper_id": "doi:10.1234/same",
            "status": "resolved",
            "resolved_via": "doi",
            "resolution_note": "keep",
            "evidence_version": REF_EVIDENCE_VERSION - 1,
        }
        cache = {"doc": {"refs": [ref]}}
        papers = {ref["paper_id"]: _paper(ref["paper_id"], doi="10.1234/same")}
        stable = {field: ref[field] for field in ("paper_id", "status", "resolved_via", "resolution_note")}

        self.assertTrue(_migrate_refs_cache(cache))
        self.assertEqual({field: ref[field] for field in stable}, stable)
        self.assertEqual(ref["title"], "New title")
        self.assertEqual(ref["authors"], ["John Smith", "Alice Doe"])
        self.assertEqual(ref["year"], "2021")
        self.assertFalse(_resolve_ref(papers, ref)[2])

    def test_disappearing_strong_evidence_clears_resolution_and_edge(self):
        raw = "John Smith and Alice Doe. A title. 2021."
        old_target = "arxiv:1111.11111"
        source = "arxiv:0000.00000"
        ref = {
            "index": 0,
            "raw": raw,
            "arxiv": "1111.11111",
            "paper_id": old_target,
            "status": "resolved",
            "resolved_via": "arxiv",
            "evidence_version": REF_EVIDENCE_VERSION - 1,
        }
        cache = {"doc": {"refs": [ref]}}
        papers = {
            source: _paper(source, arxiv="0000.00000"),
            old_target: _paper(old_target, arxiv="1111.11111"),
        }
        manifest = {"doc": {"paper_id": source}}

        self.assertTrue(_migrate_refs_cache(cache))
        self.assertIsNone(_resolve_ref(papers, ref)[0])
        self.assertTrue(all(field not in ref for field in ("paper_id", "status", "resolved_via")))
        self.assertEqual(_build_current_edges(cache, manifest, papers), set())

    def test_orphan_paper_stays_when_another_ref_still_cites_it(self):
        old_target = "arxiv:1111.11111"
        source = "arxiv:0000.00000"
        cache = {
            "doc": {
                "refs": [
                    {"arxiv": "2222.22222", "paper_id": "arxiv:2222.22222", "status": "resolved"},
                    {"arxiv": "1111.11111", "paper_id": old_target, "status": "resolved"},
                ]
            }
        }
        papers = {
            source: _paper(source, arxiv="0000.00000"),
            old_target: _paper(old_target, arxiv="1111.11111"),
            "arxiv:2222.22222": _paper("arxiv:2222.22222", arxiv="2222.22222"),
        }

        self.assertEqual(_prune_orphan_papers(papers, cache, {"doc": {"paper_id": source}}), [])
        self.assertIn(old_target, papers)

    def test_orphan_paper_stays_when_document_backed(self):
        paper_id = "arxiv:1111.11111"
        papers = {paper_id: _paper(paper_id, arxiv="1111.11111")}
        papers[paper_id]["documents"] = ["d" * 64]

        self.assertEqual(_prune_orphan_papers(papers, {}, {}), [])
        self.assertIn(paper_id, papers)

    def test_orphan_paper_stays_when_it_is_a_manifest_source(self):
        paper_id = "arxiv:1111.11111"
        papers = {paper_id: _paper(paper_id, arxiv="1111.11111")}

        self.assertEqual(
            _prune_orphan_papers(papers, {}, {"doc": {"paper_id": paper_id}}),
            [],
        )
        self.assertIn(paper_id, papers)

    def test_orphan_paper_stays_when_required_by_reconciliation_conflict(self):
        candidate = "arxiv:1111.11111"
        conflict = "paper:conflict"
        papers = {
            candidate: _paper(candidate, arxiv="1111.11111"),
            conflict: {
                **_paper(conflict),
                "documents": ["d" * 64],
                "reconciliation_conflict": [candidate],
            },
        }

        self.assertEqual(_prune_orphan_papers(papers, {}, {}), [])
        self.assertIn(candidate, papers)

    def test_second_migration_and_resolution_are_state_idempotent(self):
        raw = "A paper. doi:10.1234/ABC."
        ref = {
            "index": 0,
            "raw": raw,
            "doi": "10.1234/ABC",
            "paper_id": "doi:10.1234/abc",
            "status": "resolved",
            "resolved_via": "doi",
            "evidence_version": REF_EVIDENCE_VERSION - 1,
        }
        cache = {"doc": {"refs": [ref]}}
        manifest = {"doc": {"paper_id": "arxiv:0000.00000"}}
        papers = {
            "arxiv:0000.00000": _paper("arxiv:0000.00000", arxiv="0000.00000"),
            "doi:10.1234/abc": _paper("doi:10.1234/abc", doi="10.1234/abc"),
        }

        self.assertTrue(_migrate_refs_cache(cache))
        self.assertEqual(_resolve_ref(papers, ref)[:2], ("doi:10.1234/abc", "doi"))
        before = json.dumps({"cache": cache, "papers": papers, "edges": sorted(_build_current_edges(cache, manifest, papers))}, sort_keys=True)

        self.assertFalse(_migrate_refs_cache(cache))
        self.assertFalse(_resolve_ref(papers, ref)[2])
        after = json.dumps({"cache": cache, "papers": papers, "edges": sorted(_build_current_edges(cache, manifest, papers))}, sort_keys=True)
        self.assertEqual(after, before)

    def test_evidence_migration_removes_legacy_truncated_title(self):
        raw = "Lisa Torrey and Jude Shavlik. 2010. Transfer learn-"
        cache = {"doc": {"mode": "author", "refs": [{"index": 0, "raw": raw, "title": "Transfer learn-", "title_norm": "transfer learn-"}]}}

        self.assertTrue(_migrate_refs_cache(cache))
        ref = cache["doc"]["refs"][0]
        self.assertNotIn("title", ref)
        self.assertNotIn("title_norm", ref)
        self.assertEqual(ref["authors"], ["Lisa Torrey", "Jude Shavlik"])
        self.assertEqual(ref["year"], "2010")
        self.assertEqual(ref["evidence_version"], REF_EVIDENCE_VERSION)
        self.assertFalse(_migrate_refs_cache(cache))

    def test_evidence_migration_reparses_glued_final_author(self):
        raw = "Fang, J.; Deng, X.; Chen, H.;andZhang, N. 2026. LightMem: Lightweight and Efficient Memory-"
        cache = {
            "doc": {
                "mode": "author",
                "refs": [{"index": 0, "raw": raw, "title": "andZhang, N", "title_norm": "andzhang n"}],
            }
        }

        self.assertTrue(_migrate_refs_cache(cache))
        ref = cache["doc"]["refs"][0]
        self.assertNotIn("title", ref)
        self.assertNotIn("title_norm", ref)
        self.assertEqual(ref["authors"][-1], "Zhang, N")
        self.assertEqual(ref["authors"][:3], ["Fang, J.", "Deng, X.", "Chen, H."])

    def test_evidence_migration_replaces_legacy_venue_leak(self):
        raw = "Navindra Persaud and Alan Cowey. Blindsight is unlike normal conscious vision: evidence from an exclusion task.Consciousness and cognition, 17(3):1050–1055, 2008."
        cache = {
            "doc": {
                "mode": "bracket",
                "refs": [{
                    "index": 3,
                    "raw": raw,
                    "title": "Blindsight is unlike normal conscious vision: evidence from an exclusion task.Consciousness and cognition, 17(3):1050–1055",
                    "title_norm": "stale",
                }],
            }
        }

        self.assertTrue(_migrate_refs_cache(cache))
        ref = cache["doc"]["refs"][0]
        self.assertEqual(ref["index"], 3)
        self.assertEqual(ref["title"], "Blindsight is unlike normal conscious vision: evidence from an exclusion task")
        self.assertEqual(ref["title_norm"], "blindsight is unlike normal conscious vision evidence from an exclusion task")

    def test_evidence_migration_preserves_stable_reference_fields(self):
        raw = "John Smith and Alice Doe. A title. 2021."
        stable = {
            "index": 7,
            "raw": raw,
            "paper_id": "arxiv:2101.12345",
            "status": "resolved",
            "resolved_via": "arxiv",
            "resolution_note": "keep",
        }
        cache = {"doc": {"mode": "author", "refs": [{**stable, "title": "stale", "title_norm": "stale"}]}}

        self.assertTrue(_migrate_refs_cache(cache))
        ref = cache["doc"]["refs"][0]
        self.assertEqual({key: ref[key] for key in stable}, stable)
        self.assertNotEqual(ref["title"], "stale")


if __name__ == "__main__":
    unittest.main()
