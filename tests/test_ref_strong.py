import unittest

from ingest import _get_or_create_paper_for_ref, _migrate_papers_cache, _parse_reference
from reconcile import make_evidence, reconcile_document


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


if __name__ == "__main__":
    unittest.main()
