import unittest

from ingest import _get_or_create_paper_for_ref, _parse_reference
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


if __name__ == "__main__":
    unittest.main()
