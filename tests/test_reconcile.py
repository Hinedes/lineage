import unittest

from reconcile import (
    authors_compatible,
    extract_arxiv,
    make_evidence,
    normalize_title,
    parse_author_list,
    reconcile_document,
)


class ReconcileTests(unittest.TestCase):
    def test_title_normalization_erases_formatting_noise(self):
        self.assertEqual(
            normalize_title("A paper for GPT"),
            normalize_title("a PAPER, for GPT"),
        )

    def test_title_normalization_keeps_semantic_hyphen(self):
        self.assertNotEqual(normalize_title("GPT-4"), normalize_title("GPT 4"))

    def test_metadata_author_string_splits_into_people(self):
        authors = parse_author_list("John Smith, Alice Doe")
        self.assertEqual([a["surname"] for a in authors], ["smith", "doe"])

    def test_initial_can_match_full_given_name(self):
        a = parse_author_list(["John Smith", "Alice Doe"])
        b = parse_author_list(["J. Smith", "A. Doe"])
        self.assertTrue(authors_compatible(a, b))

    def test_different_full_given_names_conflict(self):
        a = parse_author_list(["John Smith"])
        b = parse_author_list(["Joseph Smith"])
        self.assertFalse(authors_compatible(a, b))

    def test_explicit_arxiv_marker_keeps_version_separate(self):
        base, version = extract_arxiv("arXiv:2203.05482v3 [cs.LG] 1 Jul 2022")
        self.assertEqual(base, "2203.05482")
        self.assertEqual(version, "v3")

    def test_arxiv_versions_reconcile_to_one_paper(self):
        papers = {}
        p1, s1 = reconcile_document(
            papers,
            make_evidence(
                title="Example",
                authors=["John Smith", "Alice Doe"],
                arxiv="2203.05482v1",
            ),
            "a" * 64,
        )
        p2, s2 = reconcile_document(
            papers,
            make_evidence(
                title="Example",
                authors=["J. Smith", "A. Doe"],
                arxiv="2203.05482v3",
            ),
            "b" * 64,
        )
        self.assertEqual(p1, "arxiv:2203.05482")
        self.assertEqual(p1, p2)
        self.assertEqual(s1, "new")
        self.assertEqual(s2, "matched-arxiv")
        self.assertEqual(len(papers[p1]["documents"]), 2)

    def test_doi_reconciles_same_paper(self):
        papers = {}
        p1, _ = reconcile_document(
            papers,
            make_evidence(
                title="A paper for GPT",
                authors=["John Smith"],
                doi="https://doi.org/10.1234/ABC.7",
            ),
            "a" * 64,
        )
        p2, status = reconcile_document(
            papers,
            make_evidence(
                title="a PAPER, for GPT",
                authors=["J. Smith"],
                doi="doi:10.1234/abc.7",
            ),
            "b" * 64,
        )
        self.assertEqual(p1, p2)
        self.assertEqual(status, "matched-doi")

    def test_title_plus_compatible_authors_is_fallback(self):
        papers = {}
        p1, _ = reconcile_document(
            papers,
            make_evidence(
                title="A paper for GPT",
                authors=["John Smith", "Alice Doe"],
            ),
            "a" * 64,
        )
        p2, status = reconcile_document(
            papers,
            make_evidence(
                title="a PAPER, for GPT",
                authors=["J. Smith", "A. Doe"],
            ),
            "b" * 64,
        )
        self.assertEqual(p1, p2)
        self.assertEqual(status, "matched-title-authors")

    def test_same_title_but_conflicting_full_author_does_not_merge(self):
        papers = {}
        p1, _ = reconcile_document(
            papers,
            make_evidence(title="A paper for GPT", authors=["John Smith"]),
            "a" * 64,
        )
        p2, status = reconcile_document(
            papers,
            make_evidence(title="A paper for GPT", authors=["Joseph Smith"]),
            "b" * 64,
        )
        self.assertNotEqual(p1, p2)
        self.assertEqual(status, "new")

    def test_strong_identifier_conflict_does_not_merge_by_title(self):
        papers = {}
        p1, _ = reconcile_document(
            papers,
            make_evidence(
                title="A paper for GPT",
                authors=["John Smith"],
                doi="10.1000/one",
            ),
            "a" * 64,
        )
        p2, status = reconcile_document(
            papers,
            make_evidence(
                title="A paper for GPT",
                authors=["John Smith"],
                doi="10.1000/two",
            ),
            "b" * 64,
        )
        self.assertNotEqual(p1, p2)
        self.assertEqual(status, "new")


if __name__ == "__main__":
    unittest.main()
