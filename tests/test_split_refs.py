import unittest

from ingest import REF_SPLITTER_VERSION, _migrate_splitter_entry
from split_refs import is_head, split_bib


class SplitterTests(unittest.TestCase):
    def test_lowercase_continuation_is_not_a_head(self):
        self.assertFalse(is_head("saicing algorithm. IEEE Trans. Image Processing 14, 3 (2005)."))

    def test_lowercase_prose_continuation_is_not_a_head(self):
        self.assertFalse(is_head("representations. In Proceedings of the 58th Conference."))

    def test_organization_heads_remain_heads(self):
        self.assertTrue(is_head("Adobe. 2012. Digital Negative (DNG) Specification."))
        self.assertTrue(is_head("OpenAI. 2025. Codex CLI."))

    def test_personal_author_heads_remain_heads(self):
        self.assertTrue(is_head("Smith, J., Doe, A. 2020. A title."))
        self.assertTrue(is_head("Jane Smith and Alice Doe. 2020. A title."))

    def test_bracket_mode_is_unchanged(self):
        mode, refs = split_bib(
            "References:\n"
            + "\n".join(f"[{i}] Paper {i}. 2020." for i in range(1, 9))
        )
        self.assertEqual(mode, "bracket")
        self.assertEqual(len(refs), 8)
        self.assertEqual(refs[0], "Paper 1. 2020.")


class SplitterCacheMigrationTests(unittest.TestCase):
    def test_identical_sequence_preserves_ref_object_and_resolution(self):
        ref = {
            "index": 4,
            "raw": "Smith. 2020. A title.",
            "paper_id": "arxiv:2020.00001",
            "status": "resolved",
            "resolved_via": "arxiv",
            "evidence_version": 5,
        }
        entry = {"mode": "author", "refs": [ref]}

        mutated, boundaries_changed = _migrate_splitter_entry(
            entry, "author", [ref["raw"]]
        )

        self.assertTrue(mutated)
        self.assertFalse(boundaries_changed)
        self.assertIs(entry["refs"][0], ref)
        self.assertEqual(entry["refs"][0]["paper_id"], "arxiv:2020.00001")
        self.assertEqual(entry["splitter_version"], REF_SPLITTER_VERSION)

    def test_changed_sequence_replaces_refs_without_old_resolution(self):
        entry = {
            "mode": "author",
            "refs": [
                {
                    "index": 0,
                    "raw": "Jane Smith. 2020. A title.",
                    "paper_id": "arxiv:2020.00001",
                    "status": "resolved",
                    "resolved_via": "arxiv",
                },
                {"index": 1, "raw": "continuation."},
            ],
        }

        mutated, boundaries_changed = _migrate_splitter_entry(
            entry, "author", ["Jane Smith. 2020. A title. continuation."]
        )

        self.assertTrue(mutated)
        self.assertTrue(boundaries_changed)
        self.assertEqual(len(entry["refs"]), 1)
        self.assertEqual(entry["refs"][0]["index"], 0)
        self.assertNotIn("paper_id", entry["refs"][0])
        self.assertNotIn("status", entry["refs"][0])
        self.assertNotIn("resolved_via", entry["refs"][0])
        self.assertEqual(entry["splitter_version"], REF_SPLITTER_VERSION)

    def test_same_count_different_sequence_still_replaces_refs(self):
        entry = {
            "mode": "author",
            "refs": [
                {"index": 0, "raw": "old one", "paper_id": "stale"},
                {"index": 1, "raw": "old two"},
            ],
        }

        mutated, boundaries_changed = _migrate_splitter_entry(
            entry, "author", ["new one", "new two"]
        )

        self.assertTrue(mutated)
        self.assertTrue(boundaries_changed)
        self.assertEqual([ref["raw"] for ref in entry["refs"]], ["new one", "new two"])
        self.assertNotIn("paper_id", entry["refs"][0])

    def test_legacy_string_sequence_is_materialized_when_unchanged(self):
        raw = "Jane Smith. 2020. A title."
        entry = {"mode": "author", "refs": [raw]}

        mutated, boundaries_changed = _migrate_splitter_entry(entry, "author", [raw])

        self.assertTrue(mutated)
        self.assertFalse(boundaries_changed)
        self.assertIsInstance(entry["refs"][0], dict)
        self.assertEqual(entry["refs"][0]["raw"], raw)
        self.assertEqual(entry["splitter_version"], REF_SPLITTER_VERSION)

    def test_current_splitter_version_is_not_reprocessed(self):
        entry = {
            "splitter_version": REF_SPLITTER_VERSION,
            "refs": [{"index": 0, "raw": "old", "paper_id": "keep"}],
        }

        mutated, boundaries_changed = _migrate_splitter_entry(entry, "author", ["new"])

        self.assertFalse(mutated)
        self.assertFalse(boundaries_changed)
        self.assertEqual(entry["refs"][0]["raw"], "old")


if __name__ == "__main__":
    unittest.main()
