import unittest

from bib_extract import STOP


class BibliographyStopTests(unittest.TestCase):
    def test_corpus_s_supplement_heading_stops_extraction(self):
        text = (
            "REFERENCES\n"
            "Lei Zhang, Xiaolin Wu, Antoni Buades, and Xin Li. 2011. "
            "Color demosaicking by local directional interpolation.\n"
            "S SUPPLEMENT\n"
            "S.1 Adaptive Super-Resolution and Denoising\n"
        )

        stop = STOP.search(text)

        self.assertIsNotNone(stop)
        self.assertEqual(stop.group(0), "S SUPPLEMENT")
        self.assertTrue(
            text[: stop.start()]
            .splitlines()[-1]
            .endswith("Color demosaicking by local directional interpolation.")
        )

    def test_supplement_inside_reference_is_not_a_stop(self):
        text = (
            "References\n"
            "J. Phang, T. Fevry, and S. R. Bowman. 2018. "
            "Sentence encoders on stilts: Supplementary training on intermediate labeled-data tasks.\n"
            "This reference mentions S SUPPLEMENT in prose.\n"
        )

        self.assertIsNone(STOP.search(text))

    def test_unobserved_bare_supplement_forms_are_not_enabled(self):
        for heading in ("SUPPLEMENT", "S. SUPPLEMENT"):
            with self.subTest(heading=heading):
                self.assertIsNone(STOP.search("References\n" + heading + "\n"))

    def test_lettered_section_heading_stops_extraction(self):
        for heading in ("A Evaluation data", "A E XPERIMENTS", "A. Overview"):
            with self.subTest(heading=heading):
                text = "References\nJane Smith. 2020. A title.\n" + heading + "\nThis is appendix prose.\n"
                stop = STOP.search(text)

                self.assertIsNotNone(stop)
                self.assertEqual(stop.group(0), heading)

    def test_contents_stops_extraction(self):
        stop = STOP.search("References\nJane Smith. 2020. A title.\nCONTENTS\n1 Introduction 1\n")

        self.assertIsNotNone(stop)
        self.assertEqual(stop.group(0), "CONTENTS")

    def test_citation_like_lettered_lines_are_not_section_stops(self):
        for text in (
            "References\n"
            "Jeffrey O. Zhang. 2020. A title.\n"
            "A Baseline for Network Adaptation via Additive\n"
            "Side Networks. In Proceedings of ECCV, 2020a.\n",
            "References\n"
            "Dai, D. 2025. A title.\n"
            "J. Finding skill neurons in pre-trained transformer-based\n"
            "language models. In Proceedings.\n",
        ):
            with self.subTest(text=text):
                self.assertIsNone(STOP.search(text))


if __name__ == "__main__":
    unittest.main()
