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
        )

        self.assertIsNone(STOP.search(text))

    def test_unobserved_bare_supplement_forms_are_not_enabled(self):
        for heading in ("SUPPLEMENT", "S. SUPPLEMENT"):
            with self.subTest(heading=heading):
                self.assertIsNone(STOP.search("References\n" + heading + "\n"))


if __name__ == "__main__":
    unittest.main()
