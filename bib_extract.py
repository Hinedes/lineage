import re
import sys
from pathlib import Path

from pypdf import PdfReader

HEADING = re.compile(r"(?m)^\s*(References|Bibliography)\s*:?\s*$", re.IGNORECASE)

# Post-reference starts include explicit appendix/supplement labels and the
# corpus-confirmed standalone lettered heading shape. The latter requires a
# heading-like next line and rejects citation years, sentence punctuation, and
# venue continuations such as "In Proceedings".
STOP = re.compile(
    r"(?m)(?:"
    r"(?i:^[ \t\u2022]*(?:[A-Z]\.\s*|[A-Z]\s+)?"
    r"(Appendices|Supplementary Material|Supplemental Material|Appendix)\b"
    r"|^[ \t\u2022]*S[ \t]+SUPPLEMENT[ \t]*$)"
    r"|^[ \t\u2022]*(?-i:[A-L])(?:\.\s+|\s+)"
    r"(?=[A-Z])"
    r"(?![^\n]*\b(?:19|20)\d{2}\b)"
    r"(?![^\n]*(?-i:\bIn\b)[ \t]*$)"
    r"(?![^\n]*\n[ \t\u2022]*(?-i:[a-z]))"
    r"(?![^\n]*\n[^\n]*(?i:\bProceedings\b))"
    r"(?=[^\n]*\n[ \t\u2022]*(?-i:[A-Z0-9]))"
    r"(?=[^\n]{4,120}$)"
    r"(?=[^\n]*[^\s.,:;][ \t]*$)"
    r"(?:"
    r"(?-i:[A-Z][^.,:;\n]*)"
    r"|(?-i:[A-Z][^a-z.\n]*)"
    r")[ \t]*$"
    r"|^[ \t]*(?i:CONTENTS)[ \t]*$"
    r")",
    re.IGNORECASE,
)


def extract(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    m = HEADING.search(text)
    if not m:
        return f"FAIL {Path(pdf_path).name} | no References/Bibliography heading found"
    heading = m.group(1)
    tail = text[m.start():]
    s = STOP.search(tail)
    bib = tail[:s.start()] if s else tail
    out = Path(pdf_path).with_suffix(".references.txt")
    out.write_text(bib, encoding="utf-8")
    return f"OK   {Path(pdf_path).name} | heading={heading} | chars={len(bib)} | out={out.name}"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console may be cp1251; filenames can be CJK
    for p in sys.argv[1:]:
        try:
            print(extract(p))
        except Exception as e:
            print(f"FAIL {Path(p).name} | {type(e).__name__}: {e}")
