import re
import sys
from pathlib import Path

from pypdf import PdfReader

HEADING = re.compile(r"(?m)^\s*(References|Bibliography)\s*:?\s*$", re.IGNORECASE)

# post-reference section starts; corpus-demonstrated forms: "Appendix",
# "A Appendix", "A. Appendix", "• Appendices". Mid-sentence mentions
# ("described in Appendix") don't start a line, so they never match.
STOP = re.compile(
    r"(?m)^[ \t\u2022]*(?:[A-Z]\.\s*|[A-Z]\s+)?"
    r"(Appendices|Supplementary Material|Supplemental Material|Appendix)\b",
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
