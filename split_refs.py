"""Lineage 2.0 bibliography splitter — deterministic, no LLM.

Two modes only:
  - bracket:  [1] [2] ... numbered references
  - author:  author-year style, line-based with year veto

Promoted from C:\\Temp\\opencode\\final_eval.py (1679 refs, 6 over / 40 under on 26 unnumbered).
Covers 35-paper corpus in docs/.
"""
import re

REF_SPLITTER_VERSION = 3

BLOCK = {"In","URL","ISBN","DOI","Vol","Proc","IEEE","ACM","Ed","Eds","The","Accessed","arXiv","Available","Crossref","Retrieved","Figure","Table","Fig","Listing","Equation"}

# Corpus-confirmed venue/title tails that the generic head patterns mistake for authors.
RE_KNOWN_CONTINUATION = re.compile(
    r"^(?:Trans\.|Proceedings\.|Conference Track\b|Information Processing\b|"
    r"Information Systems\b|Processing Systems\b|Computational Linguistics\b|"
    r"Language Processing\b|Language Model\b|Language Resources\b|"
    r"Language Technologies\b|Linguistics\b|Vision Conference\b|"
    r"Computer Visual\b|Res Zoom\b|Learn\. Res\.|System Demonstrations\b|"
    r"Machine Learning\b|American Chapter\b|Empirical Methods\b|"
    r"Neural Information\b|Natural Language\b|Long (?:Beach|Papers)\b)",
    re.IGNORECASE,
)

RE_SURNAME_TIGHT = re.compile(r"^[A-Z][\w'\u2019\-]+,\s*[A-Z][\.;,]")
RE_FULLNAME = re.compile(r"^[A-Z][a-z'\u2019]+(?:-[A-Z][a-z'\u2019]+)*\s+[A-Z]")
RE_ORG = re.compile(r"^[A-Za-z][\w'\u2019\-]+\.\s*(?:\((?:19|20)\d{2}[a-z]?\)|(?:19|20)\d{2})")
RE_INIT_DOT = re.compile(r"^(?:[A-Z]\.\s*)+[A-Z][\w'\u2019\-]+")
RE_INIT_SPACE = re.compile(r"^[A-Z]\s+[A-Z][\w'\u2019\-]+")
RE_ORG_NEW = re.compile(r"^[A-Za-z][\w@&'\u2019\-]*\.\s*(?:19|20)\d{2}[a-z]?\b", re.I)
RE_ORG_SIMPLE = re.compile(r"^[A-Za-z][\w@&'\u2019\-]*\.\s+[A-Z]")
RE_ORG_MULTI = re.compile(r"^(?:[A-Za-z][\w@&'\u2019\-]*\s+)?[A-Za-z][\w@&'\u2019\-]*\.\s+[A-Z]")
YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")

def is_head(s: str) -> bool:
    t = s.strip()
    if len(t) < 4 or t[0].isdigit() or t[0] == '(':
        return False
    if RE_KNOWN_CONTINUATION.match(t):
        return False
    m = re.match(r"[A-Za-z][\w@&'\u2019\-]*", t)
    if not m or m.group() in BLOCK:
        return False
    if t[0].islower():
        return False
    if RE_ORG_NEW.match(t):
        return True
    if RE_ORG_SIMPLE.match(t):
        return True
    if RE_ORG_MULTI.match(t):
        return True
    if RE_SURNAME_TIGHT.match(t):
        return True
    return bool(RE_FULLNAME.match(t) or RE_INIT_DOT.match(t) or RE_INIT_SPACE.match(t) or RE_ORG.match(t))

def split_author(body: str) -> list[str]:
    entries, cur = [], []
    for ln in body.splitlines():
        if cur and is_head(ln):
            txt = " ".join(cur)
            if not YEAR.search(txt):
                cur.append(ln.strip())
                continue
            entries.append(txt.strip())
            cur = [ln.strip()]
        else:
            cur.append(ln.strip())
    if cur:
        entries.append(" ".join(cur).strip())
    return [e for e in entries if e]

def split_bib(raw: str) -> tuple[str, list[str]]:
    """raw = full bibliography text including heading line. Returns (mode, entries)."""
    body = raw.split("\n", 1)[1] if "\n" in raw else raw
    if len(re.findall(r"\[\d{1,4}\]", body)) >= 8:
        parts = [p.strip() for p in re.split(r"\[\d{1,4}\]", body) if p.strip()]
        return "bracket", parts
    return "author", split_author(body)

# ponytail: tight surname + ORG_NEW + year veto are deliberate ceilings; loosen only if new corpus shows systematic under-splits
