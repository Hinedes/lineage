"""Lineage 2.0 ingest — manual refresh, dead-simple.

PDF is source of truth. No LLM, no S2.

  python ingest.py              # refresh docs/ (default)
  python ingest.py docs/        # explicit dir
  python ingest.py --clear      # wipe local stores

Identity:
  - document = sha256(pdf bytes)
  - paper reconciliation = DOI -> arXiv base ID -> normalized title + authors
  - arXiv version is document metadata, not paper identity
  - side stamp "arXiv:2203.05482v3 [cs.LG] 1 Jul 2022" is robust, filename-independent

Stores:
  .cache/lineage2.json  documents
  .cache/papers.json    reconciled papers
  .cache/graph.json     current graph skeleton
  .cache/refs.json      per-document split references (from split_bib, not thrown away)
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

import bib_extract
from reconcile import (
    extract_arxiv,
    extract_authors_from_front_page,
    extract_doi,
    make_evidence,
    normalize_arxiv,
    normalize_doi,
    normalize_title,
    parse_author_list,
    reconcile_document,
)
from split_refs import split_bib

ARXIV_FN = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.I)
# vertical side stamp unique to arXiv PDFs — filename-independent, survives renames
# e.g. "arXiv:2203.05482v3  [cs.LG]  1 Jul 2022" (pypdf extracts vertical at page tail)
ARXIV_STAMP = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)\s*\[([^\]]+)\]\s*(\d+\s+\w+\s+\d{4})", re.I)
ARXIV_DATE = re.compile(r"(\d{2})(\d{2})\.\d+")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_YEAR_TOKEN = r"(?:19|20)\d{2}[a-z]?"
_PREFIX_YEAR_RE = re.compile(
    rf"^(?P<authors>.+?)(?:\s*[.,]\s*|\s+\(\s*){_YEAR_TOKEN}\s*\)?\s*[.:,]?\s+(?P<rest>.+)$",
    re.I,
)
_NAME_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*\.?", re.UNICODE)
_NAME_PARTICLES = {"al", "bin", "da", "de", "del", "den", "der", "di", "du", "la", "le", "st", "ten", "van", "von"}
_AUTHOR_STOPWORDS = {
    "a", "an", "the", "this", "that", "what", "why", "how", "in", "on", "of", "for",
    "you", "using", "dataset", "details", "implementation", "baseline", "conference", "track",
    "proceedings", "computational", "linguistics", "language", "processing", "evaluation", "method",
    "analysis", "edit", "succ",
}
_AUTHOR_METADATA_WORDS = {
    "acm", "arxiv", "doi", "eds", "ieee", "issue", "journal", "no", "pages", "pp", "proc",
    "published", "vol", "volume",
}
_ORGANIZATION_MARKERS = {
    "agency", "association", "center", "centre", "college", "committee", "company", "consortium",
    "corporation", "foundation", "institute", "laboratory", "ministry", "organization", "organisation",
    "research", "society", "technology", "technologies", "university",
}
_TITLE_ABBREVIATION_RE = re.compile(r"\b(?:e\.g|i\.e|etc|vs|fig|no|vol|pp|dr|mr|ms|prof)\.$", re.I)
_TITLE_WRAP_RE = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014]\s*$")
_GLUED_AUTHOR_RE = re.compile(
    r"(^|[;,]\s*)and(?=\s*[A-Z][^,;]{1,50},\s*[A-Z](?:\.|\s|$))", re.I
)
_VENUE_PREFIX = (
    r"(?:(?:the\s+)?journal(?:\s+of(?:\s+[A-Za-z][\w-]*){0,8})?|"
    r"(?:the\s+)?proceedings(?:\s+of(?:\s+[A-Za-z][\w-]*){0,8})?|"
    r"ima\s+journal|neural\s+computation|consciousness\s+and\s+cognition|"
    r"(?:the\s+)?annals\s+of\s+statistics|neuron|arxiv(?:\s+preprint)?|corr|"
    r"siam|ieee|acm|transactions|international\s+conference|advances\s+in|"
    r"american\s+mathematical\s+society|springer|nature|neurips|icml|iclr|cvpr|"
    r"acl|emnlp|distill|dokl|commun|sn)"
)
REF_EVIDENCE_VERSION = 3
DERIVED_EVIDENCE_FIELDS = (
    "doi", "arxiv", "arxiv_version", "year", "title", "title_norm", "authors", "authors_norm",
    "authors_complete",
)
RESOLUTION_FIELDS = ("paper_id", "status", "resolved_via")

CACHE_DIR = Path(".cache")
MANIFEST = CACHE_DIR / "lineage2.json"
PAPERS = CACHE_DIR / "papers.json"
GRAPH = CACHE_DIR / "graph.json"
REFS = CACHE_DIR / "refs.json"  # per-document split references (from split_bib)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _arxiv_date(base: str | None) -> str | None:
    if not base:
        return None
    m = ARXIV_DATE.search(base)
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    yyyy = 2000 + yy if yy < 80 else 1900 + yy
    return f"{yyyy:04d}-{mm:02d}"


def _normalize_stamp_date(s: str | None) -> str | None:
    # "1 Jul 2022" -> "2022-07-01", " 1 Jul 2022 " tolerant; returns None on failure
    if not s:
        return None
    import datetime
    s = s.strip()
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _author_parts(text: str) -> tuple[list[str], bool]:
    """Return conservative ordered author pieces and whether et al. made them partial."""
    s = re.sub(r"\s+", " ", text or "").strip(" ,;")
    incomplete = bool(re.search(r"\bet\s+al\.?\b", s, re.I))
    s = re.sub(r"(?:,?\s*(?:and\s+)?et\s+al\.?)\s*$", "", s, flags=re.I).strip(" ,;")
    s = _GLUED_AUTHOR_RE.sub(r"\1", s)
    s = re.sub(r"\s+(?:and|&)\s+", ", ", s, flags=re.I)
    chunks = [p.strip() for p in re.split(r"\s*;\s*", s) if p.strip()]
    pieces = []
    for chunk in chunks:
        comma_parts = [p.strip() for p in chunk.split(",") if p.strip()]
        i = 0
        while i < len(comma_parts):
            part = comma_parts[i]
            part_tokens = _NAME_TOKEN_RE.findall(part)
            next_tokens = _NAME_TOKEN_RE.findall(comma_parts[i + 1]) if i + 1 < len(comma_parts) else []
            if (
                i + 1 < len(comma_parts)
                and len(part_tokens) == 1
                and 1 <= len(next_tokens) <= 2
                and (len(next_tokens) == 1 or all(len(token.rstrip(".")) == 1 for token in next_tokens))
            ):
                pieces.append(f"{part}, {comma_parts[i + 1]}")
                i += 2
            else:
                pieces.append(part)
                i += 1
    return pieces, incomplete


def _is_author_piece(text: str) -> bool:
    s = re.sub(r"\s+", " ", text or "").strip(" ,.;")
    if not s or len(s) > 80 or re.search(r"[0-9@/:\[\]{}()\"“”+=<>|•]", s):
        return False
    tokens = _NAME_TOKEN_RE.findall(s)
    max_tokens = 6 if "," in s else 4
    if not 2 <= len(tokens) <= max_tokens or not parse_author_list([s]):
        return False
    cores = [token.rstrip(".") for token in tokens]
    if any(core.casefold() in _AUTHOR_METADATA_WORDS for core in cores):
        return False
    if any(not core or not core[0].isupper() and core.casefold() not in _NAME_PARTICLES for core in cores):
        return False
    if any(
        token.endswith(".")
        and len(core) > 1
        and core.casefold() not in {"jr", "sr"} | _NAME_PARTICLES
        for token, core in zip(tokens, cores)
    ):
        return False
    first = cores[0].casefold()
    if first in _AUTHOR_STOPWORDS and "." not in tokens[0]:
        return False
    if len(tokens) > 2 and "," not in s and not any(token.endswith(".") or len(core) == 1 for token, core in zip(tokens, cores)):
        return False
    if "," in s:
        surname = cores[0]
        if len(surname) < 2 or (len(cores[-1]) != 1 and len(cores) > 3):
            return False
    elif len(cores[-1]) < 2:
        return False
    return True


def _parse_author_evidence(text: str) -> tuple[list[str], bool]:
    pieces, incomplete = _author_parts(text)
    if not pieces or not all(_is_author_piece(piece) for piece in pieces):
        return [], incomplete
    authors = [re.sub(r"\s+", " ", piece).strip() for piece in pieces]
    if len(parse_author_list(authors)) != len(authors):
        return [], incomplete
    return authors, not incomplete


def _clean_title(text: str) -> str:
    title = re.sub(r"\s+", " ", text or "").strip()
    title = re.split(r"(?=(?:https?://|www\.|(?:arxiv|doi)\s*:))", title, maxsplit=1, flags=re.I)[0].rstrip()
    if title.endswith(".") and not title.endswith("..."):
        title = title[:-1].rstrip()
    title = re.sub(r"\s*[,;]\s*(?:19|20)\d{2}[a-z]?\s*$", "", title, flags=re.I)
    title = re.sub(r"\s*\((?:19|20)\d{2}[a-z]?\)\s*$", "", title, flags=re.I)
    title = re.sub(r"\s*,\s+volume\s+\d+\s*$", "", title, flags=re.I)
    title = re.sub(r"\s*,\s+(?:pp?\.?|pages)\s+\d+(?:\s*[-–—]\s*\d+)?\s*$", "", title, flags=re.I)
    return title.strip(" ,;")


def _author_evidence(authors: list[str], complete: bool) -> dict:
    return {
        "authors": authors,
        "authors_norm": parse_author_list(authors),
        "authors_complete": complete,
    }


def _is_organization_prefix(text: str) -> bool:
    tokens = _NAME_TOKEN_RE.findall(text or "")
    if len(tokens) == 1:
        return len(tokens[0].rstrip(".")) >= 2 and tokens[0][0].isupper()
    cores = [token.rstrip(".") for token in tokens]
    if not 2 <= len(cores) <= 8:
        return False
    if any(not core or not core[0].isupper() and core.casefold() not in {"and", "for", "of", "the"} for core in cores):
        return False
    return any(core.casefold() in _ORGANIZATION_MARKERS for core in cores)


def _title_from_rest(text: str) -> str | None:
    rest = re.sub(r"\s+", " ", text or "").strip(" ,;")
    rest = re.sub(r"^(?:(?:18|19|20)\d{2}[a-z]?(?:[.:,])?\s*)+", "", rest, flags=re.I)
    if not rest or re.match(
        r"(?:https?://|www\.|(?:arxiv|doi)\s*:|arxiv\s+preprint|corr\b|url\b|"
        r"(?:in\s+)?(?:the\s+)?(?:proceedings|journal)\b|\(?eds?\.?\)?\s*[,;:])",
        rest,
        re.I,
    ):
        return None
    if re.match(r"^in(?:\s*the)?\b", rest, re.I) and re.search(r"\b(?:conference|journal|proceedings)\b", rest[:180], re.I):
        return None
    if re.search(r"\b(?:conference|journal)\b", rest[:120], re.I) and re.search(r"\b(?:vol\.?|volume|pp\.?|pages)\s*\d", rest[:180], re.I):
        return None
    if re.search(r"\b(?:vol\.?|volume)\s*\d.*\b(?:pp\.?|pages)\s*\d", rest[:220], re.I):
        return None
    if re.match(r"^(?:ieee|acm|siam)\b", rest, re.I) and re.search(r"\bconference\b", rest[:180], re.I):
        return None
    rest = re.split(r",\s+volume\s+\d+\s+of(?=[A-Z])", rest, maxsplit=1, flags=re.I)[0]
    inline_venue = re.search(
        rf"[.!?]\s*(?={_VENUE_PREFIX}\b[^.!?]{{0,180}}(?:,\s*(?:\d|\(|pages?\b|pp\.?\b)|\b(?:19|20)\d{{2}}\b))",
        rest,
        re.I,
    )
    if inline_venue:
        rest = rest[:inline_venue.start() + 1]
    for match in re.finditer(r"[.!?](?=\s|$|[,;])", rest):
        candidate = rest[:match.end()].strip()
        if len(candidate.rstrip(".!? ")) < 4:
            continue
        if match.group(0) == "." and _TITLE_ABBREVIATION_RE.search(candidate):
            continue
        title = _clean_title(candidate)
        if _TITLE_WRAP_RE.search(title):
            return None
        if len(title) >= 4 and re.search(r"[A-Za-z]", title):
            return title
    url = re.search(r"\s+(?=(?:https?://|www\.|(?:arxiv|doi)\s*:))", rest, re.I)
    if url:
        rest = rest[:url.start()]
    title = _clean_title(rest)
    if _TITLE_WRAP_RE.search(title) or len(title) < 4 or len(title) > 220 or not re.search(r"[A-Za-z]", title):
        return None
    if re.search(r"(?:https?://|www\.|(?:arxiv|doi)\s*:|arxiv\s+preprint)", title, re.I):
        return None
    return title


def _starts_author_continuation(rest: str) -> bool:
    if re.match(r"et\s+al\.?\b", rest, re.I):
        return True
    if re.match(r"(?:and|&)\s+", rest, re.I):
        return True
    if _GLUED_AUTHOR_RE.match(rest):
        return True
    if re.match(r"^[A-Z](?:\.-?[A-Z])?\.?\s*,", rest):
        return True
    if re.match(r"^[A-Z](?:\.-?[A-Z])?\.\s+(?:and|&)\s+", rest, re.I):
        return True
    if re.match(r"^[A-Z](?:\.-?[A-Z])?\.\s+[A-Z][a-z]", rest):
        return True
    if re.match(r"^[A-Z][a-z'’\-]{2,},\s+[A-Z][a-z'’\-]{2,}(?:\s+[A-Z][a-z'’\-]{2,})*(?:[,.;]|$)", rest):
        return True
    if re.match(r"^(?:" + "|".join(_NAME_PARTICLES) + r")\s+[A-Z][^,.;]{1,50},", rest, re.I):
        return True
    if re.match(r"^[A-Z][^,.;]{1,70},\s*[A-Z](?:[.\s,;]|$)", rest):
        return True
    head = re.split(r"[.!?](?=\s|$)", rest, maxsplit=1)[0]
    if not ("," in head or re.search(r"\s+(?:and|&)\s+", head, re.I)):
        return False
    pieces, _ = _author_parts(head)
    if len(pieces) >= 2 and all(_is_author_piece(piece) for piece in pieces):
        return True
    comma_parts = [part.strip() for part in rest.split(",") if part.strip()]
    if len(comma_parts) >= 3:
        first = f"{comma_parts[0]}, {comma_parts[1]}"
        return bool(_parse_author_evidence(first)[0])
    first_comma = re.match(r"^([^,;]+,\s*[^,;]+)\s*[,;]", rest)
    return bool(first_comma and _parse_author_evidence(first_comma.group(1))[0])


def _title_author_evidence(raw: str) -> dict:
    """Extract only structurally defensible title/author evidence from one raw citation."""
    text = re.sub(r"\s+", " ", raw or "").strip()
    text = re.sub(r"\s+\.", ".", text)
    if len(text) < 8:
        return {}

    quote = re.search(r"[\"“](.+?)[\"”]", text)
    if quote:
        authors, complete = _parse_author_evidence(text[:quote.start()].strip(" ,.;"))
        title = _clean_title(quote.group(1))
        if authors and len(title) >= 4:
            return {"title": title, "title_norm": normalize_title(title), **_author_evidence(authors, complete)}

    prefix = _PREFIX_YEAR_RE.match(text)
    if prefix and len(prefix.group("authors")) <= 320:
        prefix_authors, complete = _parse_author_evidence(prefix.group("authors"))
        title = _title_from_rest(prefix.group("rest"))
        if title:
            result = {"title": title, "title_norm": normalize_title(title)}
            if prefix_authors:
                result.update(_author_evidence(prefix_authors, complete))
            elif _is_organization_prefix(prefix.group("authors")):
                return result
            if prefix_authors:
                return result
        elif prefix_authors:
            return _author_evidence(prefix_authors, complete)

    for boundary in re.finditer(r"\.", text):
        if (boundary.start() and text[boundary.start() - 1] == "-") or text[boundary.end():].startswith("-"):
            continue
        authors, complete = _parse_author_evidence(text[:boundary.start()])
        if not authors:
            continue
        rest = text[boundary.end():].lstrip(" ,;:")
        if not rest or _starts_author_continuation(rest):
            continue
        title = _title_from_rest(rest)
        if not title:
            return _author_evidence(authors, complete)
        return {"title": title, "title_norm": normalize_title(title), **_author_evidence(authors, complete)}
    return {}


def _parse_reference(raw: str) -> dict:
    """Deterministic evidence from a raw split reference — no resolution, no network."""
    doi = extract_doi(raw)
    arxiv_base, arxiv_version = extract_arxiv(raw)
    # also try normalize_arxiv for bare IDs without 'arXiv:' prefix (e.g. '1905.03277v2')
    if not arxiv_base:
        # fallback: search for bare arXiv ID pattern
        m = re.search(r"\b(\d{4}\.\d{4,5})(v\d+)?\b", raw)
        if m:
            arxiv_base, arxiv_version = normalize_arxiv(m.group(0))
    year = None
    ym = YEAR_RE.search(raw)
    if ym:
        year = ym.group(0)
    out = {}
    if doi:
        out["doi"] = doi
    if arxiv_base:
        out["arxiv"] = arxiv_base
        if arxiv_version:
            out["arxiv_version"] = arxiv_version
    if year:
        out["year"] = year
    out.update(_title_author_evidence(raw))
    return out


def _new_ref(index: int, raw: str) -> dict:
    return {"index": index, "raw": raw, **_parse_reference(raw), "evidence_version": REF_EVIDENCE_VERSION}


def _normalized_strong_evidence(ref: dict) -> tuple[str | None, str | None]:
    doi = normalize_doi(ref.get("doi")) if ref.get("doi") else None
    arxiv, _ = normalize_arxiv(ref.get("arxiv")) if ref.get("arxiv") else (None, None)
    return doi, arxiv


def _clear_ref_resolution(ref: dict) -> bool:
    changed = False
    for field in RESOLUTION_FIELDS:
        if field in ref:
            del ref[field]
            changed = True
    return changed


def _enrich_ref_with_evidence(ref: dict) -> bool:
    """Refresh derived evidence, invalidating resolution only when strong IDs change."""
    if ref.get("evidence_version") == REF_EVIDENCE_VERSION:
        return False
    old_strong = _normalized_strong_evidence(ref)
    raw = ref.get("raw", "")
    for field in DERIVED_EVIDENCE_FIELDS:
        ref.pop(field, None)
    parsed = _parse_reference(raw)
    ref.update(parsed)
    ref["evidence_version"] = REF_EVIDENCE_VERSION
    if old_strong != _normalized_strong_evidence(ref):
        _clear_ref_resolution(ref)
    return True


def _migrate_refs_cache(refs_cache: dict) -> bool:
    """Migrate old string refs to {index, raw} objects and enrich with evidence. Returns True if mutated."""
    mutated = False
    for entry in refs_cache.values():
        refs = entry.get("refs", [])
        if not refs:
            continue
        if isinstance(refs[0], str):
            entry["refs"] = [_new_ref(i, s) for i, s in enumerate(refs)]
            mutated = True
        elif isinstance(refs[0], dict):
            # preserve index/raw and unchanged resolution fields while refreshing evidence
            for r in refs:
                if _enrich_ref_with_evidence(r):
                    mutated = True
    return mutated


def _migrate_papers_cache(papers: dict) -> bool:
    """Remove legacy paper-level arXiv versions; refs/documents keep their versions."""
    mutated = False
    for paper in papers.values():
        if isinstance(paper, dict) and "arxiv_version" in paper:
            del paper["arxiv_version"]
            mutated = True
    return mutated


def _canonical_strong_id(ref: dict) -> tuple[str | None, str | None]:
    """Return (paper_id, via) for strong IDs only (doi/arxiv), else (None, None). — legacy, use _find_strong_matches."""
    doi = ref.get("doi")
    if doi:
        norm = normalize_doi(doi)
        if norm:
            return f"doi:{norm}", "doi"
    arxiv = ref.get("arxiv")
    if arxiv:
        base, _ = normalize_arxiv(arxiv)
        if base:
            return f"arxiv:{base}", "arxiv"
    return None, None


def _find_strong_matches(papers: dict, doi_norm: str | None, arxiv_base: str | None) -> dict[str, str]:
    """Scan papers for matching doi/arxiv evidence, not just dict keys. Returns {paper_id: via}."""
    matches: dict[str, str] = {}
    if doi_norm:
        for pid, paper in papers.items():
            paper_doi, _ = _normalized_strong_evidence(paper)
            if paper_doi == doi_norm:
                matches[pid] = "doi"
    if arxiv_base:
        for pid, paper in papers.items():
            _, paper_arxiv = _normalized_strong_evidence(paper)
            if paper_arxiv == arxiv_base:
                # if same paper already matched via doi, keep first via (doi) — same paper via both
                if pid not in matches:
                    matches[pid] = "arxiv"
    return matches


def _get_or_create_paper_for_ref(papers: dict, ref: dict) -> tuple[str | None, str | None]:
    """Get or create minimal paper for a strong-ID ref, reusing existing paper evidence.
    Returns (paper_id, via) or (None, None) for unresolved/conflict.
    """
    doi_norm = normalize_doi(ref.get("doi")) if ref.get("doi") else None
    arxiv_raw = ref.get("arxiv")
    arxiv_base, _ = normalize_arxiv(arxiv_raw) if arxiv_raw else (None, None)

    if not doi_norm and not arxiv_base:
        return None, None

    matches = _find_strong_matches(papers, doi_norm, arxiv_base)
    if len(matches) == 1:
        pid, via = next(iter(matches.items()))
        paper = papers[pid]
        # conflict check: if reference brings an identifier that differs from already-stored value
        paper_doi, paper_arxiv = _normalized_strong_evidence(paper)
        if doi_norm and paper_doi and paper_doi != doi_norm:
            return None, None
        if arxiv_base and paper_arxiv and paper_arxiv != arxiv_base:
            return None, None
        # enrich paper with missing non-conflicting strong identifiers
        if doi_norm and not paper.get("doi"):
            paper["doi"] = doi_norm
        if arxiv_base and not paper.get("arxiv"):
            paper["arxiv"] = arxiv_base
        return pid, via
    if len(matches) > 1:
        # both identifiers point to different existing papers -> identity conflict, don't silently choose
        return None, None

    # no existing match -> create minimal new paper
    # canonical id prefers doi if present? follow reconcile: arxiv first, but for refs with both we store both
    # so future DOI-only or arXiv-only refs will find it via scan
    if doi_norm:
        paper_id = f"doi:{doi_norm}"
        via = "doi"
    else:
        paper_id = f"arxiv:{arxiv_base}"
        via = "arxiv"
    # if ref has both, still create with single canonical id but store both identifiers
    # check if canonical id already exists (should not, since matches==0, but be safe)
    if paper_id in papers:
        paper = papers[paper_id]
        paper_doi, paper_arxiv = _normalized_strong_evidence(paper)
        if (doi_norm and paper_doi and paper_doi != doi_norm) or (arxiv_base and paper_arxiv and paper_arxiv != arxiv_base):
            return None, None
        if doi_norm and not paper.get("doi"):
            paper["doi"] = doi_norm
        if arxiv_base and not paper.get("arxiv"):
            paper["arxiv"] = arxiv_base
        return paper_id, via
    evidence = make_evidence(title="", authors=[], doi=doi_norm, arxiv=arxiv_base)
    papers[paper_id] = {
        "id": paper_id,
        "doi": evidence.get("doi"),
        "arxiv": evidence.get("arxiv"),
        "title": "",
        "title_norm": "",
        "authors": [],
        "authors_norm": [],
        "documents": [],
    }
    if ref.get("year"):
        papers[paper_id]["year"] = ref["year"]
    # if ref had both identifiers, ensure the non-canonical one is also stored for future matching
    # (e.g., arxiv paper created via doi, but also has arxiv)
    if doi_norm and arxiv_base:
        # paper_id is doi:... but we also want arxiv field set (already via evidence)
        # if we chose arxiv as canonical, also need doi field (already)
        pass
    return paper_id, via


def _strong_evidence_matches_paper(ref: dict, paper: dict) -> bool:
    """Only trust a cached resolution when current strong evidence matches its target."""
    doi_norm, arxiv_base = _normalized_strong_evidence(ref)
    if not doi_norm and not arxiv_base:
        return False
    paper_doi = normalize_doi(paper.get("doi")) if paper.get("doi") else None
    paper_arxiv, _ = normalize_arxiv(paper.get("arxiv")) if paper.get("arxiv") else (None, None)
    if doi_norm and paper_doi and paper_doi != doi_norm:
        return False
    if arxiv_base and paper_arxiv and paper_arxiv != arxiv_base:
        return False
    return bool((doi_norm and paper_doi == doi_norm) or (arxiv_base and paper_arxiv == arxiv_base))


def _resolve_ref(papers: dict, ref: dict) -> tuple[str | None, str | None, bool]:
    """Resolve one ref, invalidating stale cached linkage before using current evidence."""
    paper_id = ref.get("paper_id")
    if ref.get("status") == "resolved" and paper_id in papers:
        if _strong_evidence_matches_paper(ref, papers[paper_id]):
            return paper_id, ref.get("resolved_via"), False

    changed = _clear_ref_resolution(ref)
    paper_id, via = _get_or_create_paper_for_ref(papers, ref)
    if not paper_id:
        return None, None, changed
    if ref.get("paper_id") != paper_id or ref.get("status") != "resolved" or ref.get("resolved_via") != via:
        ref["paper_id"] = paper_id
        ref["status"] = "resolved"
        ref["resolved_via"] = via
        changed = True
    return paper_id, via, changed


def _build_current_edges(refs_cache: dict, manifest: dict, papers: dict) -> set[tuple[str, str]]:
    """Build citation edges solely from refs resolved against current paper evidence."""
    edge_set = set()
    for doc_sha, entry in refs_cache.items():
        manifest_entry = manifest.get(doc_sha)
        if not manifest_entry or not manifest_entry.get("paper_id"):
            continue
        source_paper = manifest_entry["paper_id"]
        if source_paper not in papers:
            continue
        for ref in entry.get("refs", []):
            target = ref.get("paper_id")
            if (
                ref.get("status") == "resolved"
                and target in papers
                and _strong_evidence_matches_paper(ref, papers[target])
            ):
                edge_set.add((source_paper, target))
    return edge_set


def _prune_orphan_papers(papers: dict, refs_cache: dict, manifest: dict) -> list[str]:
    """Remove only empty-document papers with no current or document-backed dependents."""
    referenced = {
        ref.get("paper_id")
        for entry in refs_cache.values()
        for ref in entry.get("refs", [])
        if ref.get("status") == "resolved" and ref.get("paper_id")
    }
    protected = referenced | {
        entry.get("paper_id")
        for entry in manifest.values()
        if entry.get("paper_id")
    }
    for paper_id, paper in papers.items():
        if not isinstance(paper, dict):
            continue
        if paper.get("documents") or "reconciliation_conflict" in paper:
            protected.add(paper_id)
        protected.update(
            candidate
            for candidate in paper.get("reconciliation_conflict", [])
            if isinstance(candidate, str)
        )

    removed = [
        paper_id
        for paper_id, paper in papers.items()
        if isinstance(paper, dict) and paper.get("documents") == [] and paper_id not in protected
    ]
    for paper_id in removed:
        del papers[paper_id]
    return removed


def identify_pdf(pdf: Path) -> dict:
    """Read paper identity evidence from the PDF, using filename only as fallback."""
    name = pdf.name
    title = ""
    year = None
    doi = None
    authors = []
    arxiv = None
    version = None
    first_page = ""
    arxiv_stamp = None
    arxiv_category = None
    stamp_date = None

    try:
        reader = PdfReader(str(pdf))
        if reader.pages:
            first_page = reader.pages[0].extract_text() or ""

        m_stamp = ARXIV_STAMP.search(first_page)
        if m_stamp:
            arxiv = m_stamp.group(1).casefold()
            version = m_stamp.group(2).casefold() if m_stamp.group(2) else None
            arxiv_category = m_stamp.group(3)
            stamp_date = m_stamp.group(4)
            arxiv_stamp = m_stamp.group(0)
        else:
            arxiv, version = extract_arxiv(first_page)

        meta = reader.metadata
        meta_title = (meta.title or "").strip() if meta else ""
        meta_author = (meta.author or "").strip() if meta else ""

        if meta_title and len(meta_title) >= 8 and "Microsoft Word" not in meta_title:
            title = meta_title
        else:
            for line in first_page.splitlines():
                t = line.strip()
                low = t.casefold()
                if len(t) < 12 or low.startswith("arxiv:") or low.startswith("published as"):
                    continue
                if len(t.split()) >= 3:
                    title = t
                    break
            if not title:
                collapsed = first_page.replace("\n", " ").strip()
                title = collapsed[:220].split("  ")[0].strip() if collapsed else name

        if meta_author:
            authors = meta_author
        else:
            authors = extract_authors_from_front_page(first_page, title)

        doi = extract_doi(first_page)

        if not arxiv:
            fn = ARXIV_FN.search(name)
            if fn:
                arxiv, parsed_version = normalize_arxiv(fn.group(0))
                version = version or parsed_version

        if not arxiv:
            m = YEAR_RE.search(first_page[:3000])
            if m:
                year = m.group(0)
    except Exception:
        if not title:
            title = name

    # normalize stamp date to ISO; keep ID-derived month separate to avoid mixed schema
    stamp_iso = _normalize_stamp_date(stamp_date)
    arxiv_id_month = _arxiv_date(arxiv)
    out = {
        "filename": name,
        "arxiv": arxiv,
        "version": version,
        "arxiv_date": stamp_iso,  # ISO 2022-07-01 from stamp if present, else None
        "arxiv_id_month": arxiv_id_month,  # 2022-03 from ID, always YYYY-MM
        "doi": doi,
        "year": year,
        "title": title[:220],
        "authors": authors,
    }
    if arxiv_stamp:
        out["arxiv_stamp"] = arxiv_stamp
    if arxiv_category:
        out["arxiv_category"] = arxiv_category
    # keep raw stamp for debugging if needed
    if stamp_date and not stamp_iso:
        out["arxiv_stamp_raw"] = stamp_date
    return out


def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(path: Path, value) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def ingest_one(pdf: Path, h: str, ident: dict) -> dict:
    """Read -> extract bibliography -> split. Keeps the split refs for caching."""
    try:
        reader = PdfReader(str(pdf))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return {
            "id": h[:16],
            "sha256": h,
            **ident,
            "error": f"pdf_read:{e}",
            "references": 0,
            "mode": "fail",
            "refs": [],
        }

    m = bib_extract.HEADING.search(text)
    if not m:
        return {
            "id": h[:16],
            "sha256": h,
            **ident,
            "error": "no_references",
            "references": 0,
            "mode": "fail",
            "refs": [],
        }

    tail = text[m.start():]
    stop = bib_extract.STOP.search(tail)
    bib = tail[:stop.start()] if stop else tail
    mode, refs = split_bib(bib)

    return {
        "id": h[:16],
        "sha256": h,
        **ident,
        "references": len(refs),
        "mode": mode,
        "chars": len(bib),
        "refs": refs,  # kept for cache, not thrown away
    }


def reconcile_pdf(papers: dict, ident: dict, h: str) -> tuple[str, str]:
    evidence = make_evidence(
        title=ident.get("title", ""),
        authors=ident.get("authors", []),
        doi=ident.get("doi"),
        arxiv=ident.get("arxiv"),
        arxiv_version=ident.get("version"),
    )
    return reconcile_document(papers, evidence, h)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if "--clear" in argv:
        for p in (MANIFEST, PAPERS, GRAPH, REFS):
            if p.exists():
                p.unlink()
                print(f"cleared {p}")
        if len(argv) == 1:
            return 0

    dirs = [a for a in argv if not a.startswith("-")]
    pdf_dir = Path(dirs[0]) if dirs else Path("docs")
    if not pdf_dir.exists():
        print(f"no such dir: {pdf_dir}")
        return 1

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {pdf_dir}")
        return 0

    manifest = load_json(MANIFEST, {})
    papers = load_json(PAPERS, {})
    graph = load_json(GRAPH, {"nodes": [], "edges": []})
    refs_cache = load_json(REFS, {})  # sha256 -> {mode, refs: [{index, raw}]}
    # backward compat: migrate old string refs to object refs
    refs_migrated = _migrate_refs_cache(refs_cache)
    papers_migrated = _migrate_papers_cache(papers)
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])

    graph_by_hash = {
        node["sha256"]: node
        for node in graph["nodes"]
        if isinstance(node, dict) and node.get("sha256")
    }

    new_cnt = skip_cnt = reconcile_cnt = refs_written = 0
    seen_this_run = set()
    refs_dirty = refs_migrated

    for pdf in pdfs:
        h = sha256_file(pdf)
        if h in seen_this_run:
            print(f"skip  {pdf.name}  {h[:8]}  duplicate bytes in this batch")
            skip_cnt += 1
            continue
        seen_this_run.add(h)

        existing = manifest.get(h)
        needs_reconciliation = not existing or not existing.get("paper_id")

        if existing and not needs_reconciliation:
            # ensure refs are cached even for skips (backfill/migrate) — now with evidence
            if h not in refs_cache:
                ident_tmp = identify_pdf(pdf)
                rec_tmp = ingest_one(pdf, h, ident_tmp)
                refs_cache[h] = {
                    "mode": rec_tmp.get("mode"),
                    "refs": [_new_ref(i, s) for i, s in enumerate(rec_tmp.get("refs", []))],
                }
                refs_written += 1
                refs_dirty = True
                print(
                    f"backfill {pdf.name}  {h[:8]}  refs={rec_tmp.get('references',0):3}  mode={rec_tmp.get('mode')}"
                )
            else:
                # enrich existing refs with deterministic evidence if missing (no rewrite if already enriched)
                entry = refs_cache[h]
                enriched = False
                for r in entry.get("refs", []):
                    if _enrich_ref_with_evidence(r):
                        enriched = True
                if enriched:
                    refs_dirty = True
                    refs_written += 1
                print(
                    f"skip  {pdf.name}  {h[:8]}  already integrated"
                    f"  paper={existing['paper_id']}"
                )
            skip_cnt += 1
            continue

        ident = identify_pdf(pdf)
        paper_id, recon_status = reconcile_pdf(papers, ident, h)

        if existing:
            existing.update({
                **ident,
                "paper_id": paper_id,
                "reconciliation": recon_status,
            })
            node = graph_by_hash.get(h)
            if node:
                node.update({
                    "paper_id": paper_id,
                    "reconciliation": recon_status,
                    "doi": ident.get("doi"),
                    "arxiv": ident.get("arxiv"),
                    "version": ident.get("version"),
                    "authors": ident.get("authors", []),
                })
            # also backfill refs for this reconciled document with evidence
            if h not in refs_cache:
                rec_tmp = ingest_one(pdf, h, ident)
                refs_cache[h] = {
                    "mode": rec_tmp.get("mode"),
                    "refs": [_new_ref(i, s) for i, s in enumerate(rec_tmp.get("refs", []))],
                }
                refs_written += 1
                refs_dirty = True
            else:
                # enrich existing refs in place
                entry = refs_cache[h]
                enriched = False
                for r in entry.get("refs", []):
                    if _enrich_ref_with_evidence(r):
                        enriched = True
                if enriched:
                    refs_dirty = True
                    refs_written += 1
            print(
                f"recon {pdf.name}  {h[:8]}  paper={paper_id}"
                f"  via={recon_status}"
            )
            reconcile_cnt += 1
            continue

        rec = ingest_one(pdf, h, ident)
        rec["paper_id"] = paper_id
        rec["reconciliation"] = recon_status

        # cache the split refs as stable per-reference records (index + raw + deterministic evidence)
        refs_cache[h] = {
            "mode": rec.get("mode"),
            "refs": [_new_ref(i, s) for i, s in enumerate(rec.get("refs", []))],
        }
        refs_written += 1
        refs_dirty = True
        # don't duplicate large refs array in graph node — keep count there
        rec_for_graph = {k: v for k, v in rec.items() if k != "refs"}

        manifest[h] = {
            **ident,
            "sha256": h,
            "id": h[:16],
            "paper_id": paper_id,
            "reconciliation": recon_status,
            "references": rec.get("references", 0),
            "mode": rec.get("mode"),
        }
        graph["nodes"].append(rec_for_graph)
        graph_by_hash[h] = rec_for_graph

        extra = f" | {rec.get('error')}" if rec.get("error") else ""
        print(
            f"new   {pdf.name}  {h[:8]}  paper={paper_id}"
            f"  via={recon_status}  refs={rec.get('references', 0):3}"
            f"  mode={rec.get('mode'):7}  title={ident['title'][:48]!r}{extra}"
        )
        new_cnt += 1

    # --- strong-ID reference resolution + citation edges (no network, no fuzzy) ---
    refs_resolved = 0
    refs_unresolved = 0
    arxiv_resolved = 0
    doi_resolved = 0
    # track if refs/papers/edges were mutated during resolution
    refs_resolve_dirty = False
    papers_before = len(papers)
    # build deduplicated edges from resolved refs
    # resolve each ref
    for doc_sha, entry in refs_cache.items():
        manifest_entry = manifest.get(doc_sha)
        if not manifest_entry or not manifest_entry.get("paper_id"):
            continue
        source_paper = manifest_entry["paper_id"]
        # ensure source paper exists in papers (should, via reconciliation)
        if source_paper not in papers:
            continue
        for ref in entry.get("refs", []):
            paper_id, via, changed = _resolve_ref(papers, ref)
            refs_resolve_dirty |= changed
            if not paper_id:
                refs_unresolved += 1
                continue
            if via == "arxiv":
                arxiv_resolved += 1
            elif via == "doi":
                doi_resolved += 1
            refs_resolved += 1
    # count unresolved (refs without strong ID or without paper_id after attempt)
    if refs_resolved == 0 and refs_unresolved == 0:
        # we counted only resolved in loop; need to count unresolved by scanning
        for entry in refs_cache.values():
            for ref in entry.get("refs", []):
                if ref.get("status") != "resolved":
                    # check if it has strong ID but was not resolved due to missing via? Actually all with doi/arxiv should be resolved now
                    # so remaining unresolved are those without doi/arxiv
                    if "paper_id" not in ref:
                        refs_unresolved += 1
        # adjust double-count: refs_resolved already counted, refs_unresolved now correct
        # but we already incremented unresolved for those without strong ID during loop; need to ensure not double
        # Simpler: recompute totals for reporting
        arxiv_resolved = sum(1 for e in refs_cache.values() for r in e.get("refs", []) if r.get("resolved_via") == "arxiv")
        doi_resolved = sum(1 for e in refs_cache.values() for r in e.get("refs", []) if r.get("resolved_via") == "doi")
        refs_resolved = arxiv_resolved + doi_resolved
        refs_unresolved = sum(1 for e in refs_cache.values() for r in e.get("refs", []) if r.get("status") != "resolved")
    # graph.json is a cache; citation edges come only from current resolved refs.
    pruned_papers = _prune_orphan_papers(papers, refs_cache, manifest)
    edge_set = _build_current_edges(refs_cache, manifest, papers)
    # rebuild graph: nodes are papers (not documents), edges are deduplicated citations
    # deterministically sort for idempotency
    new_nodes = sorted(papers.values(), key=lambda p: p["id"])
    new_edges = sorted([{"from": s, "to": t} for s, t in edge_set], key=lambda e: (e["from"], e["to"]))
    graph_dirty = False
    if graph.get("nodes", []) != new_nodes:
        graph["nodes"] = new_nodes
        graph_dirty = True
    # edges
    existing_edges_set = {(e.get("from") or e.get("source"), e.get("to") or e.get("target")) for e in graph.get("edges", [])}
    if existing_edges_set != edge_set:
        graph["edges"] = new_edges
        graph_dirty = True
    else:
        # ensure sorted order even if same set
        if graph.get("edges", []) != new_edges:
            graph["edges"] = new_edges
            graph_dirty = True

    manifest_dirty = new_cnt > 0 or reconcile_cnt > 0
    if manifest_dirty:
        save_json(MANIFEST, manifest)
    papers_dirty = papers_migrated or len(papers) != papers_before or refs_resolve_dirty or bool(pruned_papers)
    if papers_dirty:
        save_json(PAPERS, papers)
    if graph_dirty:
        save_json(GRAPH, graph)
    elif not GRAPH.exists():
        save_json(GRAPH, graph)
    refs_needs_save = refs_dirty or refs_written or refs_resolve_dirty or refs_migrated
    if refs_needs_save:
        save_json(REFS, refs_cache)

    # recompute totals for report (in case we double-counted earlier)
    total_refs = sum(len(e.get("refs", [])) for e in refs_cache.values())
    arxiv_resolved = sum(1 for e in refs_cache.values() for r in e.get("refs", []) if r.get("resolved_via") == "arxiv")
    doi_resolved = sum(1 for e in refs_cache.values() for r in e.get("refs", []) if r.get("resolved_via") == "doi")
    refs_resolved = arxiv_resolved + doi_resolved
    refs_unresolved = total_refs - refs_resolved

    print(
        f"\ndone: scanned={len(pdfs)} new={new_cnt} reconciled={reconcile_cnt}"
        f" skip={skip_cnt} refs_written={refs_written}"
    )
    print(
        f"refs: total={total_refs}  arxiv={arxiv_resolved}  doi={doi_resolved}  unresolved={refs_unresolved}"
    )
    print(
        f"identity: documents={len(manifest)} papers={len(papers)} (was {papers_before})"
        f"  edges={len(new_edges)}  stores={MANIFEST}, {PAPERS}, {REFS}, {GRAPH}"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
