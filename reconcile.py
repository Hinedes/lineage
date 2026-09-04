"""Deterministic paper reconciliation for Lineage 2.0.

Document identity and paper identity are deliberately separate:
- sha256(pdf bytes) identifies an exact document
- reconciliation identifies the paper represented by that document

Automatic paper matches use, in order:
1. exact DOI
2. exact arXiv base ID (version stripped)
3. exact normalized title + compatible ordered authors

No fuzzy title matching, edit distance, embeddings, network lookup, or LLM.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
ARXIV_RE = re.compile(
    r"(?:arxiv\s*:\s*)?"
    r"(?P<base>\d{4}\.\d{4,5}|[a-z][a-z0-9.-]+/\d{7})"
    r"(?P<version>v\d+)?",
    re.I,
)
ARXIV_WHITESPACE_RE = re.compile(
    r"https?://arxiv\.org/(?:abs|pdf)/"
    r"(?P<year>\d{4})\.\s+(?P<number>\d{4,5})(?P<version>v\d+)?",
    re.I,
)

_DASHES = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2212": "-",
})
_NOISE_PUNCT = re.compile(r"[,;:!?()[\]{}\"'“”‘’]")
_TRAILING_DOT = re.compile(r"\.(?=\s|$)")
_WS = re.compile(r"\s+")
_AFFILIATION_MARKERS = (
    "university", "institute", "department", "laboratory", "lab ",
    "school of", "college", "research center", "research centre",
    ".edu", "@", "corresponding author",
)
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def normalize_title(title: str) -> str:
    """Erase presentation noise while preserving meaningful technical punctuation."""
    s = unicodedata.normalize("NFKC", title or "").translate(_DASHES).casefold()
    s = _NOISE_PUNCT.sub(" ", s)
    s = _TRAILING_DOT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    s = unicodedata.normalize("NFKC", value).strip().casefold()
    s = re.sub(r"^\s*(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", s, flags=re.I)
    m = DOI_RE.search(s)
    return m.group(0).rstrip(".,;:)]}") if m else None


def extract_doi(text: str) -> str | None:
    m = DOI_RE.search(text or "")
    return normalize_doi(m.group(0)) if m else None


def normalize_arxiv(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    m = ARXIV_RE.search(unicodedata.normalize("NFKC", value))
    if not m:
        return None, None
    base = m.group("base").casefold()
    version = m.group("version")
    return base, version.casefold() if version else None


def extract_arxiv(
    text: str, *, allow_whitespace_repair: bool = True
) -> tuple[str | None, str | None]:
    """Read an explicit arXiv marker from PDF text, including rotated margin text."""
    m = re.search(
        r"arxiv\s*:\s*(\d{4}\.\d{4,5}|[a-z][a-z0-9.-]+/\d{7})(v\d+)?",
        text or "",
        re.I,
    )
    if not m and allow_whitespace_repair:
        m_ws = ARXIV_WHITESPACE_RE.search(text or "")
        if not m_ws:
            return None, None
        return (
            f"{m_ws.group('year')}.{m_ws.group('number')}".casefold(),
            m_ws.group("version").casefold() if m_ws.group("version") else None,
        )
    if not m:
        return None, None
    return m.group(1).casefold(), m.group(2).casefold() if m.group(2) else None


def _name_words(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", s or "").translate(_DASHES).casefold()
    s = re.sub(r"[\d*†‡§]+", "", s).replace(".", " ")
    s = re.sub(r"[^a-z0-9\u00c0-\u024f\u1e00-\u1eff'\- ]+", " ", s)
    return [w for w in _WS.split(s.strip()) if w]


def normalize_author(author: str) -> dict[str, str] | None:
    """Keep surname + first given name/initial; do not pretend initials are unique."""
    raw = (author or "").strip()
    if not raw or raw.casefold() in {"et al", "et al.", "and"}:
        return None

    if raw.count(",") == 1:
        left, right = [x.strip() for x in raw.split(",", 1)]
        left_words, right_words = _name_words(left), _name_words(right)
        if len(left_words) == 1 and right_words:
            given = right_words[0]
            return {"surname": left_words[0], "given": given, "initial": given[0]}

    words = _name_words(raw)
    if len(words) < 2:
        return None
    if words[-1] in _SUFFIXES and len(words) >= 3:
        words = words[:-1]
    given, surname = words[0], words[-1]
    return {"surname": surname, "given": given, "initial": given[0]}


def parse_author_list(raw: str | Iterable[str] | None) -> list[dict[str, str]]:
    if not raw:
        return []
    if not isinstance(raw, str):
        out = []
        for item in raw:
            author = normalize_author(str(item))
            if author:
                out.append(author)
        return out

    s = unicodedata.normalize("NFKC", raw).strip()
    if re.search(r";|\s+\band\b\s+|[·•]", s, re.I):
        parts = re.split(r"\s*;\s*|\s+\band\b\s+|\s*[·•]\s*", s, flags=re.I)
    else:
        parts = [p.strip() for p in s.split(",")]

    out = []
    for part in parts:
        author = normalize_author(part)
        if author:
            out.append(author)
    return out


def author_compatible(a: dict[str, str], b: dict[str, str]) -> bool:
    if a["surname"] != b["surname"]:
        return False
    ga, gb = a["given"], b["given"]
    if len(ga) > 1 and len(gb) > 1:
        return ga == gb
    return a["initial"] == b["initial"]


def authors_compatible(a: list[dict[str, str]], b: list[dict[str, str]]) -> bool:
    return bool(a and b and len(a) == len(b) and all(author_compatible(x, y) for x, y in zip(a, b)))


def authors_key(authors: list[dict[str, str]]) -> str:
    return "|".join(f"{a['initial']}:{a['surname']}:{a['given']}" for a in authors)


def extract_authors_from_front_page(text: str, title: str = "") -> list[str]:
    """Conservative fallback when PDF metadata has no author field."""
    lines = [_WS.sub(" ", line).strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    title_norm = normalize_title(title)
    start = 0
    if title_norm:
        for i, line in enumerate(lines[:30]):
            n = normalize_title(line)
            if n and (n == title_norm or n in title_norm or title_norm in n):
                start = i + 1
                break

    for line in lines[start:start + 12]:
        low = line.casefold()
        if low.startswith("abstract") or low.startswith("introduction"):
            break
        if any(marker in low for marker in _AFFILIATION_MARKERS):
            continue
        parsed = parse_author_list(line)
        if len(parsed) >= 2:
            return [
                p.strip()
                for p in re.split(r"\s*,\s*|\s+\band\b\s+|\s*;\s*|\s*[·•]\s*", line, flags=re.I)
                if p.strip()
            ]
    return []


def make_evidence(*, title="", authors=None, doi=None, arxiv=None, arxiv_version=None) -> dict:
    base, parsed_version = normalize_arxiv(arxiv)
    author_items = list(authors) if authors is not None and not isinstance(authors, str) else authors
    return {
        "title": title or "",
        "title_norm": normalize_title(title),
        "authors": author_items or [],
        "authors_norm": parse_author_list(author_items),
        "doi": normalize_doi(doi),
        "arxiv": base,
        "arxiv_version": arxiv_version or parsed_version,
    }


def _strong_conflict(paper: dict, evidence: dict) -> bool:
    return bool(
        paper.get("doi") and evidence.get("doi") and paper["doi"] != evidence["doi"]
        or paper.get("arxiv") and evidence.get("arxiv") and paper["arxiv"] != evidence["arxiv"]
    )


def _fallback_match(paper: dict, evidence: dict) -> bool:
    if _strong_conflict(paper, evidence):
        return False
    if not evidence.get("title_norm") or paper.get("title_norm") != evidence["title_norm"]:
        return False
    return authors_compatible(paper.get("authors_norm", []), evidence.get("authors_norm", []))


def _new_paper_id(evidence: dict, document_sha256: str) -> str:
    if evidence.get("arxiv"):
        return f"arxiv:{evidence['arxiv']}"
    if evidence.get("doi"):
        return f"doi:{evidence['doi']}"
    material = evidence.get("title_norm", "") + "\0" + authors_key(evidence.get("authors_norm", []))
    if material.strip("\0"):
        return "paper:" + hashlib.sha256(material.encode()).hexdigest()[:16]
    return "paper:document:" + document_sha256[:16]


def _enrich_paper(paper: dict, evidence: dict, document_sha256: str) -> None:
    if document_sha256 not in paper.setdefault("documents", []):
        paper["documents"].append(document_sha256)
    for field in ("doi", "arxiv", "title", "title_norm"):
        if not paper.get(field) and evidence.get(field):
            paper[field] = evidence[field]
    if not paper.get("authors_norm") and evidence.get("authors_norm"):
        paper["authors"] = evidence.get("authors", [])
        paper["authors_norm"] = evidence["authors_norm"]
    elif evidence.get("authors_norm") and authors_compatible(paper.get("authors_norm", []), evidence["authors_norm"]):
        old_score = sum(len(a["given"]) > 1 for a in paper["authors_norm"])
        new_score = sum(len(a["given"]) > 1 for a in evidence["authors_norm"])
        if new_score > old_score:
            paper["authors"] = evidence.get("authors", [])
            paper["authors_norm"] = evidence["authors_norm"]


def _create_conflict(papers: dict, evidence: dict, document_sha256: str, candidates: list[str]):
    pid = "paper:conflict:" + document_sha256[:16]
    papers[pid] = {
        "id": pid,
        "doi": evidence.get("doi"),
        "arxiv": evidence.get("arxiv"),
        "title": evidence.get("title", ""),
        "title_norm": evidence.get("title_norm", ""),
        "authors": evidence.get("authors", []),
        "authors_norm": evidence.get("authors_norm", []),
        "documents": [document_sha256],
        "reconciliation_conflict": candidates,
    }
    return pid, "conflict"


def reconcile_document(papers: dict[str, dict], evidence: dict, document_sha256: str) -> tuple[str, str]:
    """Return (paper_id, status), mutating the paper registry in place."""
    strong = {}
    if evidence.get("doi"):
        for pid, paper in papers.items():
            if paper.get("doi") == evidence["doi"]:
                strong[pid] = "matched-doi"
    if evidence.get("arxiv"):
        for pid, paper in papers.items():
            if paper.get("arxiv") == evidence["arxiv"]:
                strong[pid] = "matched-arxiv"

    if len(strong) == 1:
        pid, status = next(iter(strong.items()))
        if _strong_conflict(papers[pid], evidence):
            return _create_conflict(papers, evidence, document_sha256, [pid])
        _enrich_paper(papers[pid], evidence, document_sha256)
        return pid, status
    if len(strong) > 1:
        return _create_conflict(papers, evidence, document_sha256, sorted(strong))

    fallback = [pid for pid, paper in papers.items() if _fallback_match(paper, evidence)]
    if len(fallback) == 1:
        pid = fallback[0]
        _enrich_paper(papers[pid], evidence, document_sha256)
        return pid, "matched-title-authors"
    if len(fallback) > 1:
        return _create_conflict(papers, evidence, document_sha256, sorted(fallback))

    pid = _new_paper_id(evidence, document_sha256)
    if pid in papers:
        if _strong_conflict(papers[pid], evidence):
            return _create_conflict(papers, evidence, document_sha256, [pid])
        _enrich_paper(papers[pid], evidence, document_sha256)
        return pid, "matched-title-authors"

    papers[pid] = {
        "id": pid,
        "doi": evidence.get("doi"),
        "arxiv": evidence.get("arxiv"),
        "title": evidence.get("title", ""),
        "title_norm": evidence.get("title_norm", ""),
        "authors": evidence.get("authors", []),
        "authors_norm": evidence.get("authors_norm", []),
        "documents": [document_sha256],
    }
    return pid, "new"
