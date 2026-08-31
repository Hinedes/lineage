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
    reconcile_document,
)
from split_refs import split_bib

ARXIV_FN = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.I)
# vertical side stamp unique to arXiv PDFs — filename-independent, survives renames
# e.g. "arXiv:2203.05482v3  [cs.LG]  1 Jul 2022" (pypdf extracts vertical at page tail)
ARXIV_STAMP = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)\s*\[([^\]]+)\]\s*(\d+\s+\w+\s+\d{4})", re.I)
ARXIV_DATE = re.compile(r"(\d{2})(\d{2})\.\d+")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

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
    return out


def _enrich_ref_with_evidence(ref: dict) -> bool:
    """Add deterministic evidence fields to a {index, raw} ref if missing. Returns True if mutated."""
    raw = ref.get("raw", "")
    parsed = _parse_reference(raw)
    mutated = False
    for k, v in parsed.items():
        if k not in ref:
            ref[k] = v
            mutated = True
    return mutated


def _migrate_refs_cache(refs_cache: dict) -> bool:
    """Migrate old string refs to {index, raw} objects and enrich with evidence. Returns True if mutated."""
    mutated = False
    for entry in refs_cache.values():
        refs = entry.get("refs", [])
        if not refs:
            continue
        if isinstance(refs[0], str):
            entry["refs"] = [{"index": i, "raw": s, **_parse_reference(s)} for i, s in enumerate(refs)]
            mutated = True
        elif isinstance(refs[0], dict):
            # repair index only, preserve every other field (paper_id, status, etc.)
            needs_fix = False
            for i, r in enumerate(refs):
                if r.get("index") != i or "raw" not in r:
                    needs_fix = True
                    break
            if needs_fix:
                for i, r in enumerate(refs):
                    r["index"] = i
                mutated = True
            # enrich old {index, raw} refs with deterministic evidence if missing
            for r in refs:
                if _enrich_ref_with_evidence(r):
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
            if paper.get("doi") == doi_norm:
                matches[pid] = "doi"
    if arxiv_base:
        for pid, paper in papers.items():
            if paper.get("arxiv") == arxiv_base:
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
    arxiv_base, arxiv_version_norm = normalize_arxiv(arxiv_raw) if arxiv_raw else (None, None)
    # version from ref evidence (explicit) takes precedence for storage, but paper identity is base only
    arxiv_version = ref.get("arxiv_version") or arxiv_version_norm

    if not doi_norm and not arxiv_base:
        return None, None

    matches = _find_strong_matches(papers, doi_norm, arxiv_base)
    if len(matches) == 1:
        pid, via = next(iter(matches.items()))
        paper = papers[pid]
        # conflict check: if reference brings an identifier that differs from already-stored value
        if doi_norm and paper.get("doi") and paper["doi"] != doi_norm:
            return None, None
        if arxiv_base and paper.get("arxiv") and paper["arxiv"] != arxiv_base:
            return None, None
        # enrich paper with missing non-conflicting strong identifiers
        if doi_norm and not paper.get("doi"):
            paper["doi"] = doi_norm
        if arxiv_base and not paper.get("arxiv"):
            paper["arxiv"] = arxiv_base
            if arxiv_version and not paper.get("arxiv_version"):
                paper["arxiv_version"] = arxiv_version
        elif arxiv_base and arxiv_version and not paper.get("arxiv_version"):
            # same arXiv base, missing version -> enrich
            paper["arxiv_version"] = arxiv_version
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
        return paper_id, via
    evidence = make_evidence(title="", authors=[], doi=doi_norm, arxiv=arxiv_base, arxiv_version=arxiv_version)
    papers[paper_id] = {
        "id": paper_id,
        "doi": evidence.get("doi"),
        "arxiv": evidence.get("arxiv"),
        "arxiv_version": evidence.get("arxiv_version"),
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
                    "refs": [{"index": i, "raw": s, **_parse_reference(s)} for i, s in enumerate(rec_tmp.get("refs", []))],
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
                    "refs": [{"index": i, "raw": s, **_parse_reference(s)} for i, s in enumerate(rec_tmp.get("refs", []))],
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
            "refs": [{"index": i, "raw": s, **_parse_reference(s)} for i, s in enumerate(rec.get("refs", []))],
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
    edge_set = set()
    # preload existing edges to keep idempotency (if graph already has edges, preserve set)
    for e in graph.get("edges", []):
        # support both {"from","to"} and {"source","target"}
        src = e.get("from") or e.get("source")
        tgt = e.get("to") or e.get("target")
        if src and tgt:
            edge_set.add((src, tgt))
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
            # if already resolved and points to existing paper, count and keep edge
            if ref.get("status") == "resolved" and ref.get("paper_id") and ref.get("paper_id") in papers:
                # ensure edge exists for already-resolved refs (idempotent, no rewrite if already present)
                tgt = ref["paper_id"]
                if (source_paper, tgt) not in edge_set:
                    edge_set.add((source_paper, tgt))
                    refs_resolve_dirty = True
                # count for reporting
                via = ref.get("resolved_via")
                if via == "arxiv":
                    arxiv_resolved += 1
                elif via == "doi":
                    doi_resolved += 1
                refs_resolved += 1
                continue
            # need strong ID
            paper_id, via = _get_or_create_paper_for_ref(papers, ref)
            if not paper_id:
                refs_unresolved += 1
                continue
            # mark ref as resolved — preserve raw/index, add only resolution fields
            if ref.get("paper_id") != paper_id or ref.get("status") != "resolved" or ref.get("resolved_via") != via:
                ref["paper_id"] = paper_id
                ref["status"] = "resolved"
                ref["resolved_via"] = via
                refs_resolve_dirty = True
            # deduplicate edge but keep each ref addressable
            if (source_paper, paper_id) not in edge_set:
                edge_set.add((source_paper, paper_id))
                refs_resolve_dirty = True
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
    # rebuild graph: nodes are papers (not documents), edges are deduplicated citations
    # deterministically sort for idempotency
    new_nodes = sorted(papers.values(), key=lambda p: p["id"])
    new_edges = sorted([{"from": s, "to": t} for s, t in edge_set], key=lambda e: (e["from"], e["to"]))
    graph_dirty = False
    if len(graph.get("nodes", [])) != len(new_nodes) or any(n["id"] not in {x["id"] for x in new_nodes} for n in graph.get("nodes", [])):
        graph["nodes"] = new_nodes
        graph_dirty = True
    else:
        # check if nodes content differs (e.g., new papers added)
        existing_ids = {n["id"] for n in graph.get("nodes", [])}
        new_ids = {n["id"] for n in new_nodes}
        if existing_ids != new_ids:
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
    papers_dirty = len(papers) != papers_before or refs_resolve_dirty
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
