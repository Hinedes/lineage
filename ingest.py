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
    """Read -> extract bibliography -> split. Reference resolution is intentionally absent."""
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
        for p in (MANIFEST, PAPERS, GRAPH):
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
    graph.setdefault("nodes", [])
    graph.setdefault("edges", [])

    graph_by_hash = {
        node["sha256"]: node
        for node in graph["nodes"]
        if isinstance(node, dict) and node.get("sha256")
    }

    new_cnt = skip_cnt = reconcile_cnt = 0
    seen_this_run = set()

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
            print(
                f"recon {pdf.name}  {h[:8]}  paper={paper_id}"
                f"  via={recon_status}"
            )
            reconcile_cnt += 1
            continue

        rec = ingest_one(pdf, h, ident)
        rec["paper_id"] = paper_id
        rec["reconciliation"] = recon_status

        manifest[h] = {
            **ident,
            "sha256": h,
            "id": h[:16],
            "paper_id": paper_id,
            "reconciliation": recon_status,
            "references": rec.get("references", 0),
            "mode": rec.get("mode"),
        }
        graph["nodes"].append(rec)
        graph_by_hash[h] = rec

        extra = f" | {rec.get('error')}" if rec.get("error") else ""
        print(
            f"new   {pdf.name}  {h[:8]}  paper={paper_id}"
            f"  via={recon_status}  refs={rec.get('references', 0):3}"
            f"  mode={rec.get('mode'):7}  title={ident['title'][:48]!r}{extra}"
        )
        new_cnt += 1

    save_json(MANIFEST, manifest)
    save_json(PAPERS, papers)
    save_json(GRAPH, graph)

    print(
        f"\ndone: scanned={len(pdfs)} new={new_cnt} reconciled={reconcile_cnt}"
        f" skip={skip_cnt}"
    )
    print(
        f"identity: documents={len(manifest)} papers={len(papers)}"
        f"  stores={MANIFEST}, {PAPERS}"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
