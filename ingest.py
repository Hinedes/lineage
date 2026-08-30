"""Lineage 2.0 ingest — manual refresh, dead-simple.

PDF is source of truth. No LLM, no S2.

  python ingest.py              # refresh docs/ (default)
  python ingest.py --refresh    # same
  python ingest.py docs/        # explicit dir
  python ingest.py --clear      # wipe manifest/graph (re-ingest next run)

ID: lazy-correct hybrid
  - dup check = sha256(pdf_bytes)  => re-ingest same PDF never duplicates
  - arxiv/version = filename  r\"(\\d{4}\\.\\d{4,5})(v\\d+)\"  (33/35 corpus)  else None
  - title       = file: metadata.title or first substantive line of p0 text
  - year/date   = file: arxiv date from filename if present else year in p0

Store: .cache/lineage2.json  (ignored by .gitignore)  +  .cache/graph.json
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

import bib_extract
from split_refs import split_bib

# ponytail: stdlib only, pypdf already required by bib_extract.py

ARXIV_FN = re.compile(r"(\d{4}\.\d{4,5})(v\d+)", re.I)
ARXIV_DATE = re.compile(r"(\d{2})(\d{2})\.\d+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DOI_RE = re.compile(r"10\.\d{4,}/[^\s,;]+")
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.I)

CACHE_DIR = Path(".cache")
MANIFEST = CACHE_DIR / "lineage2.json"
GRAPH = CACHE_DIR / "graph.json"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def identify_pdf(pdf: Path) -> dict:
    """Hybrid: filename for arxiv/version, file for title/year."""
    name = pdf.name
    fn_m = ARXIV_FN.search(name)
    arxiv = fn_m.group(0) if fn_m else None
    version = fn_m.group(2) if fn_m else None
    # date from arxiv YYMM
    arxiv_date = None
    if fn_m:
        d = ARXIV_DATE.search(fn_m.group(1))
        if d:
            yy, mm = int(d.group(1)), int(d.group(2))
            yyyy = 2000 + yy if yy < 80 else 1900 + yy
            arxiv_date = f"{yyyy:04d}-{mm:02d}"

    # file: title + year fallback
    title = ""
    year = None
    try:
        r = PdfReader(str(pdf))
        meta_title = (r.metadata.title or "").strip() if r.metadata else ""
        if meta_title and len(meta_title) >= 8 and "Microsoft Word" not in meta_title:
            title = meta_title
        else:
            # first substantive lines of p0
            txt = (r.pages[0].extract_text() or "") if r.pages else ""
            for line in txt.splitlines():
                t = line.strip()
                if len(t) >= 12 and not t.lower().startswith("arxiv:"):
                    # skip headers like "Published as..."
                    if len(t.split()) >= 3:
                        title = t
                        break
            # fallback to p0 head collapsed
            if not title:
                txt = txt.replace("\n", " ").strip()
                title = txt[:120].split("  ")[0].strip() if txt else name

        # year fallback if no arxiv date
        if not arxiv_date:
            txt = ""
            if r.pages:
                txt = (r.pages[0].extract_text() or "")[:3000]
            m = YEAR_RE.search(txt)
            if m:
                year = m.group(0)
    except Exception:
        if not title:
            title = name

    return {"filename": name, "arxiv": arxiv, "version": version, "arxiv_date": arxiv_date, "year": year, "title": title[:220]}

def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_manifest(m: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

def load_graph() -> dict:
    if GRAPH.exists():
        try:
            return json.loads(GRAPH.read_text(encoding="utf-8"))
        except Exception:
            return {"nodes": [], "edges": []}
    return {"nodes": [], "edges": []}

def save_graph(g: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH.write_text(json.dumps(g, indent=2, ensure_ascii=False), encoding="utf-8")

def ingest_one(pdf: Path, h: str, ident: dict) -> dict:
    """Read → extract → split → minimal resolveattempt → return node record."""
    # 1. extract bibliography (in-memory, don't write .references.txt to docs)
    try:
        reader = PdfReader(str(pdf))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return {"id": h[:16], "sha256": h, **ident, "error": f"pdf_read:{e}", "references": 0, "mode": "fail"}

    m = bib_extract.HEADING.search(text)
    if not m:
        return {"id": h[:16], "sha256": h, **ident, "error": "no_references", "references": 0, "mode": "fail"}
    tail = text[m.start():]
    s = bib_extract.STOP.search(tail)
    bib = tail[:s.start()] if s else tail

    mode, refs = split_bib(bib)

    # minimal ID of cited refs: extract first doi/arxiv if present, else leave unresolved
    resolved = 0
    for r in refs:
        if DOI_RE.search(r) or ARXIV_ID_RE.search(r):
            resolved += 1

    return {"id": h[:16], "sha256": h, **ident, "references": len(refs), "mode": mode, "resolved_hint": resolved, "chars": len(bib)}

def main(argv=None):
    argv = argv or sys.argv[1:]
    if "--clear" in argv:
        for p in [MANIFEST, GRAPH]:
            if p.exists():
                p.unlink()
                print(f"cleared {p}")
        # also clear --clear implies no refresh run; allow --clear --refresh to continue
        if len(argv) == 1:
            return 0

    # dir = first non-flag arg else docs/
    dirs = [a for a in argv if not a.startswith("-")]
    pdf_dir = Path(dirs[0]) if dirs else Path("docs")
    if not pdf_dir.exists():
        print(f"no such dir: {pdf_dir}")
        return 1

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {pdf_dir}")
        return 0

    manifest = load_manifest()
    graph = load_graph()
    # index graph nodes by sha256 for dedup
    seen_hashes = set(manifest.keys())
    graph_ids = {n["sha256"] for n in graph.get("nodes", []) if "sha256" in n}

    new_cnt = skip_cnt = 0
    # ponytail: track hashes seen in this run too, so duplicate file (1) is skipped intra-batch
    seen_this_run = set()
    for pdf in pdfs:
        h = sha256_file(pdf)
        if h in seen_hashes or h in graph_ids or h in seen_this_run:
            print(f"skip  {pdf.name}  {h[:8]}  already integrated")
            skip_cnt += 1
            continue
        seen_this_run.add(h)
        ident = identify_pdf(pdf)
        rec = ingest_one(pdf, h, ident)
        # store
        manifest[h] = {**ident, "sha256": h, "id": h[:16], "references": rec.get("references", 0), "mode": rec.get("mode")}
        graph.setdefault("nodes", []).append(rec)
        # edges: host -> cited (stub: one edge per reference that has an id hint)
        # for now, edges are implicit via reference count; real cited-node resolution is next step
        # we record a summary edge count
        graph.setdefault("edges", [])
        # ponytail: no per-reference node creation yet; unresolved refs stay as count, not guessed

        extra = f" | {rec.get('error')}" if rec.get("error") else ""
        print(f"new   {pdf.name}  {h[:8]}  arxiv={ident['arxiv'] or '-':15}  refs={rec.get('references',0):3}  mode={rec.get('mode'):7}  title={ident['title'][:55]!r}{extra}")
        new_cnt += 1

    save_manifest(manifest)
    save_graph(graph)
    print(f"\ndone: scanned={len(pdfs)} new={new_cnt} skip={skip_cnt}  manifest={MANIFEST}  graph={GRAPH}")
    # also show graph summary
    total_refs = sum(n.get("references", 0) for n in graph.get("nodes", []))
    print(f"graph: nodes={len(graph.get('nodes',[]))}  total_references={total_refs}")
    return 0

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
