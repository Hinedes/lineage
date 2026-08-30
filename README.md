# Lineage 2.0

Dead-simple deterministic pipeline: **PDF → bibliography → references → graph**.
No LLM, no Semantic Scholar, no inference beyond the PDF.

## Pipeline
1. ib_extract.py — extract References/Bibliography section (pypdf, heading-anchored, Appendix stop)
2. split_refs.py — split into individual references (bracket [1] vs author-year with year veto)
3. ingest.py — manual refresh: sha256(pdf) dedup + hybrid ID (filename arxiv/version + file title/year), manifest .cache/lineage2.json + graph .cache/graph.json

Re-ingesting same PDF never duplicates (sha256). Unresolvable refs stay unresolved.

## Corpus
docs/*.pdf — 35 research PDFs (git-ignored, ~202MB). Kept locally, not versioned.

## Usage
```bash
python ingest.py            # refresh docs/ (skips already integrated)
python ingest.py --clear    # wipe manifest/graph
python ingest.py docs/      # explicit dir
```
Lazy-correct ID: filename = version truth, file = title truth, hash = identity.

## Store
.cache/lineage2.json — {sha256: {filename, arxiv, version, title, references, mode}}
.cache/graph.json — {"nodes":[...],"edges":[]}

Lineage 1.0 (LLM/S2, web, src) archived in C:\lineage-old with full git history.
