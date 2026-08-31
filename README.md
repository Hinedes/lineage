# Lineage 2.0

Dead-simple deterministic pipeline: **PDF → bibliography → references → graph**.
No LLM, no Semantic Scholar, no inference beyond the PDF.

## Pipeline
1. `bib_extract.py` — extract References/Bibliography section (pypdf, heading-anchored, Appendix stop)
2. `split_refs.py` — split into individual references (bracket [1] vs author-year with year veto)
3. `ingest.py` — manual refresh, exact-document dedup, and paper reconciliation

## Identity
A PDF file and a paper are not the same identity.

`sha256(pdf bytes)` identifies one exact PDF document. Renaming the file does not change that identity; a different PDF edition/version may have different bytes while still representing the same paper.

Paper reconciliation is deterministic and conservative:
1. exact DOI
2. exact arXiv base ID, with `v1`, `v2`, etc. treated as versions of the same paper
3. exact normalized title + compatible ordered author list
4. otherwise create a separate paper record

Title normalization erases presentation noise such as case, commas, and whitespace, while preserving technical punctuation that can carry meaning: `GPT-4` is not normalized into `GPT 4`.

Author matching requires surnames to agree in order. A full first name may match its own initial (`John Smith` ↔ `J. Smith`), but two different full first names conflict (`John Smith` ≠ `Joseph Smith`). Strong DOI/arXiv conflicts are never merged by the title/author fallback.

The PDF text itself is preferred for arXiv identity, including the selectable rotated `arXiv:...` margin marker. Filename arXiv IDs are only a fallback.

Re-ingesting the exact same PDF never duplicates the document. Multiple PDF versions can reconcile to one paper.

## Corpus
`docs/*.pdf` — 35 research PDFs (git-ignored, ~202MB). Kept locally, not versioned.

## Usage
```bash
python ingest.py            # refresh docs/
python ingest.py --clear    # wipe local stores
python ingest.py docs/      # explicit dir
python -m unittest discover -s tests
```

## Store
`.cache/lineage2.json` — exact PDF documents and their `paper_id`

`.cache/papers.json` — reconciled paper records and the document hashes attached to each paper

`.cache/graph.json` — current graph skeleton; reference-to-paper resolution and edges are intentionally not part of the reconciliation step

Lineage 1.0 (LLM/S2, web, src) is archived in `C:\lineage-old` with full git history.
