# Research

This directory contains the verified literature trail for this project,
managed with [littrail](https://github.com/littrail/littrail).

## Workflow

```
generated report -> verified catalog -> paper notes -> issue candidates -> GitHub issues
```

1. Save AI-generated literature surveys in `reports/`.
2. Validate paper metadata against OpenAlex and record in `catalog.yaml`.
3. Write reading notes for papers worth studying in `notes/`.
4. Organise implementation or experiment ideas in `ideas/`.
5. Promote agreed candidates to GitHub issues.

## Directory Structure

| Path | Purpose | Git tracked |
|------|---------|-------------|
| `reports/` | AI-generated literature surveys (starting points only) | Yes |
| `catalog.yaml` | Verified paper metadata indexed by stable identifiers | Yes |
| `notes/` | Per-paper reading notes (`<key>.md`) | Yes |
| `ideas/` | Issue candidates and experiment proposals | Yes |
| `pdfs/` | Local PDF cache | **No** (gitignored) |

## Tracked Artifacts vs. Ignored PDF Cache

Markdown reports, notes, ideas, and `catalog.yaml` are Git-tracked.
PDF binaries are stored in `pdfs/` and excluded from version control via `.gitignore`.
This keeps the repository lightweight while preserving the human-readable record.

## Stable Identifiers

Always record at least one stable identifier per paper:

- **DOI** — preferred for published work
- **OpenAlex ID** — `W<number>`, always available via OpenAlex
- **arXiv ID** — `arXiv:YYMM.NNNNN` for preprints
- **ACL Anthology ID** — for ACL/EMNLP/NAACL papers

AI-generated reports may contain fabricated citations.
Do not treat a report as evidence until you have confirmed the metadata
and read the primary source.

## littrail Commands

```bash
# Initialise research/ in a new project
littrail init

# Add a paper by DOI or OpenAlex ID
littrail add-paper --doi 10.18653/v1/D19-1404
littrail add-paper --openalex W2970200208

# Verify catalog metadata against OpenAlex (read-only)
littrail verify

# Check repository consistency offline
littrail check
```

### Searching for papers

```bash
# Human-readable table
littrail search "retrieval augmented generation"

# Machine-readable JSON for agent pipelines
littrail search "retrieval augmented generation" --json --limit 20 \
  | jq -r '.[].openalex_id' \
  | xargs -I{} littrail add-paper --openalex {}
```

Search results are unverified candidates.
Run `littrail verify` after adding papers to confirm metadata against OpenAlex.
