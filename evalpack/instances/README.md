# Evaluation instances

One **instance** = the source materials one task is generated from and judged
against: a slide deck (PDF), optionally a lecture transcript, plus metadata.

## Two tiers

### 1. `starter/` — self-contained, ready to use
Three decks shipped inside the pack (these are the decks the evaluation
pipeline was developed and validated on). See `starter/manifest.jsonl`.

`KVcahce.pdf` ships with its prebuilt **vision slide digest**
(`KVcahce.pdf.digest.json`) — `evalkit` picks the sidecar up automatically, so
judges see full per-slide content (verbatim text + figure descriptions) even
though the deck itself is image-heavy. For the other decks, build the digest
once before quality judging (a few cents, cached forever):

```bash
python evalkit/slide_digest.py "instances/starter/agent_era (v9).pdf"
```

### 2. `manifest.jsonl` — the full curated test set (276 lectures, data not bundled)
The manifest carries all metadata (id, course, instructor, language, topic,
topic family, difficulty, demo-ability, page counts, transcript availability +
quality, sha256 hashes, source URLs). The **raw data (~800 MB) is not copied
into the pack** — for licensing posture and size, the pack releases pointers +
hashes and you keep/fetch the raw files separately.

Each manifest row's `rel_path` (e.g. `testset_trimmed/lectures/HTLIN_ML__05_handout`)
resolves against a **data root**:

- If you have the `ml_colab` repo, the data root is the repo root (the pack's
  parent) — this is the default.
- Otherwise copy or symlink the `testset_trimmed/` directory next to the pack
  and pass `--data-root` accordingly.

Inside each lecture directory: `slides_full.pdf`, `transcript.<lang>.txt`
(when available), `meta.json`, `source.json`.

## Using an instance

```bash
# by instance id (resolves slides + transcript from the manifest):
python evalkit/run_eval.py my_demo.ipynb --instance HTLIN_ML__05_handout \
    --data-root /path/to/ml_colab --method my_method

# or explicitly:
python evalkit/run_eval.py my_demo.ipynb \
    --slides instances/starter/KVcahce.pdf --method my_method
```

## Rules that keep the eval fair

- **Whole deck as input** (up to ~30 slides): do not cut decks into
  single-concept slices — see `../TASK_DEFINITION.md`.
- **Frozen digests**: when a deck has a `.digest.json`, every method/run is
  judged against that identical transcription. Commit digests with the deck.
- **Contamination**: courses that ship official per-concept demo notebooks are
  not clean black-box test inputs (a web-search-enabled method could retrieve
  the official demo). Use those only as calibration/reference exemplars.
- **Do not redistribute** copyrighted slide PDFs/transcripts outside the team;
  the manifest + hashes + URLs are the shareable layer.
