"""Once-per-deck slide digest: turn an image-heavy slide PDF into per-page TEXT
with a cheap vision model, cached as a sidecar file the judges read forever after.

Why this instead of attaching slide images to every judge call: a 27-page deck
at ~1k vision tokens/page costs ~27k extra input tokens *per judge call* — and
#2/#3/#4 would each pay it, on every notebook, every method, every pass@k run.
Transcribing the deck ONCE costs roughly one judge-call's worth of tokens total,
and the marginal cost per evaluation afterwards is zero. The sidecar is also a
testset asset: freeze it with the deck so every method/run is judged against
the identical source text.

Only text-poor pages are transcribed by default (pages whose PyMuPDF text is
already rich keep it for free); pass --all to vision-transcribe every page.

Usage:
  python scripts/eval_harness/slide_digest.py <slides.pdf>
      [--model M] [--min-chars 200] [--dpi 110] [--all] [--force] [--out PATH]

Writes <slides.pdf>.digest.json — picked up automatically by
source_loader.slides_text().
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:  # noqa: BLE001
    pass

_PROMPT = (
    "Transcribe this lecture slide COMPLETELY so a reader with no access to the "
    "image gets all of its content:\n"
    "1. All visible text VERBATIM, in its original language, preserving "
    "structure (titles, bullets, labels, equations in LaTeX-ish notation).\n"
    "2. For every figure/diagram/chart/table, a [FIGURE] block describing what "
    "it shows and the relationship/mechanism it communicates (axes, trends, "
    "arrows, comparisons), factually — do not interpret beyond the slide.\n"
    "Output plain text only. No commentary about the slide's quality."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--model", default=None,
                    help="vision model (default: EVAL_DIGEST_MODEL or the "
                         "planner-family default, e.g. gpt-5.4-mini)")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="pages with at least this much extractable text keep "
                         "it as-is (no API call) unless --all")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--all", action="store_true",
                    help="vision-transcribe every page, even text-rich ones")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the sidecar exists")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or args.pdf.with_name(args.pdf.name + ".digest.json")
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to rebuild).")
        return 0

    import os
    from llm import LLMClient, pick_models
    model = args.model or os.environ.get("EVAL_DIGEST_MODEL")
    if not model:
        planner, judge = pick_models()
        model = planner or judge
    if not model:
        print("no model/key available — set OPENAI_API_KEY / ANTHROPIC_API_KEY "
              "/ GEMINI_API_KEY (or --model).")
        return 2
    client = LLMClient(model)
    print(f"digesting {args.pdf.name} with {model}")

    import fitz  # PyMuPDF
    import tempfile
    doc = fitz.open(str(args.pdf))
    pages: list[dict] = []
    tok_in = tok_out = n_vision = 0
    with tempfile.TemporaryDirectory(prefix="slide_digest_") as td:
        for n in range(1, doc.page_count + 1):
            raw = doc[n - 1].get_text("text").strip()
            if not args.all and len(raw) >= args.min_chars:
                pages.append({"page": n, "text": raw, "method": "pdf_text"})
                print(f"  page {n:>3}: pdf_text ({len(raw)} chars)")
                continue
            png = Path(td) / f"page_{n:03d}.png"
            doc[n - 1].get_pixmap(dpi=args.dpi).save(str(png))
            try:
                res = client.complete(
                    system=_PROMPT,
                    user=f"Slide {n} of {doc.page_count}. Extracted text layer "
                         f"(may be incomplete): {raw[:1000] or '(empty)'}",
                    images=[png], role_tag="slide_digest")
                text = res.text.strip()
                tok_in += res.usage.input_tokens
                tok_out += res.usage.output_tokens
                n_vision += 1
                pages.append({"page": n, "text": text, "method": "vision"})
                print(f"  page {n:>3}: vision ({len(text)} chars)")
            except Exception as e:  # noqa: BLE001
                pages.append({"page": n, "text": raw, "method": "pdf_text",
                              "error": str(e)})
                print(f"  page {n:>3}: vision FAILED ({e}); kept pdf text")
    doc.close()

    out.write_text(json.dumps({
        "source": args.pdf.name,
        "model": model,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dpi": args.dpi,
        "min_chars": args.min_chars,
        "pages": pages,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({n_vision} vision page(s); "
          f"tokens in={tok_in} out={tok_out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
