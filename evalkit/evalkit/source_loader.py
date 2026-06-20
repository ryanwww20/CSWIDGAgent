"""Load the source materials a method was given — slide PDFs (text, optionally
page images) and transcripts — for the faithfulness / topic judges.

Slides: if a sidecar digest exists (`<slides>.pdf.digest.json`, produced once
per deck by `slide_digest.py` with a vision model), its per-page transcription
is used instead of raw PyMuPDF text — image-heavy decks (most lecture slides)
extract almost no text otherwise. All caps are applied with VISIBLE truncation
markers (textbudget.py), never silent cuts.

Transcripts: when over budget, a relevance-selected WINDOW (character-bigram
match against the demo's declared concept) is returned instead of a blind
head-cut — the lecture's first minutes are usually not the part that covers
the concept.
"""
from __future__ import annotations

import json
from pathlib import Path

from textbudget import truncate


def _digest_sidecar(pdf_path: Path) -> Path:
    return pdf_path.with_name(pdf_path.name + ".digest.json")


def slides_text(pdf_path: str | Path, max_chars: int = 24000) -> str:
    """Per-slide text: sidecar vision digest when available, else PyMuPDF."""
    pdf_path = Path(pdf_path)
    sidecar = _digest_sidecar(pdf_path)
    if sidecar.exists():
        try:
            d = json.loads(sidecar.read_text(encoding="utf-8"))
            parts = [f"[slide {p['page']}]\n{p['text'].strip()}"
                     for p in d.get("pages", []) if p.get("text", "").strip()]
            if parts:
                return truncate("\n\n".join(parts), max_chars, "the slide deck")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: unreadable slide digest {sidecar.name}: {e}; "
                  f"falling back to raw PDF text.")
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    parts: list[str] = []
    total_chars = 0
    for n, page in enumerate(doc, 1):
        t = page.get_text("text").strip()
        total_chars += len(t)
        if t:
            parts.append(f"[slide {n}]\n{t}")
    page_count = doc.page_count
    doc.close()
    if page_count and total_chars / page_count < 200:
        print(f"WARNING: image-heavy deck ({total_chars} extractable chars over "
              f"{page_count} pages) — judges will see almost no slide content. "
              f"Build a vision digest once per deck:\n"
              f"  python scripts/eval_harness/slide_digest.py {pdf_path}")
    return truncate("\n\n".join(parts), max_chars, "the slide deck")


def slide_images(pdf_path: str | Path, out_dir: str | Path,
                 pages: list[int] | None = None, dpi: int = 110) -> list[str]:
    """Render selected slide pages to PNGs (1-indexed). Optional vision context."""
    import fitz
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    sel = pages or list(range(1, doc.page_count + 1))
    paths: list[str] = []
    for n in sel:
        if 1 <= n <= doc.page_count:
            pix = doc[n - 1].get_pixmap(dpi=dpi)
            p = out_dir / f"slide_{n:03d}.png"
            pix.save(str(p))
            paths.append(str(p))
    doc.close()
    return paths


def _bigrams(text: str) -> set[str]:
    t = "".join(text.lower().split())
    return {t[i:i + 2] for i in range(len(t) - 1)}


def transcript_text(path: str | Path, max_chars: int = 16000,
                    query: str = "") -> str:
    """Transcript for judge grounding. Over budget + a concept query: return the
    most relevant contiguous window (character-bigram overlap — works for both
    Chinese and English) with a visible marker; otherwise a marked head-cut."""
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= max_chars:
        return text
    if not query.strip():
        return truncate(text, max_chars, "the transcript")
    qg = _bigrams(query)
    if not qg:
        return truncate(text, max_chars, "the transcript")
    step = max(1, max_chars // 4)
    best_i, best_s = 0, -1.0
    for i in range(0, max(1, len(text) - max_chars + 1), step):
        win = text[i:i + max_chars]
        wg = _bigrams(win)
        s = len(qg & wg) / len(qg)
        if s > best_s:
            best_i, best_s = i, s
    sel = text[best_i:best_i + max_chars]
    marker = (f"[TRANSCRIPT WINDOW — showing chars {best_i:,}–"
              f"{best_i + len(sel):,} of {len(text):,}, selected for relevance "
              f"to the demo's concept; the rest was NOT shown to you.]\n\n")
    return marker + sel
