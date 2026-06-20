"""Quality-metric orchestrator (headline metrics #2, #3, #4, #7).

Runs the code-aware judges over a notebook + its source materials:
  #2 Faithfulness & Correctness (2a/2b/2c + verifier)
  #3 Pedagogical Depth (marginal value over the slide)
  #4 Topic-Worthiness
  #7 Exposition / Clarity (7a visual / 7b textual / 7c code-explanation, harmonic)

LLM stages are skipped gracefully with no key (--no-llm). Run Success (#1),
Interactivity (#5) and Efficiency (#6) are produced by other tools
(eval_notebook.py / run_interactivity.py).

Usage:
  python scripts/eval_harness/run_quality_eval.py <notebook.ipynb> \
      --slides <slides.pdf> [--transcript <t.txt>] [--frames <frames_dir>] \
      [--out DIR] [--no-llm] [--judge-model M] [--verifier-model M]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:  # noqa: BLE001
    pass

from schemas import QualityReport, harmonic_mean, to_json  # noqa: E402
import notebook_digest  # noqa: E402
import source_loader  # noqa: E402


def _sampled(fn, k, agg_fields, *args, composite=None, **kwargs):
    """Self-consistency: run judge `fn` k times and aggregate to lower run-to-run
    variance — take the MEDIAN of each field in `agg_fields`, recompute `composite`
    (name, fn-of-output) from the medians, and return the run whose primary (first)
    field is closest to the median, with its fields overwritten by the medians.
    k<=1 is a single call (unchanged behaviour). The non-numeric fields (rationale,
    errors, citations) come from that representative run, so they stay consistent
    with the reported scores."""
    if k <= 1:
        return fn(*args, **kwargs)
    runs = [fn(*args, **kwargs) for _ in range(k)]
    outs = [o for o, _ in runs]
    med = {f: round(statistics.median(float(getattr(o, f)) for o in outs), 2)
           for f in agg_fields}
    prim = agg_fields[0]
    rep = min(outs, key=lambda o: abs(float(getattr(o, prim)) - med[prim]))
    for f in agg_fields:
        setattr(rep, f, med[f])
    if composite:
        name, cfn = composite
        setattr(rep, name, cfn(rep))
    info = dict(runs[0][1])
    info["self_consistency_k"] = k
    info["raw_scores"] = {f: [float(getattr(o, f)) for o in outs] for f in agg_fields}
    return rep, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--executed", type=Path, default=None,
                    help="executed notebook from stage A (eval_run/<id>/executed.ipynb); "
                         "digested instead of the raw notebook so judges see the real "
                         "stored outputs")
    ap.add_argument("--slides", type=Path, default=None, help="source slides PDF")
    ap.add_argument("--transcript", type=Path, default=None)
    ap.add_argument("--frames", type=Path, default=None,
                    help="extra rendered-output PNGs (e.g. interactivity_eval/frames)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--verifier-model", default=None)
    ap.add_argument("--max-images", type=int, default=6)
    ap.add_argument("--judge-samples", type=int,
                    default=int(os.environ.get("EVAL_JUDGE_SAMPLES", "1")),
                    help="run each quality judge K times and take the MEDIAN of its "
                         "sub-scores (self-consistency; lowers run-to-run variance). "
                         "1 = single call (default); 3 recommended for stable scores. "
                         "Also settable via EVAL_JUDGE_SAMPLES.")
    ap.add_argument("--judge-temperature", type=float,
                    default=float(os.environ.get("EVAL_JUDGE_TEMPERATURE", "0.0")),
                    help="sampling temperature for the judges (default 0.0 for "
                         "low variance; reasoning models ignore it).")
    args = ap.parse_args()

    out = args.out or args.notebook.parent / "quality_eval"
    out.mkdir(parents=True, exist_ok=True)

    src_nb = args.notebook
    if args.executed:
        if args.executed.exists():
            src_nb = args.executed
        else:
            print(f"WARNING: --executed {args.executed} not found; "
                  f"digesting {args.notebook} instead.")
    nb_text, nb_imgs = notebook_digest.digest(src_nb, out / "nb_images",
                                              budget=58000)
    # The opening cells (declared concept + early markdown) describe what the
    # demo is about — used to select the relevant transcript window.
    concept_query = nb_text[:2000]
    if not nb_imgs and not args.executed:
        print("WARNING: the raw notebook stores no image outputs — judges will "
              "only see interaction frames. Pass stage A's executed notebook "
              "via --executed so they see the real rendered outputs.")

    # Evidence images = stored notebook outputs first, then interaction frames
    # to fill the remaining slots (baseline frame first, then an even spread).
    evidence: list[tuple[str, str]] = list(nb_imgs)
    if args.frames and args.frames.exists():
        frames = sorted(p for p in args.frames.glob("*.png") if p.is_file())
        base = [p for p in frames if "baseline" in p.name]
        rest = [p for p in frames if "baseline" not in p.name]
        room = max(0, args.max_images - len(evidence) - len(base))
        if not room:
            rest = []
        elif rest and room < len(rest):
            idx = [round(j * (len(rest) - 1) / (room - 1)) if room > 1 else 0
                   for j in range(room)]
            rest = [rest[i] for i in sorted(set(idx))]
        for p in base:
            evidence.append((str(p),
                             f"interaction frame {p.stem} (initial/default view)"))
        for p in rest:
            evidence.append((str(p),
                             f"interaction frame {p.stem} (controls at swept "
                             f"values — NOT the default view a student first sees)"))
    evidence = evidence[:args.max_images]

    # Burn a self-identifying 'IMAGE k | origin' banner onto each evidence image
    # and prepend a citation manifest, so judge citations are checkable instead
    # of guessed from attachment order.
    from imaging import label_image
    nb_images: list[str] = []
    if evidence:
        lab_dir = out / "judge_images"
        lab_dir.mkdir(exist_ok=True)
        manifest_rows = []
        for k, (p, origin) in enumerate(evidence, 1):
            dst = lab_dir / f"image{k:02d}.png"
            nb_images.append(label_image(p, dst, f"IMAGE {k} | {origin}"))
            manifest_rows.append(f"  IMAGE {k} = {origin}")
        nb_text = ("ATTACHED IMAGES — each attached image has a black banner "
                   "reading 'IMAGE k | <origin>'. These are the ONLY images; "
                   "cite them as 'IMAGE k':\n"
                   + "\n".join(manifest_rows) + "\n\n" + nb_text)

    slides = source_loader.slides_text(args.slides) if args.slides else ""
    transcript = (source_loader.transcript_text(args.transcript,
                                                query=concept_query)
                  if args.transcript else "")
    from textbudget import is_truncated
    trunc = {k: is_truncated(v) for k, v in
             (("notebook", nb_text), ("slides", slides),
              ("transcript", transcript)) if v}
    print(f"digest: notebook={len(nb_text)} chars ({src_nb.name}), "
          f"{len(nb_images)} image(s); "
          f"slides={len(slides)} chars; transcript={len(transcript)} chars; "
          f"truncated={[k for k, v in trunc.items() if v] or 'nothing'}")

    if args.no_llm:
        print("--no-llm: skipping judges.")
        return 0

    from llm import pick_models, pick_verifier, LLMClient
    _, default_judge = pick_models()
    judge_model = args.judge_model or default_judge
    if not judge_model:
        print("no judge model/key available; skipping judges.")
        return 2
    verifier_model = args.verifier_model or pick_verifier(judge_model)
    print(f"judge={judge_model}  verifier={verifier_model}  "
          f"temperature={args.judge_temperature}  samples={args.judge_samples}")

    judge = LLMClient(judge_model, temperature=args.judge_temperature)
    verifier = (LLMClient(verifier_model, temperature=args.judge_temperature)
                if verifier_model else None)
    K = max(1, args.judge_samples)

    report = QualityReport(notebook=str(args.notebook),
                           meta={"judge_model": judge_model,
                                 "verifier_model": verifier_model,
                                 "digested_notebook": str(src_nb),
                                 "evidence_images": len(nb_images),
                                 "chars": {"notebook": len(nb_text),
                                           "slides": len(slides),
                                           "transcript": len(transcript)},
                                 "truncated": trunc})

    # #2 Faithfulness & Correctness
    try:
        from judge_faithfulness import judge_faithfulness
        report.faithfulness, info = _sampled(
            judge_faithfulness, K, ["assertional", "computational", "correctness"],
            judge, nb_text, nb_images, slides, transcript, verifier=verifier,
            composite=("score", lambda o: harmonic_mean(
                [o.assertional, o.computational, o.correctness]) or 0.0))
        f = report.faithfulness
        print(f"\n#2 Faithfulness&Correctness = {f.score}  "
              f"(2a={f.assertional} 2b={f.computational} 2c={f.correctness})")
        print(f"   {info}")
        print(f"   rationale: {f.rationale[:300]}")
        if f.errors_verified:
            for e in f.errors_verified:
                ax = "2a" if e.get("axis") == "assertional" else "2c"
                print(f"   {ax} error confirmed={e.get('confirmed')}: "
                      f"{e.get('error','')[:120]}")
    except Exception as e:  # noqa: BLE001
        print(f"#2 Faithfulness&Correctness = NA (judge failed after retries: {e})")

    # #3 Pedagogical Depth
    try:
        from judge_pedagogy import judge_pedagogy
        report.pedagogy, info = _sampled(
            judge_pedagogy, K, ["depth"], judge, nb_text, nb_images, slides)
        p = report.pedagogy
        print(f"\n#3 Pedagogical Depth = {p.depth}")
        print(f"   adds: {p.added_value}")
        print(f"   rationale: {p.rationale[:300]}")
    except Exception as e:  # noqa: BLE001
        print(f"#3 Pedagogical Depth = NA (judge failed after retries: {e})")

    # #4 Topic-Worthiness
    try:
        from judge_topic import judge_topic
        report.topic, info = _sampled(
            judge_topic, K, ["worthiness"], judge, nb_text, slides, transcript)
        t = report.topic
        print(f"\n#4 Topic-Worthiness = {t.worthiness}  "
              f"(interactive_right_tool={t.interactive_is_right_tool})")
        print(f"   rationale: {t.rationale[:300]}")
    except Exception as e:  # noqa: BLE001
        print(f"#4 Topic-Worthiness = NA (judge failed after retries: {e})")

    # #7 Exposition / Clarity (lane-guarded; blind to slides)
    try:
        from judge_clarity import judge_clarity
        report.clarity, info = _sampled(
            judge_clarity, K, ["visual", "textual", "code_explanation"],
            judge, nb_text, nb_images,
            composite=("score", lambda o: harmonic_mean(
                [o.visual, o.textual, o.code_explanation]) or 0.0))
        c = report.clarity
        print(f"\n#7 Clarity = {c.score}  "
              f"(7a={c.visual} 7b={c.textual} 7c={c.code_explanation})")
        print(f"   {info}")
        print(f"   rationale: {c.rationale[:300]}")
        for cite in c.citations:
            print(f"   {cite.get('axis','')}: {cite.get('where','')}: "
                  f"{str(cite.get('defect',''))[:120]}")
    except Exception as e:  # noqa: BLE001
        print(f"#7 Clarity = NA (judge failed after retries: {e})")

    (out / "quality_report.json").write_text(to_json(report), encoding="utf-8")
    print(f"\nwrote {out/'quality_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
