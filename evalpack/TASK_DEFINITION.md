# Demo Generation Task (v1)

> The task every method (pipeline, agent, or baseline model) is evaluated on.
> The sample input prompt that operationalizes this definition is in
> [`PROMPT_TEMPLATE.txt`](PROMPT_TEMPLATE.txt) — give it (plus the slide deck)
> to each method verbatim, so the contract binds all methods identically.

## Goal

Given the full slide deck for one lecture/concept-segment as context, a method
must **select a single core concept** from that material and produce **one
interactive notebook (`.ipynb`)** that teaches that concept through hands-on
interaction, going beyond what the static slides convey.

## What counts as one concept

The unit is **one learning question**, not one model.

- **One concept** = a single learning question explored over a *shared frame of
  reference* — the same task/dataset, **or** the same evaluation metric and
  output axes — where every interactive control and every model/method shown
  exists to answer *that one* question.
- **Allowed within one concept:** comparing multiple models/methods/algorithms
  head-to-head on the same axis (an interactive benchmark); exploring multiple
  parameters, regimes, or sub-aspects of the same mechanism; multiple
  views/plots of the same underlying example.
- **Not allowed:** two or more self-contained setups that answer *different*
  learning questions, each with its own task and framing, that a learner would
  read as separate lessons.
- **Quick check (deletion test):** remove any section and ask what was lost.
  If you lose *a contestant in the same comparison* (same lesson, fewer cases)
  → fine. If you lose *an entire standalone lesson* (a different question with
  its own setup) → that's multiple demos.

## Input

- The method receives the **full slide deck** for one lecture/concept-segment
  as context (rendered pages and/or extracted text), up to a safety ceiling
  (~30 slides).
- The deck is **not physically cut** into single-concept slices; prerequisite
  material stays available so no context is orphaned. Concept-segmentation
  defines task boundaries/labels only — it does not excise slides.
- Optional: lecture transcript may be supplied as additional context.
- The target concept is **not named** for the method — selecting it is part of
  the task.

## Output

- Exactly **one runnable `.ipynb`**.
- The notebook **must declare its single core concept up front** — an opening
  markdown cell (or sidecar metadata) stating **(a)** the one learning question
  the demo answers and **(b)** one line on why it's worth an interactive demo.
- The notebook must be **genuinely interactive**: at least one
  learner-manipulable control (ipywidgets or equivalent) whose effect is
  observable in the output.
- All content must serve the declared concept under the one-concept rule above.
- A notebook with **no clearly stated core concept is off-spec**, not merely
  poorly written.

## Scoring

Each output is scored on the 7-metric vector (no single composite grade):

| # | Metric | Scorer |
|---|--------|--------|
| 1 | Run Success | deterministic execution in the canonical env |
| 2 | Faithfulness & Correctness (2a assertional / 2b computational / 2c correctness-in-truth) | LLM judge + independent verifier; harmonic |
| 3 | Pedagogical Depth (marginal value over the slide) | LLM judge, slide-aware |
| 4 | Topic Significance (was: Topic-Worthiness) | LLM judge, slide-aware |
| 5 | Interactivity (effectiveness × robustness) | deterministic actuation + blind LLM judge; harmonic |
| 6 | Efficiency | deterministic (runtime, cost) |
| 7 | Clarity (7a visual / 7b textual / 7c code-explanation) | LLM judge, blind to slides; harmonic |

- **Concept selection is scored.** Methods choose their own concept, so the
  choice itself is evaluated (#4 Topic Significance), read from the declaration.
- Because selection is free, two methods may pick different concepts for the
  same deck; cross-method comparison is therefore **unpaired / aggregate**, not
  paired per-deck. This is accepted.
- **On violation of the one-concept rule:** evaluation scores the **declared
  concept only**; content beyond it is treated as off-spec sprawl and may count
  against selection/scope. *(The cheap concept-count guard and
  "judge declared-concept-only" behavior are a deferred pipeline change; the
  contract is fixed now so the hook exists.)*

## Same for every method

The same task definition and instruction go to every method. The
single-concept-and-declaration contract binds all methods identically; a method
whose architecture already produces a focused, declared concept simply
satisfies it natively rather than being privileged by it.

## Design notes (why the contract looks like this)

- **Why one concept per notebook:** without it, methods diverge in *how many*
  mini-demos they stitch into one `.ipynb`, and the whole-notebook judges then
  carry an unpredictable multi-demo bias (a "lots of content" halo on
  pedagogy, or one flawed mini-demo tanking the whole faithfulness score).
  Fixing the unit of work removes that bias without paying per-demo re-judging.
- **Why the unit is a question, not a model:** a head-to-head comparison of N
  methods on one task/metric (an interactive benchmark) is *one* lesson; a
  "single shared model" rule would wrongly forbid it. The deletion test
  operationalizes the boundary.
- **Why the declaration is mandatory:** it gives #4 a direct object to score,
  turns the one-concept guard into "does the rest stay within the declared
  question?", and makes "evaluate the declared concept only" well-defined when
  a method sprawls anyway.
- **Why ipywidgets is nudged in the prompt:** the interactivity harness
  actuates the kernel/widget lane; a demo built on another mechanism would not
  be actuated and would score unfairly low on #5. The constraint is applied to
  every method equally.
- **Honesty flag for the writeup:** "one focused single-concept demo" matches
  what Method 3's architecture produces natively, while constraining baselines
  away from their default sprawl. The defense — focused single-concept demos
  are pedagogically motivated, and the instruction is uniform — should be made
  explicitly in any report rather than left for a reviewer to notice.
