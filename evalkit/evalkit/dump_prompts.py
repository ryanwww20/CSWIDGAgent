"""Extract every LLM prompt constant from the pipeline (via AST, no imports) and
write specs/LLM_PROMPTS.md — a single readable reference. Re-run after editing any
prompt to refresh the doc:  python evalkit/dump_prompts.py"""
import ast
from pathlib import Path

KIT = Path(__file__).resolve().parent
OUT = KIT.parents[1] / "specs" / "LLM_PROMPTS.md"

def _eval(node):
    """Assemble a string from Constant / implicit-concat / BinOp(+); a Name (e.g.
    TRUNCATION_RULE) becomes an inline '[+ NAME]' marker."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval(node.left) + _eval(node.right)
    if isinstance(node, ast.Name):
        return f"\n\n[+ {node.id} — see appendix]"
    return "<unparZable>"

def consts(fname):
    tree = ast.parse((KIT / fname).read_text(encoding="utf-8"))
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = _eval(n.value)
            except Exception:
                pass
    return out

# (file, [(const, heading, blurb)])  — order defines the doc
STRUCT = [
 ("Metric #2 — Faithfulness & Correctness", "judge_faithfulness.py",
  "Three 1-5 sub-scores (2a assertional / 2b computational / 2c correctness), harmonic mean. "
  "2a/2c use burden-of-proof + a cross-family verifier (cited errors must be confirmed or the "
  "deduction reverts); 2b is judged from the code directly and was given anchored bands.",
  [("_SYSTEM","System prompt"),("_USER_TMPL","User template"),("_VERIFY_SYSTEM","Cross-family verifier prompt")]),
 ("Metric #3 — Pedagogical Depth", "judge_pedagogy.py",
  "Marginal value over the slide, scored on REALIZED value (what is visible in the outputs, not "
  "intended). Calibrated: full 1-5 range, broken/empty outputs land at 1-2.",
  [("_SYSTEM","System prompt"),("_USER_TMPL","User template")]),
 ("Metric #4 — Topic-Worthiness", "judge_topic.py",
  "Was the chosen concept worth demoing? Scored on centrality x richness x interactivity-payoff, "
  "implementation-independent. Calibrated: full 1-5 range (was collapsed at 5.0).",
  [("_SYSTEM","System prompt"),("_USER_TMPL","User template")]),
 ("Metric #5 — Interactivity (blind tier-aware judge)", "judge.py",
  "Per-control usefulness (1-5) from the recorded filmstrip; tier-aware (executed vs reasoned). "
  "Calibrated with explicit per-control bands.",
  [("_SYSTEM","System prompt"),("_USER_TMPL","User template")]),
 ("Metric #7 — Exposition / Clarity", "judge_clarity.py",
  "Three 1-5 sub-scores (7a visual / 7b textual / 7c code), harmonic mean; blind to slides. "
  "Already strict/anchored — left unchanged (the template the others were calibrated toward).",
  [("_SYSTEM","System prompt"),("_USER_TMPL","User template")]),
 ("Interactivity planner (#5 exploration, not a judge)", "planner.py",
  "Proposes what to try; emits data only, never scores. Widget planner drives ipywidgets; the "
  "semantic planner finds the signposted non-widget surface (constants / inputs / view-only).",
  [("_SYSTEM","Widget-planner system"),("_USER_TMPL","Widget-planner user"),
   ("_SEMANTIC_SYSTEM","Semantic-planner system"),("_SEMANTIC_USER","Semantic-planner user")]),
 ("Slide digest (vision, not a judge)", "slide_digest.py",
  "Once-per-deck vision pass that caches per-slide text + figure descriptions so the slide-aware "
  "judges see real slide content at ~zero marginal cost.",
  [("_PROMPT","Vision digest prompt")]),
]

lines = ["# LLM Prompts — ML-Demo Evaluation Pipeline", "",
 "Auto-generated from `evalkit/*.py` by `evalkit/dump_prompts.py` — re-run to refresh.", "",
 "`SYSTEM` = the role/instructions; `USER` = the per-sample template. The quality judges "
 "(#2/#3/#4) and the #5 judge were calibrated for discrimination; #7 clarity was the strict "
 "template they were tuned toward.", "",
 "In USER templates, `{name}` is a fill-in placeholder (slides, notebook, …) and `{{ }}` are "
 "literal braces — the JSON skeleton the model must return (Python `.format` escaping).", "",
 "## Contents", ""]
for title,_,_,_ in STRUCT:
    anchor = title.lower().replace(" ","-")
    for ch in "—()/#,.": anchor = anchor.replace(ch,"")
    anchor = anchor.replace("--","-")
    lines.append(f"- [{title}](#{anchor})")
lines.append("")

trunc = consts("textbudget.py").get("TRUNCATION_RULE","")
for title, fname, blurb, items in STRUCT:
    c = consts(fname)
    lines += [f"## {title}", "", f"_Source: `evalkit/{fname}`_  —  {blurb}", ""]
    for const, heading in items:
        text = c.get(const, "(not found)")
        lines += [f"### {heading}  (`{const}`)", "", "```text", text.strip("\n"), "```", ""]

lines += ["## Appendix — `TRUNCATION_RULE`", "",
 "Appended to the #2/#3/#4/#7 system prompts (marked `[+ TRUNCATION_RULE]` above).", "",
 "```text", trunc.strip("\n"), "```", ""]

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}  ({len(lines)} lines, {OUT.stat().st_size} bytes)")
