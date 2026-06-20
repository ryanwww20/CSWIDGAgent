# Results layout

Every `run_eval.py` invocation writes one run directory:

```
results/<method>/<run_id>/
├── summary.json               7-metric vector + provenance (start here)
├── exec/
│   ├── report.json            #1 run_success, #6 duration_seconds, stats, hardware
│   └── executed.ipynb         the notebook with real cell outputs (judge evidence)
├── interactivity/
│   ├── filmstrip.json         per-step actuation trace (incl. stdout/stderr)
│   ├── frames/                PNG snapshot after every control action
│   ├── judge.json             blind judge's usefulness verdict + citations
│   └── interactivity_score.json   #5 = harmonic(effectiveness, robustness)
├── quality/
│   ├── quality_report.json    #2 #3 #4 #7 (+ sub-scores, verified errors,
│   │                          citations, truncation/meta)
│   ├── nb_images/             raw figures extracted from the executed notebook
│   └── judge_images/          the labeled images actually sent to judges
└── llm_calls.jsonl            every LLM call verbatim: system+user prompts,
                               image paths, raw response, usage, retries and
                               contract rejections included
```

Conventions:

- `<method>` = whatever you pass as `--method` (one folder per system under
  comparison). `<run_id>` defaults to a UTC timestamp; pass `--run-id` for
  meaningful names (`pass1`, `seed42`, ...).
- **NA vs 0**: missing metrics appear as `null` in `summary.json` (judge
  unavailable, stage skipped, `--no-llm`). A numeric score is always a real
  judgment; scores are never silently defaulted.
- Nothing here is ever overwritten by other runs — each run is append-only
  evidence. Keep `llm_calls.jsonl` if you keep the scores: it is the audit
  trail that makes a number defensible later.
