# Ablation study — issues log

Locked: 7 topics (MLBasic, whydeeplearning, ml2021_self_attention, autoencoder,
gan, MetaLearning, DRL) x 6 conditions x 3 seeds; evalkit abc full quality eval.
Env: single conda env `ml-colab-eval` (torch 2.12.1+cpu) = PYTHON_BIN + EVALKIT_PY.
Run: 6 parallel condition-shards in tmux session `ablation`.

## Resolved / known

### I1 — concurrent conda builds corrupted shared pkgs cache [FIXED]
Parallel `conda create` + `conda env create` raced on /work/.../pkgs (tesseract,
python archives) -> InvalidArchiveError, both envs failed. Fix: `conda clean --all`
then build SEQUENTIALLY. Also collapsed to ONE env (ml-colab-eval has
nbformat/nbclient/ipykernel/pymupdf already) serving both roles.

### I2 — runaway notebook OOM-killed the node [FIXED]
S0 gan single-shot notebook had range(3000) loop -> ~745 GB RSS -> global OOM
killer SIGKILLed execution (node has 754 GB). `claude` (node) needs ~74 GB virtual
at baseline so a shell ulimit can't separate gen from exec. FIX: cap at KERNEL
level — wrapped the python3 kernelspec that jupyter_client resolves
(/home/b12901015/.local/share/jupyter/kernels/python3/kernel.json) with
`ulimit -v 33554432` (32 GiB). Verified: torch runs, 320 GB alloc -> catchable
MemoryError, node safe. Covers evalkit, execute_notebook.py, B+bug_solver verify.

### I3 — stochastic schema slips -> generation_failed (EXPECTED, not a bug)
Some runs fail when an agent omits required JSON keys, e.g.:
  B__MLBasic__s1: notebook-architect 02 missing ['assumptions','global_flow_notes']
  ablate-cell-analyzer__MLBasic__s1: generation_failed
pipeline_runner validates -> dies -> ablation records generation_failed (no nb).
Known transient LLM issue (see EVAL_REPORT). Do NOT edit agent prompts mid-run
(would confound ablation). Handling: after all shards finish, run RETRY passes
(same command; resume regenerates only missing notebooks) 1-2x, THEN aggregate.
Track: grep -c "generation failed" logs/ablation/shard_*.log

## Operational notes
- scratchpad (/tmp/.../scratchpad) is EPHEMERAL and was wiped mid-session; keep all
  durable scripts/logs under repo logs/ablation/.
- Launcher: logs/ablation/launch_parallel.sh ; per-shard logs shard_<cond>_<ts>.log
- Driver (single-process fallback): logs/ablation/run_study.sh {gen3|abc}
