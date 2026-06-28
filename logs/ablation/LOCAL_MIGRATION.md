# Move the ablation study to local (macOS Apple Silicon, 32 GB)

Current progress on SSH: **53/126 notebooks generated, 37/126 evalkit-scored.**
Resume is automatic — copying the output dirs over means local only fills the gaps.

Full matrix = 6 conditions × 7 topics × 3 seeds = 126.

---

## 1. (on SSH) make sure nothing is mid-write
```bash
ps -eo cmd | grep -E 'scripts/run_ablation' | grep -v grep   # expect empty
```

## 2. (on local) pull the whole project over
Run FROM the Mac. `<ssh>` = your SSH host alias.
```bash
rsync -avz --progress --exclude '__pycache__' --exclude '*.pyc' \
  <ssh>:/work/b12901015/CSWIDGAgent/  ~/CSWIDGAgent/
```
This brings code (incl. the committed `pipeline_runner` patch), `runs/`,
`evalkit/results/`, `course_source/`, `data/MNIST/`, and `evalkit/.env`
(OpenAI key already inside — 164 chars). ~250 MB.

## 3. (on local) build the eval env
```bash
cd ~/CSWIDGAgent
conda env create -f logs/ablation/environment.yml
conda run -n ml-colab-eval python -m playwright install chromium   # interactivity stage
# sanity:
conda run -n ml-colab-eval python -c "import torch,nbclient,numpy,fitz,playwright;print('env OK', torch.__version__)"
```
(If `torch==2.12.1` isn't on PyPI for arm64, install the nearest 2.12.x and note it.)

## 4. (on local) register the execution kernel
The pipeline executes notebooks on a kernelspec named by `KERNEL_NAME`. Register
this env under its own name (avoids clobbering any existing `python3` kernel):
```bash
conda run -n ml-colab-eval python -m ipykernel install --user \
  --name ml-colab-eval --display-name "Python 3 (ml-colab-eval)"
```
Then always launch with `KERNEL_NAME=ml-colab-eval` (see step 6).
> macOS note: `ulimit -v` is unreliable, so there is NO 32 GiB memory cap like on
> the Linux node. A runaway notebook can't reboot the Mac (it `MemoryError`s), and
> `PER_CELL_TIMEOUT=300` kills long cells. Acceptable on 32 GB.

## 5. (on local) claude CLI
```bash
claude --version    # install if missing, then log in so `claude -p` works headless
```

## 6. (on local) run the resume — in your OWN tmux so it survives disconnects
```bash
cd ~/CSWIDGAgent
tmux new -s ablation
KERNEL_NAME=ml-colab-eval MAXJOBS=2 bash logs/ablation/launch_local.sh 2>&1 | tee logs/ablation/local_run.log
# detach: Ctrl-b then d   |   reattach: tmux attach -t ablation
```
Watch progress from another window:
```bash
bash logs/ablation/progress.sh
```

## 7. retry pass (I3) + aggregate
A few runs fail with `generation_failed` when an LLM omits a required JSON key —
expected, do NOT edit agent prompts. Re-run the same launcher 1–2× to fill them:
```bash
KERNEL_NAME=ml-colab-eval MAXJOBS=2 bash logs/ablation/launch_local.sh 2>&1 | tee -a logs/ablation/local_run.log
```
When generated == scored == 126, build the final tables:
```bash
conda run -n ml-colab-eval python scripts/lib/aggregate_evalkit.py \
  --results-dir evalkit/results --out-dir results --baseline B
# -> results/evalkit_runs.csv , results/evalkit_by_condition.md
```

## What does NOT transfer / differs from SSH
- conda env (binary) — rebuilt in step 3, not rsynced.
- `LD_LIBRARY_PATH` hack — Linux-only, dropped in `launch_local.sh`.
- 32 GiB kernel ulimit cap (I2) — Linux-only, see step 4 note.
- absolute `/work/...` paths — `launch_local.sh` derives them; the old
  `launch_resume.sh`/`launch_detached.sh` are SSH-only.
