# CSWIDGAgent — Computer Science Webcourse Interactive Demo Generating Agent

## Project Overview

This repository generates interactive Jupyter notebook demos for NTU computer science webcourses. The agent takes lecture PDFs (`course_source/`) as input and produces self-contained, widget-driven notebooks that students can run step-by-step to explore algorithm concepts.

## Repository Structure

```
CSWIDGAgent/
├── course_source/          # Source lecture PDFs (gitignored)
├── .venv/                  # Python virtual environment (gitignored)
├── CLAUDE.md               # This file
└── README.md
```

## Python Environment

- **Python version:** 3.12.3
- **Virtual environment:** `.venv/` (activate with `source .venv/bin/activate`)
- **Key packages:**
  - `numpy` 2.4.5 — numerical computation
  - `matplotlib` 3.10.9 — plotting
  - `networkx` 3.6.1 — graph data structures and layout
  - `ipywidgets` 8.1.8 — interactive sliders, buttons, dropdowns
  - `ipykernel` 7.2.0 — Jupyter kernel
  - `jupyter_client` 8.8.0 — notebook execution
