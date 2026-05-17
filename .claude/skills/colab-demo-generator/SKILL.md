---
name: colab-demo-generator
description: Generate and improve interactive Google Colab notebooks from CS or ML course materials. Use this skill when asked to create, revise, or evaluate an educational Colab demo notebook.
---

# Colab Demo Generator Skill

## Purpose

Generate an educational Google Colab notebook that teaches one computer science or machine learning concept through runnable code, interaction, and visualization.

## Input

The input may include:

- Lecture slides
- Transcript
- Metadata
- Topic description
- Existing notebook
- Evaluation feedback

## Notebook Structure

The notebook should include:

1. Title
2. Learning objectives
3. Concept explanation
4. Setup cell
5. Core demo code
6. Interactive controls
7. Visualization
8. Student reflection questions
9. Suggested extensions

## Colab Rules

- The notebook must run from top to bottom in Google Colab.
- Avoid local file paths.
- Keep dependencies lightweight.
- Prefer built-in Colab-compatible libraries.
- Use `ipywidgets` only when it improves learning.
- Use GPU only when useful.
- If using GPU, include automatic device detection:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)