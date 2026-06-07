"""Generate kvcache_interactive_skill.ipynb from cell specs."""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "notebooks" / "kvcache_interactive_skill.ipynb"

def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

# ── Cell sources ────────────────────────────────────────────────────────

C01 = md(
"""# KV Cache: Efficient LLM Inference — Interactive Demo

*NTUEE / Machine Learning Course — Source: KV Cache Lecture Slides*

---

This notebook walks through four core concepts in efficient LLM inference:

| Section | Concept |
|---------|---------|
| **1** | KV Cache Basics — why naive decoding is O(T²) and how caching cuts it to O(T) |
| **2** | Memory Footprint — how much GPU RAM each cached token occupies |
| **3** | MHA / MQA / GQA — sharing K/V heads to shrink the cache |
| **4** | Prefix Caching — reusing the KV cache across conversations |

**Prerequisites:** Transformer self-attention (`softmax(QKᵀ/√d)V`), basic Python/NumPy.
**Runtime:** Runs fully on CPU — no GPU required. Estimated time: ~5 min.
""")

C02 = code(
"""# ── Setup ─────────────────────────────────────────────────────────────────
# Run this cell first; all other cells depend on these imports.
!pip install -q anthropic

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import ipywidgets as widgets
from ipywidgets import interact, Layout, VBox
from IPython.display import display, HTML, clear_output
import pandas as pd
import hashlib, os, time

%matplotlib inline

print(f"numpy  : {np.__version__}")
print(f"torch  : {torch.__version__}")
print(f"widgets: {widgets.__version__}")

if not torch.cuda.is_available():
    print("\\u26a0  CUDA not available \\u2014 all attention ops run on CPU (fine for this notebook).")

assert int(np.__version__.split('.')[0]) >= 1, 'numpy >= 1.x required'
print("\\nAll imports OK \\u2713")
""")

C03 = md(
"""---
## Section 1 · KV Cache Basics

### The Attention Mechanism Refresher

In each transformer layer, self-attention computes:

$$\\text{output}_t = \\text{softmax}\\!\\left(\\frac{q_t K^T}{\\sqrt{d}}\\right) V$$

where $q_t$ is the **query** for the current token, and $K$, $V$ hold all context keys/values.

### The Naive Decoding Problem

During **autoregressive generation** we produce one token at a time.  At step $t$:

| Approach | K/V projection cost per step | Total over T steps |
|---|---|---|
| **Naive** | Re-project all $t+1$ past tokens | $\\propto T^2 \\cdot d$ |
| **KV Cache** | Project only the **new** token | $\\propto T \\cdot d$ |

The cached keys and values from previous steps are stored and reused — never recomputed.
""")

C04 = code(
"""# ── C04: Naive Attention Decode ──────────────────────────────────────────
# Run C02 first

def naive_attention_decode(token_embeddings, W_q, W_k, W_v):
    \"\"\"
    Autoregressive decode WITHOUT KV cache.
    At every step t, recomputes K and V for ALL previous tokens.
    Returns (outputs, op_counts) where op_counts[t] = cumulative K/V projection ops.
    \"\"\"
    assert token_embeddings.ndim == 2, "Expected (T, d) array"
    d = token_embeddings.shape[1]
    assert W_q.shape == W_k.shape == W_v.shape == (d, d), "Weight matrix shape mismatch"

    T = token_embeddings.shape[0]
    outputs, op_counts, cumulative_ops = [], [], 0

    for t in range(T):
        # Recompute K and V for ALL tokens 0..t (the expensive part)
        K = token_embeddings[:t+1] @ W_k   # (t+1, d)
        V = token_embeddings[:t+1] @ W_v   # (t+1, d)
        q = token_embeddings[t]   @ W_q    # (d,)

        scores = q @ K.T / np.sqrt(d)      # (t+1,)
        scores -= scores.max()              # numerical stability
        weights = np.exp(scores)
        weights /= weights.sum()
        assert weights.sum() > 1e-9, "Degenerate softmax"

        outputs.append(weights @ V)         # (d,)

        # Op count: 2*(t+1) projection matmuls (K and V each over t+1 tokens)
        cumulative_ops += 2 * (t + 1)
        op_counts.append(cumulative_ops)

    return outputs, op_counts

# ── Sanity check ──────────────────────────────────────────────────────────
np.random.seed(42)
_emb = np.random.randn(8, 4)
_W   = np.eye(4)
_out, _ops = naive_attention_decode(_emb, _W, _W, _W)
assert len(_out) == 8 and len(_ops) == 8
assert _ops[-1] == 2 * sum(range(1, 9)), f"Expected {2*36}, got {_ops[-1]}"
assert _out[0].shape == (4,)
print(f"Naive op counts : {_ops}")
print("naive_attention_decode \\u2713")
""")

C05 = code(
"""# ── C05: KV-Cached Attention Decode ──────────────────────────────────────
# Run C02, C04 first

def cached_attention_decode(token_embeddings, W_q, W_k, W_v):
    \"\"\"
    Autoregressive decode WITH KV cache.
    At step t, only projects the NEW token; reuses all cached K/V.
    Returns (outputs, op_counts, (cache_k, cache_v)).
    \"\"\"
    assert token_embeddings.ndim == 2, "Expected (T, d) array"
    d = token_embeddings.shape[1]
    assert W_q.shape == W_k.shape == W_v.shape == (d, d), "Weight matrix shape mismatch"

    T = token_embeddings.shape[0]
    kv_cache_k, kv_cache_v = [], []
    outputs, op_counts, cumulative_ops = [], [], 0

    for t in range(T):
        # Project ONLY the new token (constant cost regardless of t)
        k_t = token_embeddings[t] @ W_k    # (d,)
        v_t = token_embeddings[t] @ W_v    # (d,)
        kv_cache_k.append(k_t)
        kv_cache_v.append(v_t)
        assert len(kv_cache_k) == t + 1    # cache grows monotonically

        K = np.stack(kv_cache_k)           # (t+1, d)  ← from cache
        V = np.stack(kv_cache_v)           # (t+1, d)  ← from cache
        q = token_embeddings[t] @ W_q      # (d,)

        scores = q @ K.T / np.sqrt(d)
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()
        outputs.append(weights @ V)

        # Only 2 projection ops (K and V for the new token only)
        cumulative_ops += 2
        op_counts.append(cumulative_ops)

    return outputs, op_counts, (kv_cache_k, kv_cache_v)

# ── Verify outputs match naive exactly ───────────────────────────────────
np.random.seed(42)
emb = np.random.randn(8, 4)
W   = np.eye(4)
naive_out, naive_ops           = naive_attention_decode(emb, W, W, W)
cached_out, cached_ops, _cache = cached_attention_decode(emb, W, W, W)

for t, (n, c) in enumerate(zip(naive_out, cached_out)):
    if not np.allclose(n, c, atol=1e-6):
        print(f"  Mismatch at step {t}:\\n  naive={n}\\n  cached={c}")
        raise AssertionError(f"Output mismatch at step {t}")

assert cached_ops[-1] == 2 * 8, f"Cached total ops should be 16, got {cached_ops[-1]}"
print("Outputs match: True")
print(f"Naive  op counts: {naive_ops}")
print(f"Cached op counts: {cached_ops}")
""")

C06 = code(
"""# ── C06: Recomputation Grid Visualization ────────────────────────────────
# Run C02 first

def build_naive_grid(T):
    \"\"\"naive_grid[step, pos] = 1 if pos <= step (recomputed), else 0.\"\"\"
    assert T >= 2
    grid = np.zeros((T, T), dtype=int)
    for step in range(T):
        grid[step, :step+1] = 1
    return grid

def build_cached_grid(T):
    \"\"\"cached_grid[step, pos]: 2=new this step, 1=served from cache, 0=future.\"\"\"
    assert T >= 2
    grid = np.zeros((T, T), dtype=int)
    for step in range(T):
        grid[step, :step] = 1   # served from cache (free reuse)
        grid[step, step]  = 2   # newly computed this step
    return grid

T_VIZ = 8
naive_grid  = build_naive_grid(T_VIZ)
cached_grid = build_cached_grid(T_VIZ)

# Assertions
assert naive_grid[0, 0] == 1 and naive_grid[0, 1] == 0
assert np.sum(naive_grid[7]) == 8
assert cached_grid[3, 3] == 2 and cached_grid[3, 2] == 1 and cached_grid[3, 4] == 0

cmap_naive  = ListedColormap(['#f8f9fa', '#ffcccc', '#cc0000'])
cmap_cached = ListedColormap(['#f8f9fa', '#c3e6cb', '#155724'])

plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, grid, cmap, title, legend_items in [
    (axes[0], naive_grid,  cmap_naive,
     "Naive: Recompute ALL past K/V each step",
     [mpatches.Patch(color='#cc0000', label='Recomputed (wasted work)'),
      mpatches.Patch(color='#f8f9fa', label='Not yet generated', ec='gray')]),
    (axes[1], cached_grid, cmap_cached,
     "KV Cache: Only project the NEW token",
     [mpatches.Patch(color='#155724', label='New token (projected once)'),
      mpatches.Patch(color='#c3e6cb', label='Served from cache (free!)'),
      mpatches.Patch(color='#f8f9fa', label='Not yet generated', ec='gray')]),
]:
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation='nearest', aspect='auto')
    ax.set_xticks(range(T_VIZ)); ax.set_xticklabels([f"Tok {i}" for i in range(T_VIZ)],
                                                     rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(T_VIZ)); ax.set_yticklabels([f"Step {i}" for i in range(T_VIZ)], fontsize=9)
    ax.set_xlabel("Token Position"); ax.set_ylabel("Generation Step")
    ax.set_title(title, fontweight='bold')
    ax.legend(handles=legend_items, loc='upper left', fontsize=8)
    for r in range(T_VIZ):
        for c_ in range(T_VIZ):
            if grid[r, c_] > 0:
                lbl = "N" if grid[r, c_] == 2 else ("C" if cmap == cmap_cached else "R")
                ax.text(c_, r, lbl, ha='center', va='center', fontsize=7.5,
                        color='white' if grid[r, c_] == 2 else '#333')

plt.suptitle("K/V Projection Work per Generation Step", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
""")

C07 = code(
"""# ── C07: Interactive FLOP Comparison ─────────────────────────────────────
# Run C02, C04, C05 first

def plot_flop_comparison(T=32):
    np.random.seed(0)
    d    = 16
    emb_ = np.random.randn(T, d)
    W_   = np.eye(d)

    _, n_ops        = naive_attention_decode(emb_, W_, W_, W_)
    _, c_ops, _     = cached_attention_decode(emb_, W_, W_, W_)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 4))
    steps = list(range(1, T + 1))
    ax.plot(steps, n_ops, 'r-o', markersize=3, linewidth=2,
            label=f"Naive O(T\\u00b2) \\u2014 {n_ops[-1]:,} total ops")
    ax.plot(steps, c_ops, color='#28a745', marker='o', markersize=3, linewidth=2,
            label=f"KV Cache O(T) \\u2014 {c_ops[-1]:,} total ops")
    ax.fill_between(steps, c_ops, n_ops, alpha=0.12, color='red',
                    label="Wasted work (recomputation)")
    ax.set_xlabel("Generation Step", fontsize=11)
    ax.set_ylabel("Cumulative K/V Projection Ops", fontsize=11)
    ax.set_title(f"KV Cache vs Naive Decoding  (T = {T})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    speedup = n_ops[-1] / c_ops[-1]
    print(f"  At T={T}: KV Cache is {speedup:.0f}\\u00d7 cheaper in K/V projection ops  "
          f"({c_ops[-1]:,} vs {n_ops[-1]:,})")

# Static run first (verifies no errors)
plot_flop_comparison(16)

# Interactive slider
interact(
    plot_flop_comparison,
    T=widgets.IntSlider(min=4, max=256, step=4, value=32,
                        description="Seq length T",
                        continuous_update=False,
                        style={'description_width': 'initial'})
)
""")

C08 = md(
"""---
## Section 2 · KV Cache Memory Footprint

### Formula

$$\\text{bytes per token} = N_{\\text{layers}} \\times N_{\\text{KV heads}} \\times d_{\\text{head}} \\times \\text{bytes/element} \\times 2$$

The factor **2** accounts for **K** and **V** — the query **Q** is _not_ cached.

### Worked Example: Gemma 2 27B (GQA, `num_kv_heads = 16`)

$$46 \\times 16 \\times 128 \\times 2\\,(\\text{FP16}) \\times 2 = 376{,}832 \\text{ bytes} \\approx 0.37\\,\\text{MB/token}$$

> **Note on the lecture slide:** The slide computes $46 \\times 32 \\times 128 \\times 2 \\times 2 = 753{,}664$ bytes
> using `num_heads = 32` (the *query* head count).  Our formula uses `num_kv_heads = 16` — the
> actual KV cache size for GQA.

### The HBM ↔ SRAM Bottleneck

| Tier | Size | Speed |
|------|------|-------|
| **HBM** (High-Bandwidth Memory) | 40–80 GB | Slower per-byte bandwidth |
| **SRAM** (on-chip) | ~tens of MB | Very fast |

Loading the KV cache from HBM into SRAM for every attention step is the **primary bottleneck**
for long-context inference — not FLOP count.
""")

C09 = code(
"""# ── C09: KV Cache Memory Calculator Functions ────────────────────────────
# Run C02 first

DTYPE_BYTES = {'FP32': 4, 'FP16': 2, 'BF16': 2, 'INT8': 1}

def kv_cache_bytes_per_token(num_layers, num_kv_heads, head_dim, dtype_bytes):
    \"\"\"Total bytes to cache K and V for one token, across all layers.\"\"\"
    return num_layers * num_kv_heads * head_dim * dtype_bytes * 2  # ×2: K and V

def max_tokens(gpu_memory_gb, model_weight_gb, num_layers, num_kv_heads, head_dim, dtype_bytes):
    \"\"\"How many tokens fit in GPU memory after reserving space for model weights.\"\"\"
    if model_weight_gb >= gpu_memory_gb:
        raise ValueError(
            f"Model weights ({model_weight_gb:.0f} GB) exceed GPU memory "
            f"({gpu_memory_gb:.0f} GB); no room for KV cache."
        )
    available_bytes = (gpu_memory_gb - model_weight_gb) * (1024 ** 3)
    bpt = kv_cache_bytes_per_token(num_layers, num_kv_heads, head_dim, dtype_bytes)
    return int(available_bytes / bpt) if bpt > 0 else 0

# ── Sanity checks ─────────────────────────────────────────────────────────
assert kv_cache_bytes_per_token(1, 1, 1, 2) == 4
gemma_bpt = kv_cache_bytes_per_token(46, 16, 128, 2)
assert gemma_bpt == 46 * 16 * 128 * 2 * 2, f"Got {gemma_bpt}"

# ── Gemma 2 27B demonstration ─────────────────────────────────────────────
slide_bpt = kv_cache_bytes_per_token(46, 32, 128, 2)   # slide used query heads=32
gqa_bpt   = kv_cache_bytes_per_token(46, 16, 128, 2)   # correct: num_kv_heads=16

print("=" * 56)
print("Gemma 2 27B — KV Cache Memory Analysis")
print("=" * 56)
print(f"  Slide value (num_heads=32) : {slide_bpt:>10,} bytes = {slide_bpt/1024:.1f} KB/token")
print(f"  GQA correct (num_kv=16)   : {gqa_bpt:>10,} bytes = {gqa_bpt/1024:.1f} KB/token")
max_tok = max_tokens(80, 54, 46, 16, 128, 2)
print(f"  Max tokens on A100-80GB   : {max_tok:>10,}  (~{max_tok//1000}k)")
print("\\nkv_cache_bytes_per_token and max_tokens \\u2713")
""")

C10 = code(
"""# ── C10: Interactive Memory Calculator ───────────────────────────────────
# Run C02, C09 first

def show_memory_interactive(num_layers=32, num_kv_heads=8,
                             head_dim_str='128', dtype='FP16', gpu_gb_str='80'):
    head_dim   = int(head_dim_str)
    gpu_gb     = int(gpu_gb_str)
    dtype_b    = DTYPE_BYTES[dtype]
    weight_gb  = 14   # placeholder model-weight footprint

    bpt   = kv_cache_bytes_per_token(num_layers, num_kv_heads, head_dim, dtype_b)
    max_t = (max_tokens(gpu_gb, weight_gb, num_layers, num_kv_heads, head_dim, dtype_b)
             if weight_gb < gpu_gb else 0)

    ctx_lengths = [10_000, 50_000, 100_000]
    kv_gbs      = [cl * bpt / (1024 ** 3) for cl in ctx_lengths]

    plt.close('all')
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(13, 4))

    # Left: horizontal bars
    colors = ['steelblue' if g <= gpu_gb else '#dc3545' for g in kv_gbs]
    bars = ax_bar.barh([f"{cl//1000}k tokens" for cl in ctx_lengths], kv_gbs, color=colors)
    ax_bar.axvline(gpu_gb, color='red', linestyle='--', linewidth=1.5,
                   label=f"GPU limit ({gpu_gb} GB)")
    for bar, g in zip(bars, kv_gbs):
        lbl = f"OOM ({g:.1f} GB)" if g > gpu_gb else f"{g:.2f} GB"
        ax_bar.text(max(g * 0.02, 0.05), bar.get_y() + bar.get_height() / 2,
                    lbl, va='center', fontsize=9,
                    color='white' if g > gpu_gb else 'black')
    ax_bar.set_xlabel("GPU Memory (GB)")
    ax_bar.set_title("KV Cache Size at Various Context Lengths", fontweight='bold')
    ax_bar.legend(fontsize=9)

    # Right: pie
    kv_at_50k = 50_000 * bpt / (1024 ** 3)
    kv_used   = min(kv_at_50k, max(0, gpu_gb - weight_gb))
    free_gb   = max(0.01, gpu_gb - weight_gb - kv_used)
    w_clamped = min(weight_gb, gpu_gb)
    pie_vals  = [w_clamped, kv_used, free_gb]
    pie_labels = ['Model Weights', 'KV Cache @50k', 'Free']
    pie_colors = ['#f4a261', '#2a9d8f', '#e9ecef']
    ax_pie.pie(pie_vals, labels=pie_labels, colors=pie_colors,
               autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9})
    ax_pie.set_title(f"{gpu_gb} GB GPU — Memory Split", fontweight='bold')

    fig.suptitle(
        f"Per-token KV size: {bpt:,} bytes ({bpt/1024:.1f} KB)  |  "
        f"Max context: {max_t:,} tokens",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

interact(
    show_memory_interactive,
    num_layers   = widgets.IntSlider(min=1, max=80, step=1, value=32,
                                     description="Layers",
                                     continuous_update=False),
    num_kv_heads = widgets.IntSlider(min=1, max=64, step=1, value=8,
                                     description="KV heads",
                                     continuous_update=False),
    head_dim_str = widgets.Dropdown(options=['64','128','192','256'],
                                    value='128', description="Head dim"),
    dtype        = widgets.Dropdown(options=['FP32','FP16','BF16','INT8'],
                                    value='FP16', description="Dtype"),
    gpu_gb_str   = widgets.Dropdown(options=['24','40','80'],
                                    value='80', description="GPU mem"),
)
""")

C11 = code(
"""# ── C11: Per-Model KV Cache Comparison ───────────────────────────────────
# Run C02, C09 first

MODELS = {
    'Llama-3-8B'      : {'num_layers': 32, 'num_kv_heads':  8, 'head_dim': 128,
                          'dtype': 'FP16', 'weight_gb': 16},
    'Gemma-2-9B'      : {'num_layers': 42, 'num_kv_heads':  8, 'head_dim': 256,
                          'dtype': 'FP16', 'weight_gb': 18},
    'Gemma-2-27B'     : {'num_layers': 46, 'num_kv_heads': 16, 'head_dim': 128,
                          'dtype': 'FP16', 'weight_gb': 54},
    'GPT-3-175B (est)': {'num_layers': 96, 'num_kv_heads': 96, 'head_dim': 128,
                          'dtype': 'FP16', 'weight_gb': 350},
}
GPU_MEMORY_GB = 40

names, kb_per_tok, max_toks = [], [], []
for name, cfg in MODELS.items():
    bpt = kv_cache_bytes_per_token(
        cfg['num_layers'], cfg['num_kv_heads'],
        cfg['head_dim'], DTYPE_BYTES[cfg['dtype']]
    )
    mt = 0 if cfg['weight_gb'] >= GPU_MEMORY_GB else max_tokens(
        GPU_MEMORY_GB, cfg['weight_gb'],
        cfg['num_layers'], cfg['num_kv_heads'],
        cfg['head_dim'], DTYPE_BYTES[cfg['dtype']]
    )
    names.append(name); kb_per_tok.append(bpt / 1024); max_toks.append(mt)

# Verify Llama
assert kv_cache_bytes_per_token(32, 8, 128, 2) == 131072
assert kb_per_tok[0] == 128.0

plt.close('all')
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(names, kb_per_tok, color='cornflowerblue', edgecolor='white')
max_kb = max(kb_per_tok)
for bar, mt in zip(bars, max_toks):
    x_txt = bar.get_width() + max_kb * 0.02
    if mt > 0:
        ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                f"{mt//1000}k tokens max on {GPU_MEMORY_GB} GB", va='center', fontsize=9)
    else:
        ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                f"OOM on {GPU_MEMORY_GB} GB", va='center', fontsize=9,
                color='#dc3545', fontweight='bold')
ax.set_xlabel("KV Cache per Token (KB, FP16)", fontsize=11)
ax.set_title(f"Per-Model KV Cache Size", fontsize=12, fontweight='bold')
ax.set_xlim(0, max_kb * 1.45)
plt.tight_layout()
plt.show()
print("Model comparison chart rendered. \\u2713")
""")

C12 = md(
"""---
## Section 3 · MHA / MQA / GQA — Sharing K/V Heads

### The Memory-Quality Trade-off

| Variant | KV Heads | Cache vs MHA | Quality | Used in |
|---|---|---|---|---|
| **MHA** Multi-Head Attention | = `num_q_heads` | 1× (baseline) | Best | Original Transformer |
| **MQA** Multi-Query Attention | 1 | 1/H× | May degrade | Falcon |
| **GQA** Group-Query Attention | 1 < kv < H | 1/G× | Minor impact | **Llama 3, Gemma 2**, Mistral |

### How Sharing Works

```
MHA  Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7    (8 query heads, each reads its own KV)
      |  |  |  |  |  |  |  |
     K0 K1 K2 K3 K4 K5 K6 K7    (8 KV heads  →  1× baseline)

MQA  Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7    (8 query heads, all share ONE KV pair)
          \\ | | | | | /
             K / V               (1 KV head   →  8× reduction)

GQA  Q0 Q1   Q2 Q3   Q4 Q5   Q6 Q7    (groups of 2 queries share a KV head)
      \\ /     \\ /     \\ /     \\ /
      K0      K1      K2      K3       (4 KV heads  →  2× reduction)
```

KV cache size scales linearly with `num_kv_heads` — reducing it from H to G shrinks the cache by H/G.
""")

C13 = code(
"""# ── C13: MHA, MQA, GQA Attention Classes ─────────────────────────────────
# Run C02 first

class MultiHeadAttention(nn.Module):
    \"\"\"Standard MHA: each query head has its own dedicated K/V head.\"\"\"
    def __init__(self, d_model=64, num_q_heads=8):
        super().__init__()
        self.d_model     = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads
        self.head_dim    = d_model // num_q_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.kv_cache = None

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        B, T, _ = x.shape
        hd = self.head_dim
        Q = self.W_q(x).view(B, T, self.num_q_heads, hd).transpose(1, 2)   # (B,H,T,hd)
        K = self.W_k(x).view(B, T, self.num_kv_heads, hd).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_kv_heads, hd).transpose(1, 2)
        self.kv_cache = (K, V)
        scores  = Q @ K.transpose(-2, -1) / (hd ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out).squeeze(0)


class MultiQueryAttention(nn.Module):
    \"\"\"MQA: all query heads share a SINGLE K/V pair.\"\"\"
    def __init__(self, d_model=64, num_q_heads=8):
        super().__init__()
        self.d_model      = d_model
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = 1
        self.head_dim     = d_model // num_q_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.head_dim, bias=False)  # single head
        self.W_v = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.kv_cache = None

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        B, T, _ = x.shape
        hd = self.head_dim
        Q = self.W_q(x).view(B, T, self.num_q_heads, hd).transpose(1, 2)  # (B,H,T,hd)
        K = self.W_k(x).unsqueeze(1)   # (B,1,T,hd)  ← broadcast over H
        V = self.W_v(x).unsqueeze(1)   # (B,1,T,hd)
        self.kv_cache = (K, V)
        scores  = Q @ K.transpose(-2, -1) / (hd ** 0.5)   # (B,H,T,T) via broadcast
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out).squeeze(0)


class GroupedQueryAttention(nn.Module):
    \"\"\"GQA: groups of query heads share a K/V head.\"\"\"
    def __init__(self, d_model=64, num_q_heads=8, num_kv_groups=2):
        super().__init__()
        assert num_q_heads % num_kv_groups == 0, (
            f"num_q_heads={num_q_heads} must be divisible by num_kv_groups={num_kv_groups}"
        )
        self.d_model      = d_model
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = num_kv_groups
        self.head_dim     = d_model // num_q_heads
        self.groups_per_kv = num_q_heads // num_kv_groups
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_groups * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.kv_cache = None

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        B, T, _ = x.shape
        hd = self.head_dim
        Q = self.W_q(x).view(B, T, self.num_q_heads, hd).transpose(1, 2)      # (B,H_q,T,hd)
        K = self.W_k(x).view(B, T, self.num_kv_heads, hd).transpose(1, 2)     # (B,G,T,hd)
        V = self.W_v(x).view(B, T, self.num_kv_heads, hd).transpose(1, 2)
        self.kv_cache = (K, V)
        K_exp = K.repeat_interleave(self.groups_per_kv, dim=1)                 # (B,H_q,T,hd)
        V_exp = V.repeat_interleave(self.groups_per_kv, dim=1)
        scores  = Q @ K_exp.transpose(-2, -1) / (hd ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V_exp).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out).squeeze(0)


# ── Verification ──────────────────────────────────────────────────────────
torch.manual_seed(0)
x_demo = torch.randn(6, 64)
mha = MultiHeadAttention()
mqa = MultiQueryAttention()
gqa = GroupedQueryAttention()

with torch.no_grad():
    o_mha = mha(x_demo)
    o_mqa = mqa(x_demo)
    o_gqa = gqa(x_demo)

assert o_mha.shape == o_mqa.shape == o_gqa.shape == (6, 64)
assert mha.kv_cache[0].shape == (1, 8, 6, 8),  f"MHA K cache: {mha.kv_cache[0].shape}"
assert mqa.kv_cache[0].shape == (1, 1, 6, 8),  f"MQA K cache: {mqa.kv_cache[0].shape}"
assert gqa.kv_cache[0].shape == (1, 2, 6, 8),  f"GQA K cache: {gqa.kv_cache[0].shape}"
print("Output shapes :", o_mha.shape)
print("MHA KV cache  :", mha.kv_cache[0].shape, "(1, num_q_heads, T, head_dim)")
print("MQA KV cache  :", mqa.kv_cache[0].shape, "(1, 1, T, head_dim)")
print("GQA KV cache  :", gqa.kv_cache[0].shape, "(1, num_kv_groups, T, head_dim)")
print("All attention variants produce correct output shapes. \\u2713")
""")

C14 = code(
"""# ── C14: K/V Head Sharing Grid Visualization ─────────────────────────────
# Run C02, C09, C13 first

def build_head_sharing_grid(num_q_heads, num_kv_heads, seq_len):
    \"\"\"grid[q_head, pos] = KV head index that query head q reads.\"\"\"
    assert num_q_heads % num_kv_heads == 0
    gpk = num_q_heads // num_kv_heads
    grid = np.zeros((num_q_heads, seq_len))
    for q in range(num_q_heads):
        grid[q, :] = q // gpk
    return grid

# Assertions
grid_mha = build_head_sharing_grid(8, 8, 6)
assert grid_mha[0, 0] == 0 and grid_mha[1, 0] == 1
grid_mqa = build_head_sharing_grid(8, 1, 6)
assert np.all(grid_mqa == 0)
grid_gqa = build_head_sharing_grid(8, 2, 6)
assert grid_gqa[0, 0] == 0 and grid_gqa[4, 0] == 1

NUM_Q, SEQ = 8, 6
configs = [
    ("MHA (8 KV heads)", 8),
    ("MQA (1 KV head)",  1),
    ("GQA (2 KV heads)", 2),
]

plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, num_kv) in zip(axes, configs):
    grid   = build_head_sharing_grid(NUM_Q, num_kv, SEQ)
    cmap_g = plt.cm.get_cmap('tab10', max(num_kv, 2))
    ax.pcolormesh(np.arange(SEQ + 1), np.arange(NUM_Q + 1), grid,
                  cmap=cmap_g, vmin=0, vmax=max(num_kv, 2) - 0.01)
    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_ylabel("Query Head Index", fontsize=10)
    kb = kv_cache_bytes_per_token(32, num_kv, 128, 2) // 1024
    ax.set_title(f"{name}\\nKV cache = {kb} KB/token (32-layer model)", fontweight='bold')
    ax.set_xticks(np.arange(SEQ) + 0.5);  ax.set_xticklabels([f"t{i}" for i in range(SEQ)])
    ax.set_yticks(np.arange(NUM_Q) + 0.5); ax.set_yticklabels([f"Q{i}" for i in range(NUM_Q)])
    patches = [mpatches.Patch(color=cmap_g(i / max(num_kv, 2)), label=f"KV head {i}")
               for i in range(num_kv)]
    ax.legend(handles=patches, loc='upper right', fontsize=7)

plt.suptitle("Which KV Head Each Query Head Reads", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
""")

C15 = code(
"""# ── C15: Interactive KV Cache Reduction Slider ────────────────────────────
# Run C02, C09, C13 first

NUM_Q_HEADS = 8

def plot_kv_reduction(num_kv_groups=4):
    if NUM_Q_HEADS % num_kv_groups != 0:
        print(f"\\u26a0  {NUM_Q_HEADS} is not divisible by {num_kv_groups}. "
              f"Choose from: {[d for d in range(1,9) if NUM_Q_HEADS%d==0]}")
        return

    num_layers, head_dim, dtype_b = 32, 128, 2
    mha_bytes = kv_cache_bytes_per_token(num_layers, NUM_Q_HEADS,    head_dim, dtype_b)
    gqa_bytes = kv_cache_bytes_per_token(num_layers, num_kv_groups,  head_dim, dtype_b)
    ratio     = mha_bytes / gqa_bytes

    if   num_kv_groups == 1:            regime = "MQA (maximum compression)"
    elif num_kv_groups == NUM_Q_HEADS:  regime = "MHA (no compression — baseline)"
    else:
        gpk = NUM_Q_HEADS // num_kv_groups
        regime = f"GQA  ({gpk} query heads share each KV head)"

    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ['MHA (baseline)', f'Current ({num_kv_groups} KV heads)'],
        [mha_bytes / 1024, gqa_bytes / 1024],
        color=['#dc3545', '#28a745'], width=0.5, edgecolor='white'
    )
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{bar.get_height():.0f} KB", ha='center', va='bottom', fontsize=10)
    ax.set_ylabel("KV Cache per Token (KB)", fontsize=11)
    ax.set_title(f"{regime}\\nCache reduction: {ratio:.1f}\\u00d7", fontsize=12, fontweight='bold')
    ax.set_ylim(0, mha_bytes / 1024 * 1.25)
    plt.tight_layout()
    plt.show()
    print(f"  {regime} | Reduction: {ratio:.1f}\\u00d7 vs MHA")

interact(
    plot_kv_reduction,
    num_kv_groups=widgets.IntSlider(min=1, max=8, step=1, value=4,
                                    description="KV groups",
                                    continuous_update=False,
                                    style={'description_width': 'initial'})
)
""")

C16 = code(
"""# ── C16: MHA / MQA / GQA Comparison Table ────────────────────────────────
# Run C02, C09, C13 first

num_layers, head_dim, dtype_b, num_q = 32, 128, 2, 32  # Llama-3-8B-like
mha_bytes = kv_cache_bytes_per_token(num_layers, num_q, head_dim, dtype_b)

rows = [
    {'Variant': 'MHA',
     'num_kv_heads': 32,
     'KV Cache / Token (KB)': round(mha_bytes / 1024, 1),
     'Reduction vs MHA': '1.0\\u00d7',
     'Quality Impact': 'No change (baseline)',
     'Needs Retraining': 'N/A'},
    {'Variant': 'GQA (8 KV heads)',
     'num_kv_heads': 8,
     'KV Cache / Token (KB)': round(kv_cache_bytes_per_token(num_layers, 8, head_dim, dtype_b) / 1024, 1),
     'Reduction vs MHA': f'{num_q/8:.1f}\\u00d7',
     'Quality Impact': 'Minor impact',
     'Needs Retraining': 'Yes'},
    {'Variant': 'MQA (1 KV head)',
     'num_kv_heads': 1,
     'KV Cache / Token (KB)': round(kv_cache_bytes_per_token(num_layers, 1, head_dim, dtype_b) / 1024, 1),
     'Reduction vs MHA': f'{num_q/1:.1f}\\u00d7',
     'Quality Impact': 'May hurt quality',
     'Needs Retraining': 'Yes'},
]

assert len(rows) == 3
mha_row = next(r for r in rows if r['Variant'] == 'MHA')
assert mha_row['KV Cache / Token (KB)'] == 32 * 32 * 128 * 2 * 2 / 1024

df16 = pd.DataFrame(rows)

def color_quality(val):
    v = str(val)
    if 'baseline' in v: return 'background-color: #d4edda; color: #155724'
    if 'may hurt' in v.lower(): return 'background-color: #f8d7da; color: #721c24'
    if 'minor' in v.lower(): return 'background-color: #fff3cd; color: #856404'
    return ''

def color_retrain(val):
    if val == 'Yes': return 'background-color: #fff3cd'
    if val == 'N/A': return 'background-color: #e2e3e5'
    return ''

try:
    styled = (df16.style
              .applymap(color_quality,  subset=['Quality Impact'])
              .applymap(color_retrain,  subset=['Needs Retraining']))
except AttributeError:
    styled = (df16.style
              .map(color_quality, subset=['Quality Impact'])
              .map(color_retrain, subset=['Needs Retraining']))

display(styled)
print("Comparison table rendered. \\u2713")
""")

C17 = md(
"""---
## Section 4 · Cross-Conversation Prefix Caching

### What Is Prefix Caching?

When multiple API requests share a **common prefix** (e.g., the same system prompt, tool list,
or RAG document), the server can compute the KV entries for that prefix **once** and reuse
them across all requests — without recomputing them.

### Why It Matters

| Model | Price: fresh input | Price: cached input | Discount |
|---|---|---|---|
| GPT-5.4 | $2.50 / 1M tokens | $0.25 / 1M tokens | **10×** |
| Claude Sonnet 4.5 | $3.00 / 1M tokens | $0.30 / 1M tokens | **10×** |

Research shows prefix caching can deliver up to **79.6% cost reduction** and **22.9% TTFT reduction**
when the system prompt is long and stable (arXiv 2601.06007).

### Design Rule

```
[STABLE PREFIX — put first]          [VARIABLE SUFFIX — put last]
  System prompt                          User message
  Tool definitions                       Conversation history
  RAG documents                          Current query
  Identity files
         ↑ Cache hit region ↑                ↑ Cache miss ↑
```

The longer the shared prefix, the more tokens are served cheaply from cache.
""")

C18 = code(
"""# ── C18: Prefix Cache Simulator ──────────────────────────────────────────
# Run C02 first

class PrefixCacheSimulator:
    \"\"\"
    Toy prefix-cache simulator using word-level tokenization.
    Tracks which tokens are cache hits (previously-seen prefix) vs misses.
    \"\"\"
    FRESH_COST  = 2.50 / 1_000_000   # $ per fresh token   (GPT-5.4 rate)
    CACHED_COST = 0.25 / 1_000_000   # $ per cached token  (10× discount)

    def __init__(self):
        self.cache = {}   # prefix_hash → list of tokens

    def _tokenize(self, text):
        return text.lower().split()

    def _hash(self, tokens):
        return hashlib.md5(' '.join(tokens).encode()).hexdigest()

    def process(self, system_prompt, user_message):
        if not system_prompt.strip():
            sys_toks = []
        else:
            sys_toks = self._tokenize(system_prompt)
        user_toks = self._tokenize(user_message) if user_message else []
        all_toks  = sys_toks + user_toks

        # Find longest cached prefix (scan from full system prompt down to 1 token)
        hit_len = 0
        for i in range(len(sys_toks), 0, -1):
            h = self._hash(sys_toks[:i])
            if h in self.cache:
                hit_len = i
                break

        # Store system prompt in cache for future requests
        if sys_toks:
            self.cache[self._hash(sys_toks)] = sys_toks

        miss_len = len(all_toks) - hit_len
        cost     = hit_len * self.CACHED_COST + miss_len * self.FRESH_COST
        return {
            'total_tokens': len(all_toks),
            'hit_tokens'  : hit_len,
            'miss_tokens' : miss_len,
            'cost'        : cost,
            'tokens'      : all_toks,
            'hit_mask'    : [True] * hit_len + [False] * miss_len,
        }

# ── Sanity checks ─────────────────────────────────────────────────────────
_sim = PrefixCacheSimulator()
_r1  = _sim.process('Hello world from the system', 'user says hi')
assert _r1['hit_tokens'] == 0, "First call must be all cache miss"
_r2  = _sim.process('Hello world from the system', 'user says bye')
assert _r2['hit_tokens'] == 5, f"Expected 5 hit tokens, got {_r2['hit_tokens']}"
assert _r2['cost'] < _r1['cost'], "Second call must be cheaper"
print(f"Test passed. Second-call savings: ${(_r1['cost'] - _r2['cost']) * 1e6:.3f} per million calls")

# ── Demo with meaningful prompts ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful AI travel assistant with access to real-time flight, hotel, and activity "
    "booking APIs. You always confirm booking details before finalising any reservation. "
    "You are friendly, concise, and prioritise the user's budget and schedule constraints."
)
USER1 = "Book a flight from Taipei to Boston for next Monday"
USER2 = "Book a flight from San Francisco to New York for Friday"

demo_sim = PrefixCacheSimulator()
r1 = demo_sim.process(SYSTEM_PROMPT, USER1)
r2 = demo_sim.process(SYSTEM_PROMPT, USER2)

for label, r in [("Request 1 (cold cache)", r1), ("Request 2 (warm cache)", r2)]:
    print(f"\\n{label}:")
    print(f"  Total tokens : {r['total_tokens']}")
    print(f"  Cache hits   : {r['hit_tokens']} ({r['hit_tokens']/r['total_tokens']*100:.0f}%)")
    print(f"  Cache misses : {r['miss_tokens']}")
    print(f"  Cost         : ${r['cost']*1e6:.4f} per million calls")
""")

C19 = code(
"""# ── C19: Interactive Prefix Cache Visualizer ─────────────────────────────
# Run C02, C18 first

DEFAULT_SYSTEM = (
    "You are a helpful AI travel assistant with access to real-time flight, hotel, and activity "
    "booking APIs. You always confirm booking details before finalising any reservation. "
    "You are friendly, concise, and prioritise the user's budget and schedule constraints."
)
DEFAULT_USER1 = "Book a flight from x to y x=Taipei y=Boston"
DEFAULT_USER2 = "Book a flight from x to y x=San Francisco y=New York"

sys_area   = widgets.Textarea(value=DEFAULT_SYSTEM, description='System:',
                               layout=Layout(width='95%', height='90px'))
user1_area = widgets.Textarea(value=DEFAULT_USER1, description='User 1:',
                               layout=Layout(width='95%'))
user2_area = widgets.Textarea(value=DEFAULT_USER2, description='User 2:',
                               layout=Layout(width='95%'))
btn        = widgets.Button(description='Compute Cache Hits',
                             button_style='primary', icon='search')
out_widget = widgets.Output()

def tokens_to_html(tokens, hit_mask, label, cost, savings):
    spans = []
    for tok, hit in zip(tokens, hit_mask):
        bg  = '#c3e6cb' if hit else '#ffffff'
        bdr = '#28a745' if hit else '#ced4da'
        spans.append(
            f'<span style="background:{bg};border:1px solid {bdr};border-radius:3px;'
            f'padding:2px 5px;margin:2px;font-family:monospace;font-size:12px">{tok}</span>'
        )
    hit_count = sum(hit_mask)
    rate = hit_count / len(tokens) * 100 if tokens else 0
    color = '#28a745' if rate > 0 else '#6c757d'
    return (
        f'<p><b>{label}</b></p>'
        f'<div style="line-height:2.2">{"".join(spans)}</div>'
        f'<p style="margin-top:6px;color:{color}">'
        f'Cache hit rate: <b>{rate:.0f}%</b> ({hit_count}/{len(tokens)} tokens) &nbsp;|&nbsp; '
        f'Cost: <b>${cost*1e6:.4f}</b> per M calls'
        + (f' &nbsp;|&nbsp; Savings vs all-fresh: <b>${savings*1e6:.4f}</b>' if savings > 0 else '')
        + '</p>'
    )

def on_click(b):
    with out_widget:
        clear_output(wait=True)
        try:
            if not sys_area.value.strip():
                display(HTML('<p style="color:red">\\u26a0 System prompt is empty.</p>'))
                return
            sim = PrefixCacheSimulator()
            r1  = sim.process(sys_area.value, user1_area.value)
            r2  = sim.process(sys_area.value, user2_area.value)

            full_cost_r2 = r2['total_tokens'] * PrefixCacheSimulator.FRESH_COST
            savings_r2   = full_cost_r2 - r2['cost']

            display(HTML(
                '<h4>\\U0001f7e9 green = cache hit &nbsp;&nbsp; \\u25a1 white = cache miss</h4>'
                + tokens_to_html(r1['tokens'], r1['hit_mask'], 'Request 1 (cold cache)',
                                 r1['cost'], 0)
                + '<hr style="margin:10px 0">'
                + tokens_to_html(r2['tokens'], r2['hit_mask'], 'Request 2 (warm cache)',
                                 r2['cost'], savings_r2)
            ))
        except Exception as e:
            display(HTML(f'<p style="color:red">Error: {e}</p>'))

btn.on_click(on_click)
display(VBox([sys_area, user1_area, user2_area, btn, out_widget]))
""")

C20 = code(
"""# ── C20: Optional Live API Timing Demo ───────────────────────────────────
# Run C02 first. Requires ANTHROPIC_API_KEY in environment; skips gracefully if absent.

LONG_SYSTEM_PROMPT = \"\"\"
You are an expert AI assistant specialising in machine learning systems, deep learning,
and LLM inference optimisation. You have deep knowledge of the following topics:

Transformer Architecture: Multi-head self-attention, positional encodings, layer normalisation,
feed-forward networks, residual connections. You understand both encoder-only (BERT),
decoder-only (GPT family, Llama, Gemma, Mistral), and encoder-decoder (T5, BART) architectures.

KV Cache and Inference Efficiency: You can explain why KV caching reduces autoregressive
decoding from O(T^2) to O(T) projection operations. You know the memory formula:
bytes_per_token = num_layers x num_kv_heads x head_dim x dtype_bytes x 2.
You understand Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Group-Query
Attention (GQA) and their memory-quality trade-offs. You know that Llama 3 and Gemma 2
use GQA, while DeepSeek uses Multi-Head Latent Attention (MLA) to compress KV heads into
a low-rank latent vector.

Memory Hierarchy: You can explain the HBM (High-Bandwidth Memory) and SRAM (on-chip) hierarchy
on GPU accelerators. You understand that loading the KV cache from HBM to SRAM is the primary
bottleneck in long-context inference, not the FLOP count. Flash Attention minimises HBM reads
by tiling the attention computation in SRAM.

Prompt Caching: You know how prefix caching works at the API level — shared prefixes are
computed once and reused. You know the pricing impact (e.g. 10x discount on cached tokens)
and how to structure prompts with stable content first to maximise cache hit rate.

Hardware: You know GPU generations (A100, H100, H200), their HBM capacities (40-80 GB),
and memory bandwidth. You can estimate context length limits given model weight footprint
and available GPU memory.

When answering, be precise, concise, and pedagogically clear. Use examples and analogies
where appropriate. Prefer concrete numbers over vague statements.
\"\"\".strip()

SHORT_USER = "What is the key benefit of KV Cache in one sentence?"

def measure_ttft(client, system_prompt, user_message):
    \"\"\"Returns Time-To-First-Token in seconds using the Anthropic streaming API.\"\"\"
    start = time.perf_counter()
    first_token_time = None
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=80,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        for text in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            break
    return (first_token_time or time.perf_counter()) - start

assert len(LONG_SYSTEM_PROMPT.split()) >= 400, "System prompt too short to exercise prefix caching"

try:
    import anthropic as _anthropic
    _api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not _api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    _client = _anthropic.Anthropic(api_key=_api_key)
    print("Measuring cold TTFT (first call — no cached prefix)...")
    ttft_cold = measure_ttft(_client, LONG_SYSTEM_PROMPT, SHORT_USER)

    time.sleep(0.5)
    print("Measuring warm TTFT (second call — prefix served from cache)...")
    ttft_warm = measure_ttft(_client, LONG_SYSTEM_PROMPT, SHORT_USER)

    speedup = ttft_cold / ttft_warm if ttft_warm > 0 else float('inf')
    print(f"\\n{'='*45}")
    print(f"Cold TTFT : {ttft_cold:.3f} s")
    print(f"Warm TTFT : {ttft_warm:.3f} s")
    print(f"Speedup   : {speedup:.2f}\\u00d7")
    print(f"{'='*45}")
    print("Note: TTFT variance is high over the network; run 3+ times and average.")
    print("Live API demo complete. \\u2713")

except (ImportError, ValueError) as e:
    print("API key not set \\u2014 skipping live demo; see C18\\u2013C19 for simulation.")
    print(f"Reason: {e}")
except Exception as e:
    print(f"Connection error: {e}")
    print("Check your network and API key, then retry.")
""")

C21 = md(
"""---
## Section 5 · Summary

### Comparison of KV Cache Optimisation Techniques

| Method | Approach | Changes Attention? | Needs Retraining? | Other Cost |
|---|---|:---:|:---:|---|
| **Flash Attention** | Reduce HBM↔SRAM data movement | No | No | Extra compute complexity |
| **KV Cache** | Store computed K/V; reuse at each step | No | No | GPU memory usage |
| **Multi-query Attention** | All queries share one K/V pair | Yes | Yes | May hurt quality |
| **Group-query Attention** | Query groups share a K/V head | Yes | Yes | Minor quality impact |
| **MLA (DeepSeek)** | Compress K/V to low-rank latent | Yes | Yes | Extra projection matrices |
| **Sliding Window Attention** | Limit attention to W recent tokens | Yes | ? | Cannot attend past window |
| **Streaming LLM** | Window + attention sink tokens | Yes | ? | Must keep first few tokens |
| **Pruning KV Cache** | Evict low-attention K/V entries | Yes | No | May hurt quality |

**Key insight:** Flash Attention and KV Cache are **drop-in optimisations** — they don't change the model
or require retraining. All other techniques require architectural changes and retraining.
""")

C22 = code(
"""# ── C22: Styled Summary Table ─────────────────────────────────────────────
# Run C02 first

summary_data = [
    {'Method': 'Flash Attention',
     'Approach': 'Reduce HBM\\u2194SRAM data movement',
     'Changes Attention?': 'No', 'Needs Retraining?': 'No',
     'Other Cost': 'Extra compute + code complexity'},
    {'Method': 'KV Cache',
     'Approach': 'Store computed K/V for reuse at each step',
     'Changes Attention?': 'No', 'Needs Retraining?': 'No',
     'Other Cost': 'GPU memory grows with context'},
    {'Method': 'Multi-query Attention',
     'Approach': 'All queries share one K/V pair',
     'Changes Attention?': 'Yes', 'Needs Retraining?': 'Yes',
     'Other Cost': 'May significantly hurt quality'},
    {'Method': 'Group-query Attention',
     'Approach': 'Query groups share a K/V head',
     'Changes Attention?': 'Yes', 'Needs Retraining?': 'Yes',
     'Other Cost': 'Minor quality impact (Llama/Gemma)'},
    {'Method': 'MLA (DeepSeek)',
     'Approach': 'Compress K/V to low-rank latent',
     'Changes Attention?': 'Yes', 'Needs Retraining?': 'Yes',
     'Other Cost': 'Extra projection matrices'},
    {'Method': 'Sliding Window Attention',
     'Approach': 'Limit attention to W recent tokens',
     'Changes Attention?': 'Yes', 'Needs Retraining?': '?',
     'Other Cost': 'Cannot attend past window'},
    {'Method': 'Streaming LLM',
     'Approach': 'Window + attention sink tokens',
     'Changes Attention?': 'Yes', 'Needs Retraining?': '?',
     'Other Cost': 'Must always keep first few tokens'},
    {'Method': 'Pruning KV Cache',
     'Approach': 'Evict low-attention K/V entries',
     'Changes Attention?': 'Yes', 'Needs Retraining?': 'No',
     'Other Cost': 'May significantly hurt quality'},
]

assert len(summary_data) == 8
kv_row = next(r for r in summary_data if r['Method'] == 'KV Cache')
assert kv_row['Changes Attention?'] == 'No' and kv_row['Needs Retraining?'] == 'No'

df22 = pd.DataFrame(summary_data)

def color_yn(val):
    if val == 'No':  return 'background-color: #d4edda; color: #155724'
    if val == 'Yes': return 'background-color: #f8d7da; color: #721c24'
    if val == '?':   return 'background-color: #fff3cd; color: #856404'
    return ''

try:
    styled22 = (df22.style
                .applymap(color_yn, subset=['Changes Attention?', 'Needs Retraining?'])
                .set_properties(**{'text-align': 'left'})
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('background-color', '#343a40'),
                              ('color', 'white'), ('font-weight', 'bold')]
                }]))
except AttributeError:
    styled22 = (df22.style
                .map(color_yn, subset=['Changes Attention?', 'Needs Retraining?'])
                .set_properties(**{'text-align': 'left'})
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('background-color', '#343a40'),
                              ('color', 'white'), ('font-weight', 'bold')]
                }]))

display(styled22)
print("Summary table rendered. \\u2713")
""")

C23 = md(
"""---
## Key Takeaways

1. **KV Cache** is the foundational inference optimisation: store computed K and V vectors for
   all past tokens, so each new decoding step only projects the **one new token** instead of
   recomputing everything. Cost drops from O(T²) to O(T) in projection operations.

2. **Memory is the bottleneck** for long-context inference, not compute.
   Formula: `bytes/token = num_layers × num_kv_heads × head_dim × dtype_bytes × 2`.
   A single token in Gemma 2 27B (GQA) costs ≈ 0.37 MB.

3. **MQA and GQA** trade a fraction of model quality for proportionally smaller KV caches.
   GQA (used in Llama 3 and Gemma 2) is the practical sweet spot — it cuts the cache
   by H/G without meaningfully degrading model capability.

4. **Prefix Caching** lets you reuse KV entries across requests that share a common prefix.
   Put stable content (system prompt, tools, documents) **first** in your prompt to maximise
   cache hit rate and unlock up to 10× cheaper cached-token pricing.

5. **Flash Attention and KV Cache are drop-in optimisations** — they don't change model
   behaviour and require no retraining. All head-sharing variants (MQA, GQA, MLA) and
   window-based methods (Sliding Window, Streaming LLM) modify the attention mechanism
   and require architectural retraining.

---

### Further Reading

| Paper | Topic |
|---|---|
| [StreamingLLM (arXiv 2309.17453)](https://arxiv.org/abs/2309.17453) | Attention sinks for infinite-length generation |
| [DeepSeek-V2 / MLA (arXiv 2405.04434)](https://arxiv.org/abs/2405.04434) | Multi-Head Latent Attention |
| [Scissorhands (arXiv 2305.17118)](https://arxiv.org/abs/2305.17118) | KV cache pruning via heavy-hitter observation |
| [H2O (arXiv 2306.14048)](https://arxiv.org/abs/2306.14048) | Heavy Hitter Oracle for KV eviction |
| [Prompt Caching Survey (arXiv 2601.06007)](https://arxiv.org/abs/2601.06007) | Cost/TTFT benchmarks across providers |
""")

# ── Assemble notebook ────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": [C01, C02, C03, C04, C05, C06, C07, C08, C09, C10,
               C11, C12, C13, C14, C15, C16, C17, C18, C19, C20,
               C21, C22, C23]
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Notebook written to: {OUT}")
print(f"Total cells: {len(nb['cells'])}")
