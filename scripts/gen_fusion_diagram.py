"""Generate δ-mem + Taiji OS fusion architecture diagram."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('C:/Users/1/WorkBuddy/2026-05-28-task-12/taiji-os-core/docs/figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# ─── Title ───
ax.text(8, 9.5, u'Taiji OS + δ-mem Fusion Architecture', fontsize=16, fontweight='bold',
        ha='center', va='center', color='#1a1a2e')
ax.text(8, 9.1, u'L1 Hot Cache (δ-mem)  →  L2 Cold Storage (Taiji WorldModel)  →  Φ-Gate Archival',
        fontsize=10, ha='center', va='center', color='#555', style='italic')

# ─── Helpers ───
def draw_box(ax, x, y, w, h, label, color, fontsize=9, sublabel=''):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='#333', linewidth=1.5,
                          alpha=0.92, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.05, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', zorder=3)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.3, sublabel, ha='center', va='center', fontsize=7,
                color='#ddd', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, color='#666', lw=1.5, style='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=style),
                zorder=1)

def label_arrow(ax, x1, y1, x2, y2, text, color='#555', fontsize=7):
    ax.annotate(text, xy=((x1+x2)/2, (y1+y2)/2),
                xytext=(0, -12), textcoords='offset points',
                ha='center', va='top', fontsize=fontsize, color=color,
                style='italic', zorder=4)
    draw_arrow(ax, x1, y1, x2, y2, color=color, lw=1)

# ─── LAYER 0: User Input ───
draw_box(ax, 5.5, 8.1, 5, 0.6, u'User Input / Environment', '#2C3E50', fontsize=10)
draw_arrow(ax, 8, 8.1, 8, 7.3, '#2C3E50', lw=2)

# ─── LAYER 1: δ-mem (Model-Internal, Blue) ───
rect_l1 = plt.Rectangle((0.5, 4.5), 6, 2.5, facecolor='#E8F4F8', edgecolor='#2980B9',
                         linewidth=2, linestyle='--', alpha=0.3, zorder=0)
ax.add_patch(rect_l1)
ax.text(3.5, 6.85, u'L1: Hot Cache (δ-mem) — Model Internal', fontsize=9,
        fontweight='bold', ha='center', color='#2980B9', zorder=1)

draw_box(ax, 1, 5.8, 2.2, 0.7, u'δ-mem Read', '#3498DB', fontsize=9, sublabel='r = S q^m')
draw_box(ax, 3.5, 5.8, 2.2, 0.7, u'Attention Δ', '#3498DB', fontsize=9, sublabel='Q+dQ, K, V -> O+dO')
draw_box(ax, 1.5, 4.8, 4, 0.6, u'S in R^(8x8) (64 floats)', '#5DADE2', fontsize=9,
         sublabel='Delta Rule: S_t = λ S_{t-1} + β(v-Sk)k^T')

draw_arrow(ax, 3.2, 6.15, 3.5, 6.15, '#2980B9')
draw_arrow(ax, 3.5, 5.8, 3.5, 5.4, '#2980B9', lw=1)
draw_box(ax, 1.5, 4.0, 4, 0.55, u'Frozen LLM Backbone', '#1B4F72', fontsize=8)

# ─── LAYER 2: Taiji OS (Model-External, Orange) ───
rect_l2 = plt.Rectangle((7.5, 4.5), 8, 3.5, facecolor='#FDEBD0', edgecolor='#E67E22',
                         linewidth=2, linestyle='--', alpha=0.3, zorder=0)
ax.add_patch(rect_l2)
ax.text(11.5, 8.0, u'L2: Taiji OS Kernel — Model External', fontsize=9,
        fontweight='bold', ha='center', color='#E67E22', zorder=1)

# Top row: gates
draw_box(ax, 7.8, 7.0, 2.5, 0.65, u'Φ-Gate', '#E67E22', fontsize=9, sublabel='cos_sim(psi, prev) < Φ_th?')
draw_box(ax, 10.8, 7.0, 2.5, 0.65, u'D-Core', '#D35400', fontsize=9, sublabel='Semantic Contradiction')
draw_box(ax, 13.8, 7.0, 1.5, 0.65, u'SCS', '#D35400', fontsize=9, sublabel='Drift detect')

# Middle: WorldModel + Episodic
draw_box(ax, 8.5, 5.8, 3, 0.7, u'WorldModel (psi, epsilon)', '#F39C12', fontsize=9,
         sublabel='psi: 384-1536d semantic field')
draw_box(ax, 12, 5.8, 3, 0.7, u'Episodic Memory', '#E67E22', fontsize=9, sublabel='FAISS / Walrus Index')

# Bottom: Continuation
draw_box(ax, 9, 4.8, 5.5, 0.7, u'Continuation Snapshot: <psi, S, C, S_anchor>', '#C0392B',
         fontsize=9, sublabel='suspend() / resume(k) — First-class OS abstraction')

# ─── Connections ───
# User -> δ-mem
draw_arrow(ax, 6.5, 8.4, 3.5, 6.6, '#2980B9', lw=1.5)
label_arrow(ax, 6.5, 8.4, 3.5, 6.6, 'input tokens')

# LLM Output -> Taiji OS
draw_arrow(ax, 5.5, 4.3, 7.5, 7.3, '#E67E22', lw=1.5)
label_arrow(ax, 5.5, 4.3, 7.5, 7.3, 'Output + dO')

# Gate chain
draw_arrow(ax, 9.1, 7.32, 10.8, 7.32, '#E67E22', lw=1)
draw_arrow(ax, 12.1, 7.32, 13.8, 7.32, '#D35400', lw=1)

# Gate -> WorldModel
draw_arrow(ax, 9.6, 7.0, 10, 6.55, '#E67E22', lw=1.3)
label_arrow(ax, 9.6, 7.0, 10, 6.55, 'psi update')

# WorldModel -> Episodic
draw_arrow(ax, 11.5, 5.8, 13.5, 5.8, '#E67E22', lw=1.3)
label_arrow(ax, 11.5, 5.8, 13.5, 5.8, 'Φ high -> flush to Walrus')

# WorldModel -> Continuation
draw_arrow(ax, 11, 5.8, 11, 5.55, '#C0392B', lw=1)
label_arrow(ax, 11, 5.8, 11, 5.55, 'suspend')

# KEY: δ-mem S -> Continuation Snapshot (purple dashed)
draw_arrow(ax, 3.5, 4.8, 8.5, 5.2, '#8E44AD', lw=2, style='--')
ax.annotate('Serialize S\ninto Continuation C', xy=(6, 4.95), fontsize=7, color='#8E44AD',
            ha='center', va='top', fontweight='bold', zorder=4,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3E5F5', alpha=0.9))

# ─── LAYER 3: Cross-Model ───
draw_box(ax, 3, 2.5, 10, 1.0, '', '#F4F6F6', fontsize=8)
ax.text(8, 3.1, u'Cross-Model Continuation Resume: Claude / GPT / DeepSeek / Llama',
        ha='center', va='center', fontsize=10, fontweight='bold', color='#1a1a2e')
ax.text(8, 2.7, u'Walrus Memory stores <psi, S, C> — Re-anchor on any backbone via Φ-Gate consistency check',
        ha='center', va='center', fontsize=8, color='#666', style='italic')

draw_arrow(ax, 11.5, 4.8, 8, 3.55, '#C0392B', lw=2)
label_arrow(ax, 11.5, 4.8, 8, 3.55, 'resume(k) -> load Snapshot')

# ─── Legend ───
legend_y = 1.8
items = [
    ('#3498DB', u'L1 δ-mem (Hot Cache)', u'O(r^2)/token, 0.12% params'),
    ('#E67E22', u'L2 Taiji OS (Cold Storage)', 'suspend/resume/migrate'),
    ('#8E44AD', 'S serialization link', u'δ-mem S -> Continuation C'),
]
for i, (c, label, desc) in enumerate(items):
    x = 1.5 + i * 5
    rect = plt.Rectangle((x, legend_y), 0.3, 0.25, facecolor=c, edgecolor='#333', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 0.5, legend_y + 0.12, f'{label}  ({desc})', fontsize=7, va='center', color='#444')

# ─── Quote ───
ax.text(8, 1.2, u'"δ-mem gives AI permanent memory. Taiji OS gives AI a soul container — '
        u'alive, interruptible, migratable, rebornable."',
        ha='center', fontsize=9, color='#8E44AD', fontweight='bold', style='italic')

ax.text(8, 0.7, u'Taiji OS v4.3.1  |  δ-mem arXiv:2605.12357 (Wu et al., 2026)',
        ha='center', fontsize=8, color='#999')

plt.tight_layout()
out = 'C:/Users/1/WorkBuddy/2026-05-28-task-12/taiji-os-core/docs/figures/delta_mem_fusion.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved: {out}')
