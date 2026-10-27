import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(-0.8, 10)
ax.axis('off')
fig.patch.set_facecolor('#f0ede8')
ax.set_facecolor('#f0ede8')

# Colors
C = {
    'data':      ('#b0b0b0', '#e8e8e8'),   # grey
    'ingest':    ('#3a9e7e', '#d6efe8'),   # teal
    'model':     ('#7b6fc4', '#e2dff5'),   # purple
    'batch':     ('#5b7fc4', '#d9e4f5'),   # blue
    'stream':    ('#c49a3a', '#f5ebcf'),   # tan/orange
    'eval':      ('#c47b6f', '#f5ddd9'),   # salmon
    'valid':     ('#7a9e3a', '#e4f0d0'),   # olive green
}

def box(ax, x, y, w, h, ckey, title, lines=[]):
    border, fill = C[ckey]
    # drop shadow
    ax.add_patch(FancyBboxPatch((x+0.06, y-0.06), w, h,
        boxstyle="round,pad=0,rounding_size=0.25",
        linewidth=0, facecolor='#c0b8b0', alpha=0.4, zorder=2))
    # main box
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.25",
        linewidth=2, edgecolor=border, facecolor=fill, zorder=3))
    # title
    ty = y + h - 0.45 if lines else y + h/2
    ax.text(x + w/2, ty, title,
            fontsize=11, color=border, fontweight='bold',
            ha='center', va='center', zorder=4)
    # subtitle lines
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.85 - i*0.38, line,
                fontsize=8.5, color='#444444',
                ha='center', va='center', zorder=4)

def arrow(ax, x1, y1, x2, y2, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color='#666666', lw=1.8,
            connectionstyle=f'arc3,rad={rad}',
            mutation_scale=16),
        zorder=5)

def label(ax, x, y, text):
    ax.text(x, y, text, fontsize=10, color='#555555',
            va='center', ha='left', style='italic')

# Numbers below are from logs/train_chronological_full_corpus.json,
# logs/batch_chrono_gbt_*.json and logs/streaming_eval_2024_trigger0s.json.

# ── TOP ROW ────────────────────────────────────────────────────────────────────
box(ax, 0.3, 7.2, 2.4, 2.2, 'data', 'BTS data',
    ['2021–2023: model', '2024: held-out replay', 'transtats.bts.gov'])

box(ax, 3.1, 7.2, 2.7, 2.2, 'ingest', 'Ingestion',
    ['19.15M rows (2021–23)', '6.97M rows (2024)', 'HDFS Parquet, yr/mo'])

box(ax, 6.2, 7.2, 3.2, 2.2, 'model', 'Training',
    ['GBT + LR, month-wise split', 'train Jan 21–Feb 23', 'val Mar–Jul 23 · test Aug–Dec 23'])

box(ax, 9.8, 7.2, 4.9, 2.2, 'model', 'Serialized Pipeline (one artifact)',
    ['Feature SQL → Indexer → Imputer →', 'Assembler → Scaler → GBT',
     'Test AUC-ROC 0.968 · F1 (delayed) 0.849'])

# ── BATCH ROW ──────────────────────────────────────────────────────────────────
label(ax, 0.3, 5.5, 'Batch')
box(ax, 3.1, 4.6, 6.3, 2.0, 'batch', 'Batch Inference',
    ['19.15M records · 270K rec/s',
     'Full 2024 year: AUC-ROC 0.968 · Acc 0.924'])

# ── STREAM ROW ─────────────────────────────────────────────────────────────────
label(ax, 0.3, 3.0, 'Stream')
box(ax, 1.3, 2.1, 3.0, 2.0, 'stream', 'Kafka Producer',
    ['2024 replay, 250K sampled', '~350–380 msg/s'])

box(ax, 4.8, 2.1, 4.6, 2.0, 'stream', 'Streaming Consumer',
    ['Bloom dedup · micro-batches', 'p50 2.2 s · p99 6.3 s',
     '2024 AUC-ROC 0.968 · Acc 0.947'])

# ── SMOKE TEST ─────────────────────────────────────────────────────────────────
box(ax, 3.1, 0.3, 4.5, 1.4, 'valid', 'Smoke Test',
    ['Kafka · HDFS · Spark · predictions verified'])

# ── BENCHMARK ──────────────────────────────────────────────────────────────────
box(ax, 9.8, 1.6, 4.9, 4.6, 'eval', 'Evaluation',
    ['Batch–stream parity: 0 mismatches',
     '(244K records, max |Δp| = 0.0)',
     'Accuracy vs. realised 2024 outcomes',
     'DP release (Laplace, ε = 1 / metric)',
     'LSH prediction-consistency check'])

# ── ARROWS ─────────────────────────────────────────────────────────────────────
# top row
arrow(ax, 2.7,  8.3, 3.1,  8.3)
arrow(ax, 5.8,  8.3, 6.2,  8.3)
arrow(ax, 9.4,  8.3, 9.8,  8.3)

# pipeline down to batch inference
arrow(ax, 12.25, 7.2, 9.4, 6.4)

# pipeline down to streaming consumer
arrow(ax, 11.0, 7.2, 8.4, 4.1, rad=-0.25)

# ingestion → batch (data path)
arrow(ax, 4.45, 7.2, 4.45, 6.6)

# raw 2024 CSVs → producer
arrow(ax, 1.5, 7.2, 2.2, 4.1)

# kafka → streaming
arrow(ax, 4.3,  3.1, 4.8,  3.1)

# streaming → evaluation
arrow(ax, 9.4,  3.1, 9.8,  3.1)

# batch → evaluation
arrow(ax, 9.4,  5.6, 9.8,  5.6)

# ── LEGEND ─────────────────────────────────────────────────────────────────────
legend_items = [
    ('data',   'Data'),
    ('ingest', 'Ingestion'),
    ('model',  'Training / Model'),
    ('batch',  'Batch'),
    ('stream', 'Streaming'),
    ('eval',   'Evaluation'),
    ('valid',  'Validation'),
]
lx = 0.3
ly = -0.35
ax.text(lx, ly, 'Legend:', fontsize=9, color='#555555',
        fontweight='bold', va='center')
for i, (key, name) in enumerate(legend_items):
    border, fill = C[key]
    bx = lx + 1.1 + i * 2.1
    rect = FancyBboxPatch((bx, ly - 0.18), 0.32, 0.36,
        boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=1.5, edgecolor=border, facecolor=fill, zorder=3)
    ax.add_patch(rect)
    ax.text(bx + 0.45, ly, name, fontsize=8.5, color='#444444', va='center')

plt.tight_layout(pad=0.5)
plt.savefig('architecture.png', dpi=180, bbox_inches='tight',
            facecolor='#f0ede8', edgecolor='none')
print("Saved: architecture.png")
