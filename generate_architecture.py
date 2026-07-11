import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
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

# ── TOP ROW ────────────────────────────────────────────────────────────────────
box(ax, 0.3, 7.2, 2.2, 2.2, 'data', 'BTS data',
    ['2018–2024 CSVs', 'transtats.bts.gov'])

box(ax, 3.0, 7.2, 2.5, 2.2, 'ingest', 'Ingestion',
    ['19M rows → HDFS', 'Parquet, by yr/mo'])

box(ax, 6.2, 7.2, 2.8, 2.2, 'model', 'Training',
    ['GBT + LR', '3.8M rows (20%)'])

box(ax, 10.0, 7.2, 4.2, 2.2, 'model', 'GBT Pipeline',
    ['Serialized → HDFS', 'AUC-ROC 0.9386  ·  F1 0.9029'])

# ── BATCH ROW ──────────────────────────────────────────────────────────────────
label(ax, 0.3, 5.5, 'Batch')
box(ax, 3.0, 4.6, 6.5, 2.0, 'batch', 'Batch Inference',
    ['537,183 records  ·  340K rec/s', 'AUC-ROC 0.9386  ·  F1 0.9029  ·  Acc 0.901'])

# ── STREAM ROW ─────────────────────────────────────────────────────────────────
label(ax, 0.3, 3.0, 'Stream')
box(ax, 1.5, 2.1, 2.8, 2.0, 'stream', 'Kafka Producer',
    ['~100–500 msg/sec', 'Replay 2024 data'])

box(ax, 5.0, 2.1, 3.2, 2.0, 'stream', 'Streaming Consumer',
    ['Micro-batch 10s', '62,815 preds  ·  p50 5.2s'])

# ── SMOKE TEST ─────────────────────────────────────────────────────────────────
box(ax, 3.0, 0.3, 4.5, 1.4, 'valid', 'Smoke Test',
    ['Kafka · HDFS · Spark · predictions verified'])

# ── BENCHMARK ──────────────────────────────────────────────────────────────────
box(ax, 10.0, 2.1, 4.2, 4.1, 'eval', 'Benchmark',
    ['DP metrics (ε=1.0, Laplace)',
     'LSH anomaly detection',
     'Batch vs streaming report'])

# ── ARROWS ─────────────────────────────────────────────────────────────────────
# top row
arrow(ax, 2.5,  8.3, 3.0,  8.3)
arrow(ax, 5.5,  8.3, 6.2,  8.3)
arrow(ax, 9.0,  8.3, 10.0, 8.3)

# GBT pipeline down to batch inference
arrow(ax, 12.1, 7.2, 9.5, 6.6)

# GBT pipeline down to streaming consumer
arrow(ax, 12.1, 7.2, 8.0, 4.1, rad=-0.3)

# ingestion → batch (data path)
arrow(ax, 4.25, 7.2, 4.25, 6.6)

# kafka → streaming
arrow(ax, 4.3,  3.1, 5.0,  3.1)

# streaming → benchmark
arrow(ax, 8.2,  3.1, 10.0, 3.5)

# batch → benchmark
arrow(ax, 9.5,  5.6, 10.0, 5.0)

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
ly = 0.62
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
