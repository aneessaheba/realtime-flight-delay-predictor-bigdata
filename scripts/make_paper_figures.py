"""
make_paper_figures.py
---------------------
Draw the paper's figures at their printed size (IEEE two-column: 3.5 in
column, 7.16 in text width), so font sizes below are the sizes on the page.

  architecture.png            pipeline diagram (full text width)
  fig_class_distribution.png  from logs/eda_full_corpus.json
  fig_delay_by_carrier.png    from logs/eda_full_corpus.json
  fig_correlation_heatmap.png from logs/eda_full_corpus.json

The screen-size versions are drawn by generate_architecture.py and
src/training/plot_eda_full_corpus.py.

Usage:
    python scripts/make_paper_figures.py --eda logs/eda_full_corpus.json --output paper
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

COLUMN_W = 3.4
TEXT_W = 7.0
DPI = 300

plt.rcParams.update({
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# The diagram is plain black on white in a Times-style serif, to match the paper's text.
DIAGRAM_FONT = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "mathtext.fontset": "stix"}
HEATMAP_LABELS = {
    "DAY_OF_WEEK": "DAY_OF_WEEK", "CRS_DEP_TIME": "CRS_DEP_TIME", "DEP_DELAY": "DEP_DELAY",
    "TAXI_OUT": "TAXI_OUT", "CRS_ELAPSED_TIME": "CRS_ELAPSED", "DISTANCE": "DISTANCE",
    "MONTH": "MONTH", "label": "label",
}


def save(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved: {path}")


def box(ax, x, y, w, h, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                linewidth=0.7, edgecolor="black", facecolor="white", zorder=3))
    top = y + h - 0.16
    ax.text(x + w / 2, top, title, fontsize=7.5, fontweight="bold", color="black",
            ha="center", va="center", zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top - 0.19 - i * 0.155, line, fontsize=6.5, color="black",
                ha="center", va="center", zorder=4)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7, mutation_scale=7,
                                shrinkA=0, shrinkB=0))


def plot_architecture(out_dir: str) -> None:
    with plt.rc_context(DIAGRAM_FONT):
        _draw_architecture(out_dir)


def _draw_architecture(out_dir: str) -> None:
    # Numbers are from logs/train_chronological_full_corpus.json,
    # logs/batch_chrono_gbt_*.json and logs/streaming_eval_2024_trigger0s.json.
    fig, ax = plt.subplots(figsize=(TEXT_W, 3.0))
    ax.set_xlim(0, TEXT_W)
    ax.set_ylim(0.2, 3.2)
    ax.axis("off")

    box(ax, 0.05, 2.35, 1.35, 0.8, "BTS on-time data",
        ["2021–23: 19.15M flights", "2024: held-out year"])
    box(ax, 1.6, 2.35, 1.6, 0.8, "Ingestion (PySpark)",
        ["label: ARR_DELAY > 15 min", "Parquet on HDFS"])
    box(ax, 3.4, 2.35, 1.65, 0.8, "Training",
        ["GBT + LR, split by month", "params and τ set on val."])
    box(ax, 5.25, 2.35, 1.7, 0.8, "Serialized pipeline",
        ["feature SQL → index → impute", "→ assemble → scale → GBT", "test AUC-ROC 0.968"])

    box(ax, 1.6, 1.35, 2.6, 0.7, "Batch inference",
        ["2021–23: 19.15M rows at 270K/s", "full 2024 (6.97M): AUC-ROC 0.968"])

    box(ax, 0.05, 0.25, 1.35, 0.8, "Kafka producer",
        ["250K 2024 flights", "reservoir-sampled", "~360 events/s"])
    box(ax, 1.6, 0.25, 3.0, 0.8, "Structured Streaming consumer",
        ["Bloom-filter dedup · micro-batches", "2024 AUC-ROC 0.968 · F1 (pos.) 0.866",
         "latency p50 2.2 s · p99 6.3 s"])

    box(ax, 5.35, 0.25, 1.6, 1.8, "Evaluation",
        ["accuracy vs. realized", "2024 outcomes", "batch/stream parity", "(max |Δp| = 0.0)",
         "latency distribution", "DP metric release", "LSH consistency check"])

    arrow(ax, 1.4, 2.75, 1.6, 2.75)
    arrow(ax, 3.2, 2.75, 3.4, 2.75)
    arrow(ax, 5.05, 2.75, 5.25, 2.75)
    arrow(ax, 2.4, 2.35, 2.4, 2.05)            # ingested data -> batch
    arrow(ax, 5.3, 2.35, 4.2, 1.85)            # pipeline -> batch
    arrow(ax, 5.45, 2.35, 4.45, 1.05)          # pipeline -> streaming
    arrow(ax, 0.725, 2.35, 0.725, 1.05)        # 2024 CSVs -> producer
    arrow(ax, 1.4, 0.65, 1.6, 0.65)
    arrow(ax, 4.2, 1.55, 5.35, 1.55)
    arrow(ax, 4.6, 0.65, 5.35, 0.65)
    save(fig, out_dir, "architecture.png")


def plot_class_distribution(d: dict, out_dir: str) -> None:
    counts = [d["class_counts"]["on_time"], d["class_counts"]["delayed"]]
    fig, ax = plt.subplots(figsize=(COLUMN_W, 1.9))
    bars = ax.bar(["On-time", "Delayed (>15 min)"], counts, color=["#4CAF50", "#F44336"], width=0.55)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,} ({100 * val / sum(counts):.2f}%)", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0"))
    ax.set_ylabel("Flights")
    save(fig, out_dir, "fig_class_distribution.png")


def plot_delay_by_carrier(d: dict, out_dir: str) -> None:
    rows = d["by_carrier"]
    fig, ax = plt.subplots(figsize=(COLUMN_W, 1.9))
    ax.bar([r["OP_UNIQUE_CARRIER"] for r in rows], [r["delay_rate_pct"] for r in rows], color="#FF9800")
    ax.axhline(d["delayed_pct"], color="#555555", lw=0.8, ls="--")
    ax.text(len(rows) - 0.5, d["delayed_pct"] + 0.6, f"all carriers {d['delayed_pct']:.1f}%",
            ha="right", va="bottom", fontsize=6.5, color="#555555")
    ax.set_ylabel("Flights delayed (%)")
    ax.set_xlabel("Carrier (BTS code)")
    ax.tick_params(axis="x", labelrotation=90)
    ax.margins(x=0.01)
    save(fig, out_dir, "fig_delay_by_carrier.png")


def plot_correlation_heatmap(d: dict, out_dir: str) -> None:
    cols = list(HEATMAP_LABELS)
    corr = pd.DataFrame(d["correlation"]).loc[cols, cols].rename(index=HEATMAP_LABELS, columns=HEATMAP_LABELS)
    fig, ax = plt.subplots(figsize=(COLUMN_W, 3.0))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                square=True, linewidths=0.4, annot_kws={"fontsize": 5.5},
                cbar_kws={"shrink": 0.7})
    ax.tick_params(axis="both", labelsize=6)
    ax.collections[0].colorbar.ax.tick_params(labelsize=6)
    save(fig, out_dir, "fig_correlation_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the paper's figures at print size.")
    parser.add_argument("--eda", default="logs/eda_full_corpus.json")
    parser.add_argument("--output", default="paper")
    args = parser.parse_args()

    with open(args.eda, encoding="utf-8") as fh:
        d = json.load(fh)
    os.makedirs(args.output, exist_ok=True)
    plot_architecture(args.output)
    plot_class_distribution(d, args.output)
    plot_delay_by_carrier(d, args.output)
    plot_correlation_heatmap(d, args.output)


if __name__ == "__main__":
    main()
