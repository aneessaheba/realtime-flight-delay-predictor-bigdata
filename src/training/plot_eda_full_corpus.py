"""
plot_eda_full_corpus.py
-----------------------
Draw the paper's EDA figures (class distribution, delay rate by carrier,
correlation matrix, plus month / day-of-week) from the full-corpus
aggregates written by eda_full_corpus.py. Styling matches eda_analysis.py.

Usage:
    python src/training/plot_eda_full_corpus.py \
        --input logs/eda_full_corpus.json --output plots/full_corpus
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

DAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
HEATMAP_COLS = ["DAY_OF_WEEK", "CRS_DEP_TIME", "DEP_DELAY", "TAXI_OUT", "CRS_ELAPSED_TIME", "DISTANCE", "MONTH", "label"]


def save(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_class_distribution(d: dict, out_dir: str) -> None:
    counts = [d["class_counts"]["on_time"], d["class_counts"]["delayed"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["On-time", "Delayed (>15m)"], counts, color=["#4CAF50", "#F44336"], edgecolor="white")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}\n({100 * val / sum(counts):.2f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.set_title("Class Distribution (2021–2023)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Flights")
    ax.set_xlabel("Label")
    save(fig, out_dir, "01_class_distribution.png")


def plot_delay_by_month(d: dict, out_dir: str) -> None:
    rows = d["by_month"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([int(r["MONTH"]) for r in rows], [r["delay_rate_pct"] for r in rows], color="#2196F3", edgecolor="white")
    ax.set_xticks(range(1, 13))
    ax.set_title("Delay Rate by Month (2021–2023)", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Flights Delayed (>15m)")
    ax.set_xlabel("Month")
    save(fig, out_dir, "02_delay_by_month.png")


def plot_delay_by_day_of_week(d: dict, out_dir: str) -> None:
    rows = d["by_day_of_week"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([DAY_NAMES[int(r["DAY_OF_WEEK"])] for r in rows], [r["delay_rate_pct"] for r in rows],
           color="#9C27B0", edgecolor="white")
    ax.set_title("Delay Rate by Day of Week (2021–2023)", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Flights Delayed (>15m)")
    ax.set_xlabel("Day")
    save(fig, out_dir, "03_delay_by_day.png")


def plot_delay_by_carrier(d: dict, out_dir: str) -> None:
    rows = d["by_carrier"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([r["OP_UNIQUE_CARRIER"] for r in rows], [r["delay_rate_pct"] for r in rows],
           color="#FF9800", edgecolor="white")
    ax.set_title("Delay Rate by Carrier (2021–2023)", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Flights Delayed (>15m)")
    ax.set_xlabel("Carrier")
    save(fig, out_dir, "04_delay_by_carrier.png")


def plot_correlation_heatmap(d: dict, out_dir: str) -> None:
    corr = pd.DataFrame(d["correlation"]).loc[HEATMAP_COLS, HEATMAP_COLS]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix (2021–2023)", fontsize=13, fontweight="bold")
    save(fig, out_dir, "06_correlation_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot full-corpus EDA figures.")
    parser.add_argument("--input", default="logs/eda_full_corpus.json")
    parser.add_argument("--output", default="plots/full_corpus")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        d = json.load(fh)
    os.makedirs(args.output, exist_ok=True)
    plot_class_distribution(d, args.output)
    plot_delay_by_month(d, args.output)
    plot_delay_by_day_of_week(d, args.output)
    plot_delay_by_carrier(d, args.output)
    plot_correlation_heatmap(d, args.output)


if __name__ == "__main__":
    main()
