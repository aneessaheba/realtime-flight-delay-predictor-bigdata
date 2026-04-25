"""
eda_analysis.py
---------------
Exploratory Data Analysis for the BTS flight delay dataset.
Reads the local CSV (or Parquet) and produces:
  - Class distribution plot
  - Delay distribution by carrier, month, day-of-week
  - Correlation heatmap of numeric features
  - Missing-value summary

Run:
    python src/eda_analysis.py --input data/sample_flights.csv
    python src/eda_analysis.py --input data/sample_flights.csv --output plots/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DELAY_THRESHOLD = 15.0
NUMERIC_COLS = [
    "DAY_OF_WEEK", "CRS_DEP_TIME", "DEP_DELAY",
    "CRS_ELAPSED_TIME", "DISTANCE", "MONTH",
]


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet") or os.path.isdir(path):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df["label"] = (df["ARR_DELAY"] > DELAY_THRESHOLD).astype(int)
    return df


def print_basic_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Shape           : {df.shape}")
    print(f"  Columns         : {list(df.columns)}")
    print(f"\n  Class balance:")
    vc = df["label"].value_counts()
    total = len(df)
    for label, count in vc.items():
        name = "Delayed (>15m)" if label == 1 else "On-time"
        print(f"    {name:20s}: {count:>8,}  ({100*count/total:.1f}%)")

    print("\n  Missing values per column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("    None")
    else:
        for col, cnt in missing.items():
            print(f"    {col:30s}: {cnt:>8,}  ({100*cnt/total:.1f}%)")

    print("\n  Numeric feature summary:")
    print(df[NUMERIC_COLS].describe().to_string())
    print("=" * 60)


def plot_class_distribution(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(["On-time", "Delayed (>15m)"], counts.values,
                  color=["#4CAF50", "#F44336"], edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    plt.tight_layout()
    path = os.path.join(out_dir, "01_class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_delay_by_month(df: pd.DataFrame, out_dir: str) -> None:
    monthly = df.groupby("MONTH")["label"].mean().reset_index()
    monthly.columns = ["Month", "Delay Rate"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(monthly["Month"], monthly["Delay Rate"] * 100, color="#2196F3", edgecolor="white")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_title("Delay Rate by Month", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Flights Delayed (>15m)")
    ax.set_xlabel("Month")
    plt.tight_layout()
    path = os.path.join(out_dir, "02_delay_by_month.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_delay_by_day_of_week(df: pd.DataFrame, out_dir: str) -> None:
    day_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu",
                 5: "Fri", 6: "Sat", 7: "Sun"}
    daily = df.groupby("DAY_OF_WEEK")["label"].mean().reset_index()
    daily["Day"] = daily["DAY_OF_WEEK"].map(day_names)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(daily["Day"], daily["label"] * 100, color="#9C27B0", edgecolor="white")
    ax.set_title("Delay Rate by Day of Week", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Flights Delayed (>15m)")
    ax.set_xlabel("Day")
    plt.tight_layout()
    path = os.path.join(out_dir, "03_delay_by_day.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_delay_by_carrier(df: pd.DataFrame, out_dir: str) -> None:
    carrier_stats = (
        df.groupby("OP_UNIQUE_CARRIER")["label"]
        .agg(["mean", "count"])
        .reset_index()
    )
    carrier_stats.columns = ["Carrier", "Delay Rate", "Flights"]
    carrier_stats = carrier_stats[carrier_stats["Flights"] >= 100]
    carrier_stats = carrier_stats.sort_values("Delay Rate", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(carrier_stats["Carrier"], carrier_stats["Delay Rate"] * 100,
           color="#FF9800", edgecolor="white")
    ax.set_title("Delay Rate by Carrier", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Flights Delayed (>15m)")
    ax.set_xlabel("Carrier")
    plt.tight_layout()
    path = os.path.join(out_dir, "04_delay_by_carrier.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_arr_delay_distribution(df: pd.DataFrame, out_dir: str) -> None:
    data = df["ARR_DELAY"].dropna()
    data = data[(data >= -60) & (data <= 300)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(data, bins=100, color="#607D8B", edgecolor="none", alpha=0.8)
    ax.axvline(DELAY_THRESHOLD, color="#F44336", linestyle="--", linewidth=2,
               label=f"Threshold ({DELAY_THRESHOLD}m)")
    ax.set_title("Arrival Delay Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Arrival Delay (minutes)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "05_arr_delay_dist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    cols = NUMERIC_COLS + ["label"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "06_correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_dep_time_delay(df: pd.DataFrame, out_dir: str) -> None:
    """Delay rate by hour of day (extracted from CRS_DEP_TIME)."""
    df = df.copy()
    df["dep_hour"] = df["CRS_DEP_TIME"] // 100
    df = df[df["dep_hour"].between(0, 23)]
    hourly = df.groupby("dep_hour")["label"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly["dep_hour"], hourly["label"] * 100,
            marker="o", color="#E91E63", linewidth=2, markersize=5)
    ax.set_title("Delay Rate by Departure Hour", fontsize=13, fontweight="bold")
    ax.set_xlabel("Departure Hour (CRS)")
    ax.set_ylabel("% Flights Delayed (>15m)")
    ax.set_xticks(range(0, 24))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "07_delay_by_hour.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="EDA for flight delay dataset")
    parser.add_argument("--input", default="data/sample_flights.csv",
                        help="Path to input CSV or Parquet")
    parser.add_argument("--output", default="plots/",
                        help="Directory to save plots")
    args = parser.parse_args()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from: {args.input}")
    df = load_data(args.input)

    print_basic_stats(df)

    print("\nGenerating plots...")
    plot_class_distribution(df, out_dir)
    plot_arr_delay_distribution(df, out_dir)
    plot_delay_by_month(df, out_dir)
    plot_delay_by_day_of_week(df, out_dir)
    plot_delay_by_carrier(df, out_dir)
    plot_dep_time_delay(df, out_dir)
    plot_correlation_heatmap(df, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
