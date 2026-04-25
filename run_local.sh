#!/usr/bin/env bash
# run_local.sh — One-shot setup + training for local dev
# Usage:
#   chmod +x run_local.sh
#   ./run_local.sh               # uses sample data
#   ./run_local.sh <path/to/bts_data.csv>  # uses real BTS data

set -e

INPUT="${1:-data/sample_flights.csv}"
MODEL_DIR="models"
PLOTS_DIR="plots"

echo "============================================================"
echo " DATA-228 · Flight Delay ML Training (local mode)"
echo "============================================================"

# 1. Install Python dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -q -r requirements.txt

# 2. Generate sample data if no real data provided and file missing
if [ ! -f "$INPUT" ] && [ "$INPUT" = "data/sample_flights.csv" ]; then
    echo ""
    echo "[2/4] Generating synthetic sample data (100k rows)..."
    python src/training/generate_sample_data.py
else
    echo ""
    echo "[2/4] Using input: $INPUT"
fi

# 3. Run EDA
echo ""
echo "[3/4] Running EDA & generating plots..."
mkdir -p "$PLOTS_DIR"
python src/training/eda_analysis.py --input "$INPUT" --output "$PLOTS_DIR"

# 4. Train models
echo ""
echo "[4/4] Training LR + GBT with CrossValidator..."
mkdir -p "$MODEL_DIR"
python src/training/train_local.py \
    --input "$INPUT" \
    --model-dir "$MODEL_DIR" \
    --cv-folds 3

echo ""
echo "============================================================"
echo " Done! Artifacts:"
echo "   Models  → $MODEL_DIR/"
echo "   Plots   → $PLOTS_DIR/"
echo "   Metrics → $MODEL_DIR/metrics.json"
echo "============================================================"
