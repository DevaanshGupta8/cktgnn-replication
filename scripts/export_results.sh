#!/usr/bin/env bash
# Copy the small, shareable result files from the data folder into results/ for version control.
# Checkpoints, latents and datasets stay in the data folder (too large / not ours to redistribute).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${CKTGNN_DATA:-/d/CktGNN_data}/results"
OUT="$ROOT/results"
mkdir -p "$OUT"
cp -r "$DATA/audit" "$DATA/summary" "$OUT/" 2>/dev/null
mkdir -p "$OUT/simulator"
cp "$DATA/simulator/"{validation_summary.json,label_diagnostics.json} "$OUT/simulator/" 2>/dev/null
for run in "$DATA"/vae/*_seed*; do
  name=$(basename "$run"); mkdir -p "$OUT/vae/$name"
  for f in config.json train_log.csv results.json sgp_results.csv circuit_eval.json; do
    [ -f "$run/$f" ] && cp "$run/$f" "$OUT/vae/$name/"
  done
done
echo "exported to $OUT"; du -sh "$OUT"
