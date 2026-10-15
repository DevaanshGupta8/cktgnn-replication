#!/usr/bin/env bash
# Copy every small, shareable result file from the data folder into results/ for version control.
# Not copied (too large or not ours to redistribute): datasets, checkpoints (last.pt, model_checkpoint*.pth) and
# latents.npz. The final checkpoints and latents of every trained run are in the GitHub release instead
# (see README, "Using this repository without training anything").
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${CKTGNN_DATA:-/d/CktGNN_data}/results"
OUT="$ROOT/results"
mkdir -p "$OUT"
cp -r "$DATA/audit" "$DATA/summary" "$OUT/" 2>/dev/null
mkdir -p "$OUT/simulator"
cp "$DATA"/simulator/* "$OUT/simulator/" 2>/dev/null     # validation, label diagnostics, per-circuit labels
for run in "$DATA"/vae/*_seed*; do
  name=$(basename "$run"); mkdir -p "$OUT/vae/$name"
  for f in config.json train_log.csv results.json sgp_results.csv circuit_eval.json guided_generation.json \
           guided_latents.csv stdout.log sgp_log.txt circuit_eval_log.txt; do
    [ -f "$run/$f" ] && cp "$run/$f" "$OUT/vae/$name/"
  done
done
# optimisation traces (one CSV per run) and their logs: Phase 4 (bo), Phases 5-6 (bo5), Phase 7 (bo7)
for d in bo bo5 bo7; do
  [ -d "$DATA/$d" ] && mkdir -p "$OUT/$d" && cp "$DATA"/$d/*_seed*.csv "$OUT/$d/" 2>/dev/null
  for f in log.txt errors.txt; do [ -f "$DATA/$d/$f" ] && cp "$DATA/$d/$f" "$OUT/$d/$f"; done
done
mkdir -p "$OUT/jobs"                                         # the job lists that defined every BO batch
cp "$DATA"/bo*_jobs*.txt "$OUT/jobs/" 2>/dev/null
echo "exported to $OUT"; du -sh "$OUT"
