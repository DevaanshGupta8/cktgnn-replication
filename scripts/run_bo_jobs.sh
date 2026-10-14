#!/usr/bin/env bash
# Run a list of latent_bo.py jobs ("MODEL METHOD SEED" per line) sequentially on the CPU.
# Usage: bash scripts/run_bo_jobs.sh JOBFILE
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${CKTGNN_DATA:-/d/CktGNN_data}/results/bo/log.txt"
mkdir -p "$(dirname "$LOG")"
while read -r MODEL METHOD SEED; do
  [ -z "$MODEL" ] && continue
  CUDA_VISIBLE_DEVICES="" "$ROOT/.venv/Scripts/python.exe" "$ROOT/scripts/latent_bo.py" --model "$MODEL" \
      --method "$METHOD" --seed "$SEED" --threads 1 2>/dev/null | tail -1 >> "$LOG"
done < "$1"
