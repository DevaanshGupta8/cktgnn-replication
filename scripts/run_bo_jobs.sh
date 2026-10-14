#!/usr/bin/env bash
# Run a list of latent_bo.py jobs ("MODEL METHOD SEED" per line) on the CPU, PAR at a time (default 1).
# Jobs whose trace CSV already exists are skipped, so an interrupted batch can simply be restarted.
# Usage: [PAR=4] [OUT=<data>/results/bo] bash scripts/run_bo_jobs.sh JOBFILE
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export ROOT OUT="${OUT:-${CKTGNN_DATA:-/d/CktGNN_data}/results/bo}"
mkdir -p "$OUT"
job() {
  [ -f "$OUT/$1_$2_seed$3.csv" ] && return
  CUDA_VISIBLE_DEVICES="" "$ROOT/.venv/Scripts/python.exe" "$ROOT/scripts/latent_bo.py" --model "$1" \
      --method "$2" --seed "$3" --threads 1 --out "$OUT" 2>>"$OUT/errors.txt" | tail -1 >> "$OUT/log.txt"
}
export -f job
grep -v '^\s*$' "$1" | xargs -P "${PAR:-1}" -L 1 bash -c 'job "$@"' _
