#!/usr/bin/env bash
# Launch one VAE training run with logs in the data folder; resumes if last.pt exists.
# Usage: [DEVICE=cuda|cpu] [THREADS=2] bash scripts/launch_job.sh MODEL SEED [extra run_vae.py args...]
# Example: DEVICE=cuda bash scripts/launch_job.sh CktGNN 1 --epochs 300 --patience 10
set -u
MODEL=$1; SEED=$2; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${CKTGNN_DATA:-/d/CktGNN_data}"
RUN="$DATA/results/vae/${MODEL}_seed${SEED}"
DEVICE="${DEVICE:-cpu}"
mkdir -p "$RUN"
RESUME=""
[ -f "$RUN/last.pt" ] && RESUME="--resume"
[ "$DEVICE" = "cpu" ] && export CUDA_VISIBLE_DEVICES=""   # keep the CUDA runtime out of CPU runs
"$ROOT/.venv/Scripts/python.exe" -u "$ROOT/scripts/run_vae.py" --model "$MODEL" --seed "$SEED" \
    --device "$DEVICE" --threads "${THREADS:-2}" $RESUME "$@" >> "$RUN/stdout.log" 2>&1
echo "exit $? $(date -Iseconds)" >> "$RUN/stdout.log"
