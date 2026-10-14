#!/usr/bin/env bash
# For runs started before --eval-device existed: wait until epoch-300 weights are saved, stop the
# training process (its lane then continues), and run the evaluation on the CPU instead.
# Usage: bash scripts/finish_run_on_cpu.sh MODEL SEED
set -u
MODEL=$1; SEED=$2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${CKTGNN_DATA:-/d/CktGNN_data}/results/vae/${MODEL}_seed${SEED}"
until [ -f "$RUN/model_checkpoint300.pth" ]; do sleep 30; done
sleep 20                                            # let the checkpoint write finish
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -like '*run_vae.py*--model $MODEL --seed $SEED *' -and \$_.CommandLine -notlike '*--eval-only*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
echo "training process stopped after epoch 300; evaluating on CPU $(date -Iseconds)" >> "$RUN/stdout.log"
CUDA_VISIBLE_DEVICES="" "$ROOT/.venv/Scripts/python.exe" -u "$ROOT/scripts/run_vae.py" --model "$MODEL" --seed "$SEED" \
    --eval-only --device cpu --threads 2 >> "$RUN/stdout.log" 2>&1
echo "eval exit $? $(date -Iseconds)" >> "$RUN/stdout.log"
