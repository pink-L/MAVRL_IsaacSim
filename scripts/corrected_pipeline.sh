#!/usr/bin/env bash
# Source-matched short training; requires completed 512k initial policy.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ROOT="${1:-artifacts/corrected}"
mkdir -p "$RUN_ROOT/logs"
scripts/python.sh - "$RUN_ROOT" <<'PY'
import sys,torch
from pathlib import Path
checkpoint=torch.load(Path(sys.argv[1])/'initial/latest.pt',map_location='cpu',weights_only=False)
assert checkpoint['phase']=='initial' and checkpoint['timesteps']>=512000
PY
if [[ "${2:-}" != "--after-camera" ]]; then
scripts/python.sh scripts/run.py evaluate --policy "$RUN_ROOT/initial/latest.pt" --out "$RUN_ROOT/initial_evaluation" --densities empty --episodes 50 --num-envs 16 > "$RUN_ROOT/logs/initial_evaluation.log" 2>&1
scripts/python.sh scripts/check_camera_parity.py --out artifacts/camera_checks/individual > "$RUN_ROOT/logs/camera_individual.log" 2>&1
scripts/python.sh scripts/check_camera_parity.py --out artifacts/camera_checks/tiled --tiled > "$RUN_ROOT/logs/camera_tiled.log" 2>&1
scripts/python.sh scripts/check_camera_parity.py --compare artifacts/camera_checks/individual artifacts/camera_checks/tiled --out artifacts/camera_checks --allow-boundary-differences > "$RUN_ROOT/logs/camera_compare.log" 2>&1
fi
scripts/python.sh scripts/run.py collect --out "$RUN_ROOT/dataset" --policy "$RUN_ROOT/initial/latest.pt" --num-envs 16 --frames 16384 --seed 100 > "$RUN_ROOT/logs/collect.log" 2>&1
scripts/python.sh scripts/run.py vae --out "$RUN_ROOT/vae" --dataset "$RUN_ROOT/dataset" --epochs 20 > "$RUN_ROOT/logs/vae.log" 2>&1
scripts/python.sh scripts/run.py lstm --out "$RUN_ROOT/lstm" --dataset "$RUN_ROOT/dataset" --vae "$RUN_ROOT/vae/best.pt" --policy "$RUN_ROOT/initial/latest.pt" --epochs 30 > "$RUN_ROOT/logs/lstm.log" 2>&1
# The full source 1.2e6-step sparse warmup and 8e6-step LR schedule are retained.
# This 512k stopping point remains inside warmup, not a completed final policy.
scripts/python.sh scripts/run.py train --out "$RUN_ROOT/final" --phase final --representation "$RUN_ROOT/lstm/best.pt" --steps 512000 --num-envs 16 > "$RUN_ROOT/logs/final.log" 2>&1
scripts/python.sh scripts/run.py evaluate --out "$RUN_ROOT/evaluation" --policy "$RUN_ROOT/final/latest.pt" --episodes 30 --num-envs 16 > "$RUN_ROOT/logs/evaluate.log" 2>&1
scripts/python.sh scripts/run.py evaluate --out "$RUN_ROOT/evaluation_reset" --policy "$RUN_ROOT/final/latest.pt" --episodes 30 --num-envs 16 --reset-memory > "$RUN_ROOT/logs/evaluate_reset.log" 2>&1
scripts/python.sh scripts/validate_artifacts.py "$RUN_ROOT" > "$RUN_ROOT/logs/validate.log" 2>&1
scripts/python.sh scripts/plot_results.py "$RUN_ROOT" > "$RUN_ROOT/logs/plot.log" 2>&1
