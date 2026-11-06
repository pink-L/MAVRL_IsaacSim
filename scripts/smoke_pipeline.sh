#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ROOT="${1:-artifacts/smoke_replay}"
mkdir -p "$RUN_ROOT/logs"
scripts/python.sh -m unittest discover -s tests -v > "$RUN_ROOT/logs/tests.log" 2>&1
scripts/python.sh scripts/run.py train --out "$RUN_ROOT/initial" --steps 8192 --n-steps 128 --batch-size 512 --num-envs 4 > "$RUN_ROOT/logs/initial.log" 2>&1
scripts/python.sh scripts/run.py collect --out "$RUN_ROOT/dataset" --policy "$RUN_ROOT/initial/latest.pt" --num-envs 16 --frames 4096 --sequence-length 64 > "$RUN_ROOT/logs/collect.log" 2>&1
scripts/python.sh scripts/run.py vae --out "$RUN_ROOT/vae" --dataset "$RUN_ROOT/dataset" --epochs 10 > "$RUN_ROOT/logs/vae.log" 2>&1
scripts/python.sh scripts/run.py lstm --out "$RUN_ROOT/lstm" --dataset "$RUN_ROOT/dataset" --vae "$RUN_ROOT/vae/best.pt" --policy "$RUN_ROOT/initial/latest.pt" --epochs 10 > "$RUN_ROOT/logs/lstm.log" 2>&1
scripts/python.sh scripts/run.py train --out "$RUN_ROOT/final" --phase final --representation "$RUN_ROOT/lstm/best.pt" --steps 16384 --n-steps 128 --batch-size 512 --warmup-steps 4096 --num-envs 16 > "$RUN_ROOT/logs/final.log" 2>&1
scripts/python.sh scripts/run.py evaluate --out "$RUN_ROOT/evaluation" --policy "$RUN_ROOT/final/latest.pt" --episodes 12 --num-envs 4 > "$RUN_ROOT/logs/evaluate.log" 2>&1
