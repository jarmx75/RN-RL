#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
MODEL_PATH=${1:-artifacts/models/ppo_latest.zip}
python -m app.exec.runner_backtest --model-path "$MODEL_PATH" "${@:2}"
