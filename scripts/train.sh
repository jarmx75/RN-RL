#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m app.exec.runner_train "$@"
