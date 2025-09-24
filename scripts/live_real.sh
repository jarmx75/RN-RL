#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export IQOPTION_ACCOUNT_TYPE=REAL
python -m app.gui "$@"
