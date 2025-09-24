#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
export IQOPTION_ACCOUNT_TYPE=PRACTICE
python -m app.gui "$@"
