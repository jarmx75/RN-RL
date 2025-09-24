#!/bin/bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  python -m app.data_history
else
  python -m app.data_history "$@"
fi