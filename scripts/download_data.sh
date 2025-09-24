#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m app.exec.runner_live --help >/dev/null 2>&1 || true
python - <<'PY'
from app import load_config
from app.data_history import HistoricalDataManager, HistoryRequest
from app.iq_client import IQOptionClient
from dotenv import load_dotenv
import os

load_dotenv()
cfg = load_config()
client = IQOptionClient(
    email=os.getenv('IQOPTION_EMAIL', ''),
    password=os.getenv('IQOPTION_PASSWORD', ''),
    account_type=os.getenv('IQOPTION_ACCOUNT_TYPE', 'PRACTICE'),
)
manager = HistoricalDataManager(client, cfg['storage']['data_dir'])
requests = [
    HistoryRequest(asset=asset, timeframe=tf, days=180)
    for asset in cfg['assets']
    for tf in cfg['timeframes']
]
manager.download(requests)
PY
