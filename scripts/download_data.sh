#!/bin/bash
ASSET=${1:-EURUSD}
TF=${2:-60}
python -c "from app.data_history import HistoricalDataManager; from app.iq_client import make_client_from_env; HistoricalDataManager(make_client_from_env()).download_data('$ASSET', $TF)"