import os
import csv
from typing import Dict, Any, Optional

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def append_trade_log(row: Dict[str, Any], filename: str = "logs/trade_log.csv"):
    ensure_dirs()
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)