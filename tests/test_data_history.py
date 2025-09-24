import datetime as dt
from pathlib import Path

import sqlite3

from app.data_history import HistoricalDataManager
from app.iq_client import DryRunIQClient


def test_download_range_persists(tmp_path: Path) -> None:
    client = DryRunIQClient(["EURUSD"], [60])
    manager = HistoricalDataManager(
        client,
        db_path=str(tmp_path / "history.sqlite"),
        csv_dir=str(tmp_path / "csv"),
    )

    end = int(dt.datetime.utcnow().timestamp())
    start = end - 60 * 10

    df = manager.download_range("EURUSD", 60, start, end, chunk_size=5)
    manager.close()

    assert not df.empty
    csv_file = tmp_path / "csv" / "EURUSD_60.csv"
    assert csv_file.exists()

    conn = sqlite3.connect(tmp_path / "history.sqlite")
    count = conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    conn.close()
    assert count > 0
