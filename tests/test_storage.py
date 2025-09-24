from pathlib import Path

import sqlite3

from app.utils.storage import TradeLogger, TradeRecord


def test_trade_logger_writes_csv_and_sqlite(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs" / "trades.csv"
    db_path = tmp_path / "data" / "trades.sqlite"
    logger = TradeLogger(csv_path=str(csv_path), db_path=str(db_path), rotate_bytes=1024)

    record = TradeRecord(
        timestamp=1_000_000.0,
        asset="EURUSD",
        timeframe=60,
        direction="CALL",
        amount=10.0,
        probability=0.8,
        result=0.8,
        payout=0.8,
    )

    logger.log(record)
    logger.close()

    assert csv_path.exists()
    contents = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 2
    assert "EURUSD" in contents[1]

    conn = sqlite3.connect(db_path)
    rows = list(conn.execute("SELECT asset, amount, result FROM trades"))
    conn.close()
    assert rows == [("EURUSD", 10.0, 0.8)]


def test_trade_logger_rotation(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs" / "trades.csv"
    db_path = tmp_path / "data" / "trades.sqlite"
    logger = TradeLogger(csv_path=str(csv_path), db_path=str(db_path), rotate_bytes=200)

    for i in range(5):
        logger.log(
            {
                "timestamp": 1_000_000.0 + i,
                "asset": "EURUSD",
                "timeframe": 60,
                "direction": "CALL",
                "amount": 10.0,
                "probability": 0.8,
                "result": 0.8,
                "payout": 0.8,
            }
        )

    logger.close()
    rotated_files = list((tmp_path / "logs").glob("trades_*.csv"))
    assert rotated_files, "Expected rotated CSV file"
