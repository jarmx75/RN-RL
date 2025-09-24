"""Historical data downloader using IQ Option API."""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from app.iq_client import make_client_from_env


def _ensure_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                asset TEXT NOT NULL,
                timeframe INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (asset, timeframe, timestamp)
            )
            """
        )


class HistoricalDataManager:
    def __init__(self, client, db_path: str = "data/history.sqlite", csv_dir: str = "data/history") -> None:
        self.client = client
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        _ensure_schema(self.conn)

    def close(self) -> None:
        self.conn.close()

    def download_range(self, asset: str, timeframe: int, start: int, end: int, chunk_size: int = 1000) -> pd.DataFrame:
        records: List[Dict] = []
        to_time = int(end)
        timeframe = int(timeframe)
        while to_time > start:
            candles = self._fetch_candles(asset, timeframe, chunk_size, to_time)
            if not candles:
                break
            valid_timestamps: List[int] = []
            for candle in candles:
                ts = int(candle.get("from") or candle.get("timestamp") or candle.get("epoch"))
                if ts < start or ts > end:
                    continue
                valid_timestamps.append(ts)
                records.append(
                    {
                        "asset": asset,
                        "timeframe": timeframe,
                        "timestamp": ts,
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": float(candle.get("volume", 0.0)),
                    }
                )
            if not valid_timestamps:
                break
            earliest = min(valid_timestamps)
            to_time = earliest - timeframe * chunk_size

        if records:
            self._store(records)
        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values("timestamp", inplace=True)
            self._export_csv(asset, timeframe, df)
        return df

    def _fetch_candles(self, asset: str, timeframe: int, count: int, end: int):
        getter = getattr(self.client, "get_candles", None)
        if callable(getter):
            return getter(asset, timeframe, count, end)
        return []

    def _store(self, records: List[Dict]) -> None:
        with self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO candles (asset, timeframe, timestamp, open, high, low, close, volume)
                VALUES (:asset, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
                """,
                records,
            )

    def _export_csv(self, asset: str, timeframe: int, df: pd.DataFrame) -> None:
        path = self.csv_dir / f"{asset}_{timeframe}.csv"
        df.to_csv(path, index=False)

    def load_dataframe(self, asset: str, timeframe: int) -> pd.DataFrame:
        query = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE asset=? AND timeframe=? ORDER BY timestamp"
        return pd.read_sql_query(query, self.conn, params=(asset, timeframe))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download IQ Option historical data")
    parser.add_argument("--assets", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    parser.add_argument("--timeframes", nargs="+", type=int, default=[60, 300, 900])
    parser.add_argument("--days", type=int, default=7, help="Number of days to backfill")
    parser.add_argument("--db", type=str, default="data/history.sqlite")
    parser.add_argument("--csv-dir", type=str, default="data/history")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end = int(dt.datetime.utcnow().timestamp())
    start = end - args.days * 86400
    client = make_client_from_env()
    manager = HistoricalDataManager(client, db_path=args.db, csv_dir=args.csv_dir)
    try:
        for asset in args.assets:
            for timeframe in args.timeframes:
                print(f"Downloading {asset} {timeframe}s")
                manager.download_range(asset, timeframe, start, end)
    finally:
        manager.close()


if __name__ == "__main__":
    main()

