"""Persistent storage helpers for trades and metrics."""

from __future__ import annotations

import csv
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def ensure_dirs(base_dir: str = ".") -> None:
    base = Path(base_dir)
    for sub in ("models", "logs", "reports", "data"):
        (base / sub).mkdir(parents=True, exist_ok=True)


@dataclass
class TradeRecord:
    timestamp: float
    asset: str
    timeframe: int
    direction: str
    amount: float
    probability: float
    result: float
    payout: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "amount": self.amount,
            "probability": self.probability,
            "result": self.result,
            "payout": self.payout,
        }


class TradeLogger:
    """Write trade executions to CSV and SQLite."""

    def __init__(
        self,
        *,
        csv_path: str = "logs/trades.csv",
        db_path: str = "data/trades.sqlite",
        rotate_bytes: int = 5 * 1024 * 1024,
    ) -> None:
        ensure_dirs()
        self.csv_path = Path(csv_path)
        self.db_path = Path(db_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.rotate_bytes = rotate_bytes
        self._conn = sqlite3.connect(self.db_path)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # SQLite schema
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    asset TEXT NOT NULL,
                    timeframe INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    amount REAL NOT NULL,
                    probability REAL NOT NULL,
                    result REAL NOT NULL,
                    payout REAL NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(self, record: TradeRecord | Dict[str, Any]) -> None:
        if isinstance(record, TradeRecord):
            payload = record.to_dict()
        else:
            payload = dict(record)
        payload.setdefault("timestamp", time.time())
        self._write_csv(payload)
        self._write_sqlite(payload)

    def _write_csv(self, payload: Dict[str, Any]) -> None:
        rotate = self.rotate_bytes and self.csv_path.exists() and self.csv_path.stat().st_size > self.rotate_bytes
        if rotate:
            rotated = self.csv_path.with_name(f"{self.csv_path.stem}_{int(time.time())}.csv")
            self.csv_path.rename(rotated)

        file_exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(payload)

    def _write_sqlite(self, payload: Dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO trades (timestamp, asset, timeframe, direction, amount, probability, result, payout)
                VALUES (:timestamp, :asset, :timeframe, :direction, :amount, :probability, :result, :payout)
                """,
                payload,
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def export_csv(self, destination: str) -> None:
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cursor = self._conn.cursor()
        rows = cursor.execute(
            "SELECT timestamp, asset, timeframe, direction, amount, probability, result, payout FROM trades ORDER BY timestamp"
        )
        with dest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "asset", "timeframe", "direction", "amount", "probability", "result", "payout"])
            writer.writerows(rows)

    def close(self) -> None:
        self._conn.close()


__all__ = ["TradeLogger", "TradeRecord", "ensure_dirs"]

