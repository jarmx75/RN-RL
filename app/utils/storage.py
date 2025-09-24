"""Persistencia en CSV y SQLite para operaciones y equity."""

from __future__ import annotations

import csv
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .logger import get_logger


@dataclass(slots=True)
class TradeRecord:
    """Representa una operación ejecutada."""

    timestamp: float
    asset: str
    timeframe: int
    direction: str
    probability: float
    amount: float
    payout: float
    result: float


class TradeStorage:
    """Gestor de persistencia para operaciones."""

    def __init__(self, csv_path: str, sqlite_path: str) -> None:
        self.csv_path = Path(csv_path)
        self.sqlite_path = Path(sqlite_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._logger = get_logger("storage")
        self._ensure_sqlite()

    def _ensure_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    asset TEXT,
                    timeframe INTEGER,
                    direction TEXT,
                    probability REAL,
                    amount REAL,
                    payout REAL,
                    result REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    balance REAL
                )
                """
            )
            conn.commit()

    def append_trade(self, record: TradeRecord) -> None:
        """Añade una operación a CSV y SQLite."""

        with self._lock:
            write_header = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if write_header:
                    writer.writerow(
                        [
                            "timestamp",
                            "asset",
                            "timeframe",
                            "direction",
                            "probability",
                            "amount",
                            "payout",
                            "result",
                        ]
                    )
                writer.writerow(
                    [
                        record.timestamp,
                        record.asset,
                        record.timeframe,
                        record.direction,
                        record.probability,
                        record.amount,
                        record.payout,
                        record.result,
                    ]
                )

            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    """
                    INSERT INTO trades (timestamp, asset, timeframe, direction, probability, amount, payout, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.timestamp,
                        record.asset,
                        record.timeframe,
                        record.direction,
                        record.probability,
                        record.amount,
                        record.payout,
                        record.result,
                    ),
                )
                conn.commit()
            self._logger.info(
                "Operación guardada: %s %s tf=%s result=%.2f",
                record.asset,
                record.direction,
                record.timeframe,
                record.result,
            )

    def load_trades(self) -> pd.DataFrame:
        with self.csv_path.open("r", encoding="utf-8") as fh:
            return pd.read_csv(fh)

    def append_equity(self, timestamp: float, balance: float) -> None:
        with self._lock:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    "INSERT INTO equity (timestamp, balance) VALUES (?, ?)",
                    (timestamp, balance),
                )
                conn.commit()

    def get_equity(self) -> pd.DataFrame:
        with sqlite3.connect(self.sqlite_path) as conn:
            df = pd.read_sql_query("SELECT timestamp, balance FROM equity", conn)
        return df


class EquityTracker:
    """Calcula métricas de equity a partir de operaciones."""

    def __init__(self, storage: TradeStorage, equity_csv: str) -> None:
        self.storage = storage
        self.equity_csv = Path(equity_csv)
        self.equity_csv.parent.mkdir(parents=True, exist_ok=True)

    def update(self) -> pd.DataFrame:
        trades = self.storage.load_trades()
        trades = trades.sort_values("timestamp")
        balance = 0.0
        equity_series: List[Dict[str, float]] = []
        for _, trade in trades.iterrows():
            balance += trade["result"]
            equity_series.append(
                {
                    "timestamp": float(trade["timestamp"]),
                    "balance": balance,
                }
            )
        df = pd.DataFrame(equity_series)
        df.to_csv(self.equity_csv, index=False)
        return df
