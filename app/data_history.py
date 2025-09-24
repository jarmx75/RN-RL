"""Descarga y carga de datos históricos de IQ Option."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

from .iq_client import IQOptionClient
from .utils.logger import get_logger


@dataclass
class HistoryRequest:
    asset: str
    timeframe: int
    days: int


class HistoricalDataManager:
    """Gestiona descargas históricas en bloques."""

    def __init__(self, client: IQOptionClient, data_dir: str) -> None:
        self.client = client
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("history")

    def _fetch_block(self, asset: str, timeframe: int, end: int, count: int) -> List[Dict[str, float]]:
        self.client.ensure_connection()
        candles = self.client.client.get_candles(asset, timeframe, count, end)
        return candles

    def download(self, requests: Iterable[HistoryRequest]) -> Dict[Tuple[str, int], Path]:
        results: Dict[Tuple[str, int], Path] = {}
        for request in requests:
            filename = self.data_dir / f"{request.asset}_{request.timeframe}.csv"
            end = int(time.time())
            candles: List[Dict[str, float]] = []
            total = request.days * 24 * 60 // (request.timeframe // 60)
            block = min(1000, total)
            with tqdm(total=total, desc=f"Descargando {request.asset} {request.timeframe}s") as pbar:
                while len(candles) < total:
                    chunk = self._fetch_block(request.asset, request.timeframe, end, block)
                    if not chunk:
                        break
                    candles.extend(chunk)
                    end = chunk[0]["from"] - request.timeframe
                    pbar.update(len(chunk))
                    time.sleep(0.2)
            if candles:
                df = pd.DataFrame(candles)
                df = df.sort_values("from")
                df.to_csv(filename, index=False)
                results[(request.asset, request.timeframe)] = filename
                self._logger.info(
                    "Descargadas %s velas de %s %ss", len(df), request.asset, request.timeframe
                )
        return results

    def load(self, asset: str, timeframe: int, limit: int | None = None) -> pd.DataFrame:
        filename = self.data_dir / f"{asset}_{timeframe}.csv"
        if not filename.exists():
            raise FileNotFoundError(filename)
        df = pd.read_csv(filename)
        if limit is not None:
            df = df.tail(limit)
        return df
