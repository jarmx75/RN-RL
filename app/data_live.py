"""Gestión de streams de velas en vivo."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, List, Tuple

import numpy as np

from .iq_client import IQOptionClient
from .utils.logger import get_logger
from .utils.time_sync import TimeSynchronizer

Candle = Dict[str, float]


@dataclass
class StreamConfig:
    assets: List[str]
    timeframes: List[int]
    buffer: int


class LiveDataManager:
    """Coordina la suscripción a streams y construye observaciones."""

    def __init__(
        self,
        client: IQOptionClient,
        config: StreamConfig,
        callback: Callable[[str, int, List[Candle]], None],
    ) -> None:
        self.client = client
        self.config = config
        self.callback = callback
        self._logger = get_logger("live_data")
        self._sync = TimeSynchronizer()
        self._candles: Dict[Tuple[str, int], Deque[Candle]] = {
            (asset, tf): deque(maxlen=config.buffer)
            for asset in config.assets
            for tf in config.timeframes
        }
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._sync.sync()
        for asset in self.config.assets:
            for tf in self.config.timeframes:
                thread = threading.Thread(
                    target=self._stream_loop,
                    args=(asset, tf),
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1)
        self._threads.clear()

    def _stream_loop(self, asset: str, timeframe: int) -> None:
        buffer = self._candles[(asset, timeframe)]
        self.client.start_candles_stream(asset, timeframe, self.config.buffer)
        try:
            while not self._stop.is_set():
                candles = self.client.get_realtime_candles(asset, timeframe)
                if not candles:
                    time.sleep(0.5)
                    continue
                sorted_candles = sorted(candles.values(), key=lambda x: x["from"])
                for candle in sorted_candles:
                    if buffer and buffer[-1]["from"] == candle["from"]:
                        buffer[-1] = candle
                    elif not buffer or candle["from"] > buffer[-1]["from"]:
                        buffer.append(candle)
                        self._logger.debug(
                            "Nueva vela %s %ss close=%s", asset, timeframe, candle["close"]
                        )
                # Detecta cierre de vela y dispara callback
                now = self._sync.now()
                candle_close = self._sync.next_candle_close(timeframe)
                if candle_close - now <= 0.5 and len(buffer) >= 2:
                    self.callback(asset, timeframe, list(buffer))
                time.sleep(0.25)
        finally:
            self.client.stop_candles_stream(asset, timeframe)

    def latest(self, asset: str, timeframe: int) -> List[Candle]:
        return list(self._candles[(asset, timeframe)])

    def build_observation(self, window_size: int | None = None) -> Dict[Tuple[str, int], np.ndarray]:
        obs: Dict[Tuple[str, int], np.ndarray] = {}
        for (asset, tf), candles in self._candles.items():
            if len(candles) < 2:
                continue
            arr = np.array(
                [[
                    c["open"],
                    c["close"],
                    c["high"],
                    c["low"],
                    c.get("volume", 0.0),
                ]
                 for c in candles],
                dtype=np.float32,
            )
            if window_size is not None and len(arr) > window_size:
                arr = arr[-window_size:]
            obs[(asset, tf)] = arr
        return obs
