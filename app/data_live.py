"""Real time market data ingestion utilities."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Iterable, List, MutableMapping, Optional, Tuple

logger = logging.getLogger(__name__)

Candle = Dict[str, float]
Callback = Callable[[str, int, Candle], None]


@dataclass
class _StreamState:
    buffer: Deque[Candle]
    last_epoch: Optional[float] = None


class RealTimeDataFeed:
    """Fan out real-time candles to multiple consumers.

    The feed manages candle streams for multiple ``asset``/``timeframe``
    combinations.  It stores a rolling buffer (``maxlen`` candles) for
    each stream and notifies registered callbacks whenever a new candle
    closes.  Callbacks execute on the feeder thread; handlers should be
    non-blocking.
    """

    def __init__(
        self,
        client,
        assets: Iterable[str],
        timeframes: Iterable[int],
        *,
        maxlen: int = 500,
        poll_interval: float = 1.0,
    ) -> None:
        self.client = client
        self.assets = list(dict.fromkeys(assets))
        self.timeframes = sorted(set(int(tf) for tf in timeframes))
        self.maxlen = max(10, int(maxlen))
        self.poll_interval = max(0.01, float(poll_interval))

        self._callbacks: List[Callback] = []
        self._streams: Dict[Tuple[str, int], _StreamState] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start streaming candles."""

        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._initialise_streams()
            self._thread = threading.Thread(target=self._run, name="iq-data-feed", daemon=True)
            self._thread.start()
            logger.info("RealTimeDataFeed started for %s assets", len(self.assets))

    def stop(self) -> None:
        """Stop streaming and join the worker thread."""

        self._stop_event.set()
        thread = None
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread:
            thread.join(timeout=5.0)
        logger.info("RealTimeDataFeed stopped")

    def register_callback(self, fn: Callback) -> None:
        """Register a callback fired for every new candle."""

        if not callable(fn):  # pragma: no cover - guard clause
            raise TypeError("Callback must be callable")
        self._callbacks.append(fn)

    # ------------------------------------------------------------------
    # Stream management
    # ------------------------------------------------------------------
    def _initialise_streams(self) -> None:
        for asset in self.assets:
            for timeframe in self.timeframes:
                key = (asset, timeframe)
                buffer = deque(maxlen=self.maxlen)
                self.client.start_candles_stream(asset, timeframe, self.maxlen)
                candles = self._normalise_candles(self.client.get_realtime_candles(asset, timeframe))
                for candle in candles:
                    buffer.append(candle)
                last_epoch = buffer[-1]["epoch"] if buffer else None
                self._streams[key] = _StreamState(buffer=buffer, last_epoch=last_epoch)

    def update_assets(self, assets: Iterable[str], timeframes: Optional[Iterable[int]] = None) -> None:
        """Update tracked assets/timeframes without restarting the feed."""

        new_assets = list(dict.fromkeys(assets))
        new_timeframes = sorted(set(int(tf) for tf in (timeframes or self.timeframes)))

        with self._lock:
            old_keys = set(self._streams.keys())
            new_keys = {(asset, tf) for asset in new_assets for tf in new_timeframes}

            # stop removed streams
            for asset, timeframe in old_keys - new_keys:
                logger.info("Stopping stream %s/%ss", asset, timeframe)
                self.client.stop_candles_stream(asset, timeframe)
                self._streams.pop((asset, timeframe), None)

            # start new streams
            for asset, timeframe in new_keys - old_keys:
                logger.info("Starting stream %s/%ss", asset, timeframe)
                buffer = deque(maxlen=self.maxlen)
                self.client.start_candles_stream(asset, timeframe, self.maxlen)
                candles = self._normalise_candles(self.client.get_realtime_candles(asset, timeframe))
                for candle in candles:
                    buffer.append(candle)
                last_epoch = buffer[-1]["epoch"] if buffer else None
                self._streams[(asset, timeframe)] = _StreamState(buffer=buffer, last_epoch=last_epoch)

            self.assets = new_assets
            self.timeframes = new_timeframes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_candles(candles: MutableMapping[Any, Candle] | Iterable[Candle] | None) -> List[Candle]:
        if not candles:
            return []
        if isinstance(candles, dict):
            iterable = candles.values()
        else:
            iterable = candles
        normalised = []
        for candle in iterable:
            if not candle:
                continue
            try:
                epoch = float(candle["epoch"])
            except (KeyError, TypeError, ValueError):
                continue
            normalised.append({**candle, "epoch": epoch})
        normalised.sort(key=lambda item: item["epoch"])
        return normalised

    def buffers(self) -> Dict[str, Dict[int, Deque[Candle]]]:
        with self._lock:
            result: Dict[str, Dict[int, Deque[Candle]]] = {}
            for asset, timeframe in self._streams:
                result.setdefault(asset, {})[timeframe] = self._streams[(asset, timeframe)].buffer
            return result

    def get_buffer(self, asset: str, timeframe: int) -> Deque[Candle]:
        try:
            return self._streams[(asset, timeframe)].buffer
        except KeyError as exc:  # pragma: no cover - runtime guard
            raise KeyError(f"Stream {asset}/{timeframe}s is not tracked") from exc

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            try:
                self._poll_once()
            except Exception as exc:  # pragma: no cover - logged for diagnosis
                logger.exception("Realtime data poll failed: %s", exc)
            elapsed = time.time() - start
            sleep_for = max(0.0, self.poll_interval - elapsed)
            if sleep_for:
                self._stop_event.wait(sleep_for)

    def _poll_once(self) -> None:
        with self._lock:
            for (asset, timeframe), state in list(self._streams.items()):
                candles = self._normalise_candles(self.client.get_realtime_candles(asset, timeframe))
                if not candles:
                    continue
                for candle in candles:
                    epoch = candle["epoch"]
                    if state.last_epoch is None or epoch > state.last_epoch:
                        state.buffer.append(candle)
                        state.last_epoch = epoch
                        for cb in list(self._callbacks):
                            try:
                                cb(asset, timeframe, candle)
                            except Exception:  # pragma: no cover - callback bug shouldn't kill feed
                                logger.exception("Callback %s failed", cb)

