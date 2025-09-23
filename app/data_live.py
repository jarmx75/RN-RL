import threading
import time
from collections import deque
from typing import Dict, Callable, List, Any

class RealTimeDataFeed:
    def __init__(self, client, assets: List[str], timeframes: List[int], maxlen: int = 200):
        self.client = client
        self.assets = assets
        self.timeframes = timeframes
        self.maxlen = maxlen
        self.deques: Dict[str, Dict[int, deque]] = {
            asset: {tf: deque(maxlen=maxlen) for tf in timeframes} for asset in assets
        }
        self.callbacks: List[Callable[[str, int, dict], None]] = []
        self.thread = None
        self.running = False

    def register_callback(self, fn: Callable[[str, int, dict], None]):
        self.callbacks.append(fn)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run(self):
        while self.running:
            for asset in self.assets:
                for tf in self.timeframes:
                    candles = self.client.get_realtime_candles(asset, tf)
                    if candles:
                        for candle in candles[-1:]:
                            self.deques[asset][tf].append(candle)
                            for cb in self.callbacks:
                                cb(asset, tf, candle)
            time.sleep(1)