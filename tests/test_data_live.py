import threading
import time

import pytest

from app.data_live import RealTimeDataFeed
from app.iq_client import DryRunIQClient


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


@pytest.fixture()
def fake_clock() -> FakeClock:
    return FakeClock()


def test_feed_buffers_and_callbacks(fake_clock: FakeClock) -> None:
    client = DryRunIQClient(["EURUSD"], [60], time_provider=fake_clock.__call__)
    feed = RealTimeDataFeed(client, ["EURUSD"], [60], poll_interval=0.01, maxlen=20)

    events = []
    event = threading.Event()

    def on_candle(asset: str, timeframe: int, candle):
        events.append((asset, timeframe, candle["epoch"]))
        event.set()

    feed.register_callback(on_candle)
    feed.start()

    # Allow the worker to boot and read initial candles
    time.sleep(0.05)
    buffer = feed.get_buffer("EURUSD", 60)
    assert len(buffer) == 20

    # Advance time to trigger a new candle emission
    fake_clock.advance(60)
    if not event.wait(timeout=0.2):
        # Advance once more in case the worker polled just before advancing time
        fake_clock.advance(60)
        event.wait(timeout=0.2)

    feed.stop()

    assert events, "Expected at least one callback invocation"
    last_epoch = buffer[-1]["epoch"]
    assert events[-1][2] == last_epoch


def test_feed_update_assets(fake_clock: FakeClock) -> None:
    client = DryRunIQClient(["EURUSD", "GBPUSD"], [60, 300], time_provider=fake_clock.__call__)
    feed = RealTimeDataFeed(client, ["EURUSD"], [60], poll_interval=0.05, maxlen=5)

    feed.start()
    time.sleep(0.05)
    feed.update_assets(["EURUSD", "GBPUSD"], [60])
    time.sleep(0.05)

    buffers = feed.buffers()
    feed.stop()

    assert set(buffers.keys()) == {"EURUSD", "GBPUSD"}
    assert all(60 in tf_map for tf_map in buffers.values())
