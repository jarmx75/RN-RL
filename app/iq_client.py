"""IQ Option client abstraction with DryRun support.

This module exposes two concrete client implementations:

* :class:`IQClient` – a thin yet battle-tested wrapper around the
  ``iqoptionapi`` websocket client.  It adds reconnection, optional
  PRACTICE/REAL balance selection, candle stream management and helpers
  to synchronise order placement with the next candle close.
* :class:`DryRunIQClient` – an in-memory simulator used for automated
  testing and local development when real credentials are not supplied.

Both clients share a common public API so higher level components do not
have to distinguish between the real and simulated implementation.

The module also exposes :func:`make_client_from_env` that inspects
environment variables and configuration defaults to decide whether a
real or simulated client should be returned.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class IQClientError(RuntimeError):
    """Raised when the real IQ Option client cannot be initialised."""


def _exponential_backoff(
    attempt: int,
    base: float = 1.5,
    cap: float = 30.0,
    jitter: float = 0.2,
) -> float:
    """Return backoff seconds for ``attempt`` with decorrelated jitter."""

    sleep = min(cap, base * (2 ** attempt))
    if jitter:
        sleep = sleep * (1.0 - jitter) + random.random() * sleep * jitter
    return sleep


def _now_ts() -> float:
    """Return monotonic-ish timestamp used for candle alignment."""

    return time.time()


class IQClient:
    """High level wrapper around :mod:`iqoptionapi`.

    The wrapper focuses on resiliency.  It reconnects automatically,
    guards API access with a lock and exposes convenience helpers used by
    the live runner.
    """

    def __init__(
        self,
        reconnect_attempts: int = 5,
        account: str = "PRACTICE",
        candle_sync_slack: float = 0.25,
    ) -> None:
        try:
            from iqoptionapi.stable_api import IQ_Option  # type: ignore
        except Exception as exc:  # pragma: no cover - only triggered without dependency
            raise IQClientError(
                "The iqoptionapi package is required for live trading."
            ) from exc

        self._iq_option_cls = IQ_Option
        self._iq = None
        self._lock = threading.RLock()
        self._connected = False
        self._reconnect_attempts = reconnect_attempts
        self._account = account
        self._candle_sync_slack = candle_sync_slack

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect(self, email: str, password: str, account: Optional[str] = None) -> bool:
        """Connect to IQ Option.

        Args:
            email: IQ Option account email.
            password: IQ Option account password.
            account: Optional balance type to switch to (``PRACTICE`` or
                ``REAL``).  If omitted the value supplied when initialising
                the client is used.
        Returns:
            ``True`` when a websocket connection was established.
        """

        target_account = account or self._account
        with self._lock:
            self._iq = self._iq_option_cls(email, password)
            for attempt in range(self._reconnect_attempts):
                logger.info("Connecting to IQ Option (attempt %s)", attempt + 1)
                self._iq.connect()
                if self._iq.check_connect():
                    try:
                        self._iq.change_balance(target_account)
                    except Exception:
                        logger.warning("Unable to switch balance to %s", target_account)
                    self._connected = True
                    logger.info("Connected to IQ Option")
                    return True

                sleep_for = _exponential_backoff(attempt)
                logger.warning("IQ Option connection failed, retrying in %.1fs", sleep_for)
                time.sleep(sleep_for)

            self._connected = False
            logger.error("IQ Option connection could not be established")
            return False

    def _ensure_connection(self) -> None:
        if not self._iq:
            raise IQClientError("Client not initialised – call connect first")
        if self._connected and self._iq.check_connect():
            return

        logger.warning("IQ Option connection dropped – attempting reconnection")
        for attempt in range(self._reconnect_attempts):
            try:
                self._iq.connect()
            except Exception as exc:  # pragma: no cover - network error
                logger.exception("Reconnect attempt %s failed: %s", attempt + 1, exc)
            if self._iq.check_connect():
                self._connected = True
                logger.info("Reconnected to IQ Option")
                return
            time.sleep(_exponential_backoff(attempt))

        self._connected = False
        raise IQClientError("Unable to reconnect to IQ Option")

    def change_balance(self, balance_type: str) -> None:
        with self._lock:
            self._ensure_connection()
            try:
                self._iq.change_balance(balance_type)
            except Exception as exc:  # pragma: no cover - depends on remote API
                logger.exception("Failed to change balance: %s", exc)
                raise

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_all_open_time(self) -> Dict[str, Any]:
        with self._lock:
            self._ensure_connection()
            try:
                return self._iq.get_all_open_time()
            except Exception as exc:  # pragma: no cover - network errors
                logger.exception("Failed to fetch open time: %s", exc)
                return {}

    def start_candles_stream(self, asset: str, timeframe: int, count: int) -> None:
        with self._lock:
            self._ensure_connection()
            self._iq.start_candles_stream(asset, timeframe, count)

    def get_realtime_candles(self, asset: str, timeframe: int) -> List[Dict[str, Any]]:
        with self._lock:
            self._ensure_connection()
            candles = self._iq.get_realtime_candles(asset, timeframe)
            return list(candles.values()) if isinstance(candles, dict) else candles

    def stop_candles_stream(self, asset: str, timeframe: int) -> None:
        with self._lock:
            if not self._iq:
                return
            self._iq.stop_candles_stream(asset, timeframe)

    # ------------------------------------------------------------------
    # Trading helpers
    # ------------------------------------------------------------------
    def wait_for_next_candle(self, timeframe: int) -> None:
        """Sleep until just before the next candle for ``timeframe``.

        ``timeframe`` is expressed in seconds (e.g. 60, 300, 900).
        """

        with self._lock:
            self._ensure_connection()
            server_ts = None
            try:
                server_ts = self._iq.get_server_timestamp()
            except Exception:  # pragma: no cover - not available in dry tests
                logger.debug("Falling back to local clock for candle sync")

        now = float(server_ts or _now_ts())
        timeframe = int(timeframe)
        next_candle = ((now // timeframe) + 1) * timeframe
        sleep_for = max(0.0, next_candle - now - self._candle_sync_slack)
        if sleep_for:
            time.sleep(sleep_for)

    def buy(self, amount: float, asset: str, direction: str, duration: int) -> Tuple[bool, Optional[str]]:
        with self._lock:
            self._ensure_connection()
            status, order_id = self._iq.buy(amount, asset, direction, duration)
            return bool(status), order_id

    def buy_digital_spot(
        self, amount: float, asset: str, direction: str, duration: int
    ) -> Tuple[bool, Optional[str]]:
        with self._lock:
            self._ensure_connection()
            try:
                status, order_id = self._iq.buy_digital_spot(asset, amount, direction, duration)
                return bool(status), order_id
            except Exception as exc:  # pragma: no cover - not all accounts support digital
                logger.warning("Digital spot order failed (%s), falling back to turbo", exc)
                return self.buy(amount, asset, direction, duration)

    def check_win_v2(self, trade_id: str, poll_sec: int = 3) -> float:
        with self._lock:
            self._ensure_connection()
            time.sleep(max(0, poll_sec))
            return float(self._iq.check_win_v2(trade_id))

    def check_connect(self) -> bool:
        with self._lock:
            return bool(self._iq and self._iq.check_connect())


# ----------------------------------------------------------------------
# Dry run client
# ----------------------------------------------------------------------


def _generate_candle(
    last_price: float,
    timeframe: int,
    timestamp: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[float, Dict[str, Any]]:
    rng = rng or random
    timestamp = timestamp or _now_ts()
    drift = rng.normalvariate(0.0, 0.0005)
    open_p = last_price
    close_p = max(0.0001, open_p + drift)
    high = max(open_p, close_p) + abs(rng.normalvariate(0.0, 0.0003))
    low = max(0.0001, min(open_p, close_p) - abs(rng.normalvariate(0.0, 0.0003)))
    volume = max(1, int(abs(rng.normalvariate(5, 2))))
    candle = {
        "open": open_p,
        "close": close_p,
        "high": high,
        "low": low,
        "volume": volume,
        "epoch": timestamp,
        "timeframe": timeframe,
    }
    return close_p, candle


@dataclass
class _DryStream:
    candles: Deque[Dict[str, Any]]
    price: float
    timeframe: int


class DryRunIQClient:
    """In-memory deterministic-ish simulation of IQ Option API."""

    def __init__(
        self,
        assets: Iterable[str],
        timeframes: Iterable[int],
        initial_balance: float = 1_000.0,
        default_payout: float = 0.8,
        sleep_fn: Callable[[float], None] | None = None,
        time_provider: Callable[[], float] | None = None,
    ) -> None:
        self.assets = list(assets)
        self.timeframes = list(timeframes)
        self.balance = float(initial_balance)
        self.default_payout = float(default_payout)
        self.trades: Dict[str, Dict[str, Any]] = {}
        self._streams: Dict[Tuple[str, int], _DryStream] = {}
        self._rng = random.Random(7)
        self._sleep = sleep_fn or (lambda x: None)
        self._time = time_provider or _now_ts

    # connection compatible API -------------------------------------------------
    def connect(self, email: str = "", password: str = "", account: str = "PRACTICE") -> bool:  # noqa: ARG002
        return True

    def change_balance(self, balance_type: str) -> None:  # noqa: ARG002
        return None

    def get_all_open_time(self) -> Dict[str, Any]:
        now = self._time()
        return {
            asset: {
                "open": True,
                "next_update": now + 3600,
                "timeframes": {tf: {"open": True} for tf in self.timeframes},
            }
            for asset in self.assets
        }

    # market data ---------------------------------------------------------------
    def start_candles_stream(self, asset: str, timeframe: int, count: int) -> None:
        key = (asset, timeframe)
        if key not in self._streams:
            from collections import deque

            price = 1.0 + 0.01 * self._rng.random()
            candles = deque(maxlen=count)
            now = self._time()
            start_ts = now - timeframe * (count - 1)
            ts = start_ts
            for _ in range(count):
                price, candle = _generate_candle(price, timeframe, timestamp=ts, rng=self._rng)
                candles.append(candle)
                ts += timeframe
            self._streams[key] = _DryStream(candles=candles, price=price, timeframe=timeframe)

    def get_realtime_candles(self, asset: str, timeframe: int) -> List[Dict[str, Any]]:
        key = (asset, timeframe)
        stream = self._streams.get(key)
        if not stream:
            self.start_candles_stream(asset, timeframe, 10)
            stream = self._streams[key]

        last_ts = stream.candles[-1]["epoch"] if stream.candles else self._time()
        now = self._time()
        if now - last_ts >= timeframe:
            price, candle = _generate_candle(stream.price, timeframe, rng=self._rng)
            candle["epoch"] = last_ts + timeframe
            stream.price = price
            stream.candles.append(candle)
        return list(stream.candles)

    def stop_candles_stream(self, asset: str, timeframe: int) -> None:
        self._streams.pop((asset, timeframe), None)

    # trading -------------------------------------------------------------------
    def wait_for_next_candle(self, timeframe: int) -> None:
        self._sleep(0.0)

    def _payout(self, amount: float) -> float:
        return round(amount * self.default_payout, 2)

    def buy(self, amount: float, asset: str, direction: str, duration: int) -> Tuple[bool, str]:  # noqa: ARG002
        trade_id = f"DRY_{self._rng.randint(1000, 9999)}"
        self.trades[trade_id] = {
            "asset": asset,
            "direction": direction,
            "amount": float(amount),
            "duration": duration,
            "timestamp": self._time(),
        }
        return True, trade_id

    def buy_digital_spot(self, amount: float, asset: str, direction: str, duration: int) -> Tuple[bool, str]:  # noqa: ARG002
        return self.buy(amount, asset, direction, duration)

    def check_win_v2(self, trade_id: str, poll_sec: int = 1) -> float:
        self._sleep(max(0, poll_sec))
        trade = self.trades.get(trade_id)
        if not trade:
            return 0.0

        win = self._rng.random() > 0.45
        return self._payout(trade["amount"]) if win else -trade["amount"]

    def check_connect(self) -> bool:
        return True

    def get_candles(self, asset: str, timeframe: int, count: int, end: Optional[int] = None):
        rng = random.Random(hash((asset, timeframe, int(end or self._time()))))
        price = 1.0 + 0.01 * rng.random()
        end_ts = int(end or self._time())
        start_ts = end_ts - timeframe * (count - 1)
        candles = []
        ts = start_ts
        for _ in range(count):
            price, candle = _generate_candle(price, timeframe, timestamp=ts, rng=rng)
            candles.append({**candle, "from": ts, "to": ts + timeframe})
            ts += timeframe
        return candles


def make_client_from_env(config_path: str = "config/config.yaml"):
    """Return a real client when credentials exist, otherwise DryRun.

    Environment variables recognised:

    ``IQ_EMAIL`` – account email
    ``IQ_PASSWORD`` – account password
    ``IQ_ACCOUNT`` – PRACTICE or REAL
    ``IQ_PAYOUT`` – optional fallback payout for DryRun
    """

    load_dotenv()
    email = (os.getenv("IQ_EMAIL") or "").strip()
    password = (os.getenv("IQ_PASSWORD") or "").strip()
    account = (os.getenv("IQ_ACCOUNT") or "PRACTICE").strip() or "PRACTICE"
    default_payout = float(os.getenv("IQ_PAYOUT", "0.8"))

    try:
        import yaml  # type: ignore

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (FileNotFoundError, ModuleNotFoundError):
        cfg = {}

    assets = cfg.get("assets", ["EURUSD", "GBPUSD"])
    timeframes = cfg.get("timeframes", [60, 300, 900])

    if not email or not password:
        logger.info("Credentials not provided – using DryRunIQClient")
        return DryRunIQClient(assets, timeframes, default_payout=default_payout)

    client = IQClient(account=account)
    if client.connect(email, password, account=account):
        logger.info("Live IQClient initialised in %s mode", account)
        return client

    logger.warning("Falling back to DryRunIQClient due to failed login")
    return DryRunIQClient(assets, timeframes, default_payout=default_payout)


__all__ = ["IQClient", "DryRunIQClient", "IQClientError", "make_client_from_env"]

