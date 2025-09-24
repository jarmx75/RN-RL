"""Risk management utilities for live trading."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


HourWindow = Tuple[int, int]


def _normalise_hours(windows: Iterable[HourWindow]) -> Tuple[HourWindow, ...]:
    normalised = []
    for window in windows:
        start, end = window
        normalised.append((int(start) % 24, int(end) % 24 if end != 24 else 24))
    return tuple(normalised)


@dataclass
class RiskConfig:
    max_consecutive_losses: int = 3
    daily_stop_loss: float = 50.0
    daily_take_profit: float = 100.0
    daily_trade_limit: int | None = None
    max_concurrent_positions: int = 3
    max_concurrent_per_pair: int = 1
    operating_hours: Tuple[HourWindow, ...] = ((0, 24),)
    base_threshold: float = 0.7
    dynamic_threshold: bool = False
    threshold_step: float = 0.05
    min_threshold: float = 0.5
    max_threshold: float = 0.9


class RiskEngine:
    """Encapsulates guardrails for executing new trades."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.config.operating_hours = _normalise_hours(self.config.operating_hours)
        self._current_threshold = self.config.base_threshold
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._open_positions = defaultdict(int)
        self._current_date = dt.date.today()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def threshold(self) -> float:
        return self._current_threshold

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def daily_trades(self) -> int:
        return self._daily_trades

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _reset_daily_counters(self, today: dt.date) -> None:
        self._current_date = today
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._open_positions.clear()
        self._current_threshold = self.config.base_threshold

    def _ensure_date(self, now: dt.datetime) -> None:
        today = now.date()
        if today != self._current_date:
            self._reset_daily_counters(today)

    # ------------------------------------------------------------------
    # Guard checks
    # ------------------------------------------------------------------
    def can_trade(
        self,
        *,
        asset: str,
        timeframe: int,
        probability: float,
        now: dt.datetime,
    ) -> bool:
        """Return ``True`` if a new trade can be placed."""

        self._ensure_date(now)
        hour = now.hour + now.minute / 60.0
        if not any(start <= hour < end for start, end in self.config.operating_hours):
            return False

        if self._consecutive_losses >= self.config.max_consecutive_losses:
            return False

        if self.config.daily_stop_loss and self._daily_pnl <= -abs(self.config.daily_stop_loss):
            return False

        if self.config.daily_take_profit and self._daily_pnl >= abs(self.config.daily_take_profit):
            return False

        if self.config.daily_trade_limit and self._daily_trades >= self.config.daily_trade_limit:
            return False

        total_open = sum(self._open_positions.values())
        if total_open >= self.config.max_concurrent_positions:
            return False

        pair_key = (asset, timeframe)
        if self._open_positions[pair_key] >= self.config.max_concurrent_per_pair:
            return False

        return probability >= self._current_threshold

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def register_open(self, asset: str, timeframe: int) -> None:
        self._open_positions[(asset, timeframe)] += 1

    def register_close(self, *, asset: str, timeframe: int, result: float, now: dt.datetime) -> None:
        self._ensure_date(now)
        pair_key = (asset, timeframe)
        if self._open_positions[pair_key] > 0:
            self._open_positions[pair_key] -= 1

        self._daily_trades += 1
        self._daily_pnl += result

        if result > 0:
            self._consecutive_losses = 0
            if self.config.dynamic_threshold:
                self._current_threshold = max(
                    self.config.min_threshold,
                    self._current_threshold - self.config.threshold_step,
                )
        elif result < 0:
            self._consecutive_losses += 1
            if self.config.dynamic_threshold:
                self._current_threshold = min(
                    self.config.max_threshold,
                    self._current_threshold + self.config.threshold_step,
                )


__all__ = ["RiskEngine", "RiskConfig"]

