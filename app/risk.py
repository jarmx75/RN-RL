"""Lógica de gestión de riesgo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from typing import Dict, Optional

from .utils.logger import get_logger


@dataclass
class RiskConfig:
    max_consecutive_losses: int
    daily_stop_loss: float
    daily_take_profit: float
    trading_hours: Dict[str, str]
    pause_after_losses: int
    pause_duration_minutes: int
    dynamic_threshold: Dict[str, float]


class RiskManager:
    """Evalúa reglas de riesgo antes de cada operación."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._logger = get_logger("risk")
        self._reset_state()

    def _reset_state(self) -> None:
        self.consecutive_losses = 0
        self.daily_profit = 0.0
        self.last_reset = datetime.utcnow().date()
        self.pause_until: Optional[float] = None
        self.dynamic_threshold = self.config.dynamic_threshold.get("min", 0.7)

    def _maybe_reset(self) -> None:
        today = datetime.utcnow().date()
        if today != self.last_reset:
            self._logger.info("Reinicio diario de métricas de riesgo")
            self._reset_state()

    def _within_trading_hours(self) -> bool:
        start = self._parse_time(self.config.trading_hours.get("start", "00:00"))
        end = self._parse_time(self.config.trading_hours.get("end", "23:59"))
        now = datetime.utcnow().time()
        if start <= end:
            return start <= now <= end
        # Overnight window
        return now >= start or now <= end

    @staticmethod
    def _parse_time(text: str) -> dt_time:
        hour, minute = map(int, text.split(":"))
        return dt_time(hour, minute)

    def register_trade(self, result: float) -> None:
        self._maybe_reset()
        self.daily_profit += result
        if result < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if (
            self.config.pause_after_losses
            and self.consecutive_losses >= self.config.pause_after_losses
        ):
            self.pause_until = time.time() + self.config.pause_duration_minutes * 60
            self._logger.warning(
                "Pausa activada por racha de pérdidas (%s)",
                self.consecutive_losses,
            )

        if self.config.dynamic_threshold.get("enabled", False):
            step = self.config.dynamic_threshold.get("step", 0.02)
            if result > 0:
                self.dynamic_threshold = min(
                    self.config.dynamic_threshold.get("max", 0.9),
                    self.dynamic_threshold + step,
                )
            else:
                self.dynamic_threshold = max(
                    self.config.dynamic_threshold.get("min", 0.6),
                    self.dynamic_threshold - step,
                )
            self._logger.info("Umbral dinámico actualizado a %.2f", self.dynamic_threshold)

    def allowed(self, probability: float, amount: float, min_probability: float) -> bool:
        """Determina si se puede operar según el estado actual."""

        self._maybe_reset()

        if self.pause_until and time.time() < self.pause_until:
            self._logger.warning("En pausa hasta %s", datetime.fromtimestamp(self.pause_until))
            return False

        if not self._within_trading_hours():
            self._logger.debug("Fuera del horario permitido")
            return False

        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self._logger.warning("Máximo de pérdidas consecutivas alcanzado")
            return False

        if self.daily_profit <= self.config.daily_stop_loss:
            self._logger.warning("Stop loss diario alcanzado")
            return False

        if self.daily_profit >= self.config.daily_take_profit:
            self._logger.info("Meta diaria alcanzada")
            return False

        threshold = (
            self.dynamic_threshold
            if self.config.dynamic_threshold.get("enabled", False)
            else min_probability
        )
        return probability >= threshold

    def remaining_drawdown(self) -> float:
        return self.config.daily_stop_loss - self.daily_profit

    def remaining_target(self) -> float:
        return self.config.daily_take_profit - self.daily_profit
