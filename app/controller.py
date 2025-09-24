"""Controlador principal que orquesta la GUI con el backend."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import yaml
from PyQt6.QtCore import QObject, pyqtSignal

from .rl.backtest import run_backtest
from .exec.runner_live import main as live_main
from .exec.runner_train import main as train_main
from .utils.logger import get_logger


@dataclass
class GuiState:
    assets: List[str]
    timeframes: List[int]
    threshold: float
    amount: float
    amount_percent: bool


class ControllerSignals(QObject):
    log = pyqtSignal(str)
    metrics = pyqtSignal(dict)
    balance = pyqtSignal(float)
    connection = pyqtSignal(bool)


class Controller(QObject):
    """Controla la ejecución de scripts desde la GUI."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        super().__init__()
        self.config_path = Path(config_path)
        self._logger = get_logger("controller")
        with self.config_path.open("r", encoding="utf-8") as fh:
            self._config = yaml.safe_load(fh)
        self.state = GuiState(
            assets=self._config["assets"],
            timeframes=self._config["timeframes"],
            threshold=self._config["threshold"],
            amount=self._config["amount"],
            amount_percent=self._config.get("amount_percent", False),
        )
        self.signals = ControllerSignals()
        self._threads: List[threading.Thread] = []

    def _run_async(self, target: Callable[[], None]) -> None:
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self._threads.append(thread)

    def start_training(self) -> None:
        self.signals.log.emit("Iniciando entrenamiento...")
        self._run_async(train_main)

    def run_backtest(self, model_path: str) -> None:
        def _target():
            self.signals.log.emit("Ejecutando backtest...")
            run_backtest(model_path)
            
        self._run_async(_target)

    def run_live(self, mode: str) -> None:
        def _target():
            os.environ["IQOPTION_ACCOUNT_TYPE"] = mode
            self.signals.log.emit(f"Iniciando live en modo {mode}")
            live_main()

        self._run_async(_target)

    def save_preferences(
        self,
        assets: List[str],
        timeframes: List[int],
        threshold: float,
        amount: float,
        amount_percent: bool,
    ) -> None:
        self._config["assets"] = assets
        self._config["timeframes"] = timeframes
        self._config["threshold"] = threshold
        self._config["amount"] = amount
        self._config["amount_percent"] = amount_percent
        with self.config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self._config, fh)
        self.signals.log.emit("Preferencias guardadas")
