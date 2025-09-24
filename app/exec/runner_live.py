"""Ejecución en vivo del bot IQRL."""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from stable_baselines3 import PPO

from .. import load_config
from ..data_live import LiveDataManager, StreamConfig
from ..iq_client import IQOptionClient
from ..risk import RiskConfig, RiskManager
from ..utils.logger import get_logger
from ..utils.storage import TradeRecord, TradeStorage
from ..utils.time_sync import TimeSynchronizer
from ..rl.features import build_feature_tensor
from ..rl.policy import MultiTemporalCnnLstmPolicy


@dataclass
class LiveContext:
    client: IQOptionClient
    model: PPO
    storage: TradeStorage
    risk: RiskManager
    min_probability: float
    timeframes: List[int]
    amount: float
    amount_percent: bool
    window_size: int
    assets: List[str]


class LiveRunner:
    def __init__(self, context: LiveContext, assets: List[str]) -> None:
        self.ctx = context
        self.assets = assets
        self.logger = get_logger("live_runner")
        self.sync = TimeSynchronizer()
        self.order_locks: Dict[Tuple[str, int], threading.Lock] = defaultdict(threading.Lock)
        self.pending_orders: Dict[Tuple[str, int], Optional[int]] = defaultdict(lambda: None)
        self.trade_queue: "queue.Queue[TradeRecord]" = queue.Queue()
        self.manager: Optional[LiveDataManager] = None

    def start(self) -> None:
        self.sync.sync()
        stream_config = StreamConfig(
            assets=self.assets,
            timeframes=self.ctx.timeframes,
            buffer=128,
        )
        self.manager = LiveDataManager(
            client=self.ctx.client,
            config=stream_config,
            callback=self.on_candle_close,
        )
        self.manager.start()
        try:
            while True:
                try:
                    trade = self.trade_queue.get(timeout=1.0)
                    self.ctx.storage.append_trade(trade)
                    self.ctx.risk.register_trade(trade.result)
                except queue.Empty:
                    continue
        finally:
            if self.manager:
                self.manager.stop()

    def on_candle_close(self, asset: str, timeframe: int, candles: List[Dict[str, float]]) -> None:
        key = (asset, timeframe)
        lock = self.order_locks[key]
        if not lock.acquire(blocking=False):
            return
        try:
            try:
                observation = self.build_observation()
            except ValueError:
                return
            action, _ = self.ctx.model.predict(observation, deterministic=False)
            obs_tensor, _ = self.ctx.model.policy.obs_to_tensor(observation)
            dist = self.ctx.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            probability = float(probs[action])
            direction = "call" if action == 1 else "put" if action == 2 else "none"
            if action == 0:
                return
            min_prob = max(self.ctx.min_probability, self.ctx.risk.dynamic_threshold)
            if probability < min_prob:
                return
            amount = self.ctx.amount
            if self.ctx.amount_percent:
                balance = self.ctx.client.client.get_balance()  # type: ignore[attr-defined]
                amount = balance * self.ctx.amount
            if not self.ctx.risk.allowed(probability, amount, min_prob):
                return
            expiry = self.sync.next_candle_close(timeframe)
            target = max(self.sync.now(), expiry - 0.5)
            self.sync.wait_until(target)
            success, trade_id = self.ctx.client.buy(asset, amount, direction, timeframe)
            if not success or trade_id is None:
                success, trade_id = self.ctx.client.buy_digital(asset, amount, direction, timeframe)
            if success and trade_id is not None:
                self.logger.info("Orden enviada %s %s tf=%s prob=%.2f", asset, direction, timeframe, probability)
                threading.Thread(
                    target=self.await_result,
                    args=(key, trade_id, amount, probability, direction, timeframe),
                    daemon=True,
                ).start()
        finally:
            lock.release()

    def build_observation(self):
        if self.manager is None:
            raise RuntimeError("LiveDataManager no inicializado")
        data = self.manager.build_observation(window_size=self.ctx.window_size)
        expected = len(self.ctx.assets) * len(self.ctx.timeframes)
        if len(data) < expected:
            raise ValueError("Datos insuficientes para construir observación")
        tensors = build_feature_tensor(data)
        return np.expand_dims(tensors, axis=0)

    def await_result(
        self,
        key: Tuple[str, int],
        trade_id: int,
        amount: float,
        probability: float,
        direction: str,
        timeframe: int,
    ) -> None:
        start = time.time()
        while time.time() - start < 600:
            result = self.ctx.client.check_win(trade_id)
            if result is not None:
                trade = TradeRecord(
                    timestamp=time.time(),
                    asset=key[0],
                    timeframe=timeframe,
                    direction=direction,
                    probability=probability,
                    amount=amount,
                    payout=amount * (probability - 0.5),
                    result=result,
                )
                self.trade_queue.put(trade)
                return
            time.sleep(5)
        self.logger.error("Timeout esperando resultado de %s", trade_id)


def load_model(path: str) -> PPO:
    return PPO.load(path, custom_objects={"policy": MultiTemporalCnnLstmPolicy})


def main() -> None:
    load_dotenv()
    cfg = load_config()
    logger = get_logger("live_main")
    client = IQOptionClient(
        email=os.getenv("IQOPTION_EMAIL", ""),
        password=os.getenv("IQOPTION_PASSWORD", ""),
        account_type=os.getenv("IQOPTION_ACCOUNT_TYPE", "PRACTICE"),
        reconnect_interval=cfg["live"]["reconnect_interval"],
    )
    model_path = Path(cfg["storage"]["artifacts_dir"]) / "models/ppo_latest.zip"
    model = load_model(str(model_path))
    storage = TradeStorage(
        csv_path=cfg["logging"]["csv_path"],
        sqlite_path=cfg["logging"]["sqlite_path"],
    )
    risk = RiskManager(
        RiskConfig(**cfg["risk"])
    )
    context = LiveContext(
        client=client,
        model=model,
        storage=storage,
        risk=risk,
        min_probability=cfg["live"]["min_probability"],
        timeframes=cfg["timeframes"],
        amount=cfg["amount"],
        amount_percent=cfg.get("amount_percent", False),
        window_size=cfg["window_size"],
        assets=cfg["assets"],
    )
    runner = LiveRunner(context, assets=cfg["assets"])
    runner.start()


if __name__ == "__main__":
    main()
