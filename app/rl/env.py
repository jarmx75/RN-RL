"""Gymnasium environment modelling multi-asset binary options trading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding


def _normalise_ohlcv(data: np.ndarray) -> np.ndarray:
    prices = data[:, :4].astype(np.float32)
    close = prices[:, 3:4]
    close[close == 0] = 1.0
    prices = (prices / close) - 1.0
    volume = data[:, 4].astype(np.float32)
    vol_std = float(volume.std() or 1.0)
    volume = (volume - float(volume.mean())) / vol_std
    return np.column_stack((prices, volume.reshape(-1, 1)))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _generate_synthetic_history(length: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    price = 1.0
    rows: List[Tuple[float, float, float, float, float, float]] = []
    ts = 1_000_000.0
    for _ in range(length):
        change = rng.normal(0, 0.0008)
        o = price
        c = max(1e-4, o + change)
        high = max(o, c) + abs(rng.normal(0, 0.0005))
        low = max(1e-4, min(o, c) - abs(rng.normal(0, 0.0005)))
        vol = max(1.0, rng.normal(10, 3))
        rows.append((o, high, low, c, vol, ts))
        price = c
        ts += 60
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume", "epoch"])


@dataclass
class EnvConfig:
    assets: List[str]
    timeframes: List[int]
    window_size: int = 64
    episode_length: int = 288  # approx one trading day at 5m
    data_dir: str = "data"
    payout: float = 0.8


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        assets: Optional[Iterable[str]] = None,
        timeframes: Optional[Iterable[int]] = None,
        window_size: int = 64,
        episode_length: int = 288,
        data_dir: str = "data",
        payout: float = 0.8,
    ) -> None:
        super().__init__()
        self.config = EnvConfig(
            assets=list(assets or ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]),
            timeframes=list(timeframes or [60, 300, 900]),
            window_size=int(window_size),
            episode_length=int(episode_length),
            data_dir=data_dir,
            payout=float(payout),
        )
        self.config.assets = list(dict.fromkeys(self.config.assets))
        self.config.timeframes = sorted(set(int(tf) for tf in self.config.timeframes))

        self._feature_dim = 6  # 5 OHLCV + focus flag
        self.action_space = spaces.Discrete(3, seed=None)
        obs_shape = (
            len(self.config.assets),
            len(self.config.timeframes),
            self.config.window_size,
            self._feature_dim,
        )
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=obs_shape, dtype=np.float32)

        self._raw_data: Dict[Tuple[str, int], np.ndarray] = {}
        self._norm_data: Dict[Tuple[str, int], np.ndarray] = {}
        self._offsets: Dict[Tuple[str, int], int] = {}
        self._max_steps: int = 0
        self._focus_schedule: List[Tuple[str, int]] = [
            (asset, tf) for asset in self.config.assets for tf in self.config.timeframes
        ]
        self._focus_index = 0
        self._current_step = 0

        self._load_dataset()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self._determine_offsets()
        self._current_step = 0
        self._focus_index = 0

        observation = self._build_observation()
        info = {"asset": self._focus_schedule[self._focus_index][0], "timeframe": self._focus_schedule[self._focus_index][1]}
        return observation, info

    def step(self, action: int):
        assert self.action_space.contains(action)

        asset, timeframe = self._focus_schedule[self._focus_index]
        offset = self._offsets[(asset, timeframe)] + self._current_step
        reward = 0.0

        # compute reward using raw close prices
        raw = self._raw_data[(asset, timeframe)]
        current_close = raw[offset + self.config.window_size - 1, 3]
        next_close = raw[offset + self.config.window_size, 3]

        if action == 1:  # CALL
            reward = self.config.payout if next_close > current_close else -1.0 if next_close < current_close else 0.0
        elif action == 2:  # PUT
            reward = self.config.payout if next_close < current_close else -1.0 if next_close > current_close else 0.0

        info = {
            "asset": asset,
            "timeframe": timeframe,
            "step": self._current_step,
            "action": action,
            "current_close": float(current_close),
            "next_close": float(next_close),
        }

        self._advance_focus()
        self._current_step += 1

        terminated = self._current_step >= self._max_steps
        truncated = False
        observation = self._build_observation()
        return observation, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _advance_focus(self) -> None:
        self._focus_index = (self._focus_index + 1) % len(self._focus_schedule)

    def _determine_offsets(self) -> None:
        available = []
        for key, raw in self._raw_data.items():
            length = raw.shape[0]
            max_start = length - (self.config.window_size + 1 + self.config.episode_length)
            available.append(max_start)
        max_start_all = min(available)
        if max_start_all < 0:
            raise RuntimeError("Not enough data for the configured window/episode length")
        base_start = int(self.np_random.integers(0, max_start_all + 1)) if max_start_all > 0 else 0
        self._offsets = {key: base_start for key in self._raw_data}
        self._max_steps = min(
            self.config.episode_length,
            min(raw.shape[0] - (self.config.window_size + 1 + base_start) for raw in self._raw_data.values()),
        )

    def _build_observation(self) -> np.ndarray:
        focus_asset, focus_tf = self._focus_schedule[self._focus_index]
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        for asset_idx, asset in enumerate(self.config.assets):
            for tf_idx, timeframe in enumerate(self.config.timeframes):
                key = (asset, timeframe)
                data = self._norm_data[key]
                start = self._offsets[key] + self._current_step
                end = start + self.config.window_size
                window = data[start:end]
                obs[asset_idx, tf_idx, :, :5] = window
                if asset == focus_asset and timeframe == focus_tf:
                    obs[asset_idx, tf_idx, :, 5] = 1.0
        return obs

    def _load_dataset(self) -> None:
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        for asset in self.config.assets:
            for timeframe in self.config.timeframes:
                path = data_dir / f"{asset}_{timeframe}.csv"
                try:
                    df = _load_csv(path)
                except FileNotFoundError:
                    df = _generate_synthetic_history()
                    df.to_csv(path, index=False)
                if df.empty:
                    df = _generate_synthetic_history()
                values = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
                self._raw_data[(asset, timeframe)] = values
                self._norm_data[(asset, timeframe)] = _normalise_ohlcv(values)


__all__ = ["TradingEnv", "EnvConfig"]

