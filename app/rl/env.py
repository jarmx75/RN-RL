"""Entorno Gymnasium multi-activo para IQ Option."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .features import build_feature_tensor


@dataclass
class EnvConfig:
    assets: List[str]
    timeframes: List[int]
    window_size: int
    payout: float = 0.8
    action_penalty: float = -0.01


Observation = np.ndarray
Action = int


class IQOptionMultiAssetEnv(gym.Env[Observation, Action]):
    """Entorno que consume datos históricos para entrenamiento RL."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvConfig, history: Dict[Tuple[str, int], pd.DataFrame]):
        super().__init__()
        self.config = config
        self.history = history
        self.indices: Dict[Tuple[str, int], int] = {key: config.window_size for key in history}
        self.max_steps = min(len(df) for df in history.values()) - config.window_size - 1
        self.current_step = 0
        self.balance = 0.0

        num_pairs = len(history)
        feature_dim = 6
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(num_pairs, config.window_size, feature_dim),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

    def _get_slice(self) -> Dict[Tuple[str, int], np.ndarray]:
        window = {}
        for key, df in self.history.items():
            idx = self.indices[key]
            window[key] = df.iloc[idx - self.config.window_size : idx][
                ["open", "close", "max", "min", "volume"]
            ].to_numpy(dtype=np.float32)
        return window

    def _get_price_move(self, key: Tuple[str, int]) -> float:
        df = self.history[key]
        idx = self.indices[key]
        close_current = df.iloc[idx - 1]["close"]
        close_next = df.iloc[idx]["close"]
        return float(close_next - close_current)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 0.0
        self.indices = {key: self.config.window_size for key in self.history}
        return self._obs(), {}

    def _obs(self) -> Observation:
        slices = self._get_slice()
        return build_feature_tensor(slices)

    def step(self, action: Action):
        reward = 0.0
        terminated = False
        info: Dict[str, float] = {}
        if action != 0:
            direction = 1 if action == 1 else -1
            total_reward = 0.0
            for key in self.history:
                move = self._get_price_move(key)
                payout = self.config.payout
                if direction * move > 0:
                    total_reward += payout
                else:
                    total_reward -= 1.0
            reward += total_reward / len(self.history)
        else:
            reward += self.config.action_penalty

        self.balance += reward
        self.current_step += 1
        for key in self.history:
            self.indices[key] += 1
        if self.current_step >= self.max_steps:
            terminated = True
        return self._obs(), reward, terminated, False, info

    def render(self):  # pragma: no cover - solo para debugging
        print(f"Step {self.current_step} balance={self.balance:.2f}")
