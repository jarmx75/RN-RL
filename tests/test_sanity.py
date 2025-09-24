"""Pruebas de humo para iqrl-bot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.rl.env import EnvConfig, IQOptionMultiAssetEnv


def test_env_step_shape():
    history = {}
    assets = ["EURUSD", "GBPUSD"]
    timeframes = [60, 300]
    window = 10
    for asset in assets:
        for tf in timeframes:
            data = pd.DataFrame(
                {
                    "open": np.linspace(1, 1.5, 100),
                    "close": np.linspace(1.01, 1.6, 100),
                    "max": np.linspace(1.02, 1.7, 100),
                    "min": np.linspace(0.99, 1.4, 100),
                    "volume": np.random.randint(1, 50, size=100),
                }
            )
            history[(asset, tf)] = data
    env = IQOptionMultiAssetEnv(EnvConfig(assets, timeframes, window), history)
    obs, _ = env.reset()
    assert obs.shape == (len(history), window, 6)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs.shape == (len(history), window, 6)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
