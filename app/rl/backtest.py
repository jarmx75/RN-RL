"""Backtesting del modelo entrenado."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .. import load_config
from ..data_history import HistoricalDataManager
from ..iq_client import IQOptionClient
from ..utils.logger import get_logger
from ..utils.storage import TradeRecord, TradeStorage
from .env import EnvConfig, IQOptionMultiAssetEnv
from .policy import MultiTemporalCnnLstmPolicy


def run_backtest(model_path: str, days: int = 30) -> pd.DataFrame:
    cfg = load_config()
    assets = cfg["assets"]
    timeframes = cfg["timeframes"]
    window = cfg["window_size"]
    client = IQOptionClient(
        email=os.getenv("IQOPTION_EMAIL", ""),
        password=os.getenv("IQOPTION_PASSWORD", ""),
        account_type=os.getenv("IQOPTION_ACCOUNT_TYPE", "PRACTICE"),
    )
    history_manager = HistoricalDataManager(client, cfg["storage"]["data_dir"])
    history: Dict[Tuple[str, int], pd.DataFrame] = {}
    for asset in assets:
        for tf in timeframes:
            df = history_manager.load(asset, tf)
            history[(asset, tf)] = df.tail(days * (60 * 24 // (tf // 60)))
    env_config = EnvConfig(assets=assets, timeframes=timeframes, window_size=window)
    env = IQOptionMultiAssetEnv(env_config, history)

    model = PPO.load(model_path, custom_objects={"policy": MultiTemporalCnnLstmPolicy})

    obs, _ = env.reset()
    done = False
    records = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
        records.append(
            {
                "step": len(records),
                "reward": reward,
                "balance": env.balance,
            }
        )
    df = pd.DataFrame(records)
    report = {
        "win_rate": (df["reward"] > 0).mean(),
        "profit_factor": df[df["reward"] > 0]["reward"].sum()
        / abs(df[df["reward"] < 0]["reward"].sum() or 1.0),
        "max_drawdown": (df["balance"].cummax() - df["balance"]).max(),
        "pl": df["reward"].sum(),
    }
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    df.to_csv("artifacts/reports/backtest.csv", index=False)
    pd.DataFrame([report]).to_csv("artifacts/reports/summary.csv", index=False)
    return pd.DataFrame([report])


def main():
    parser = argparse.ArgumentParser(description="Backtest del modelo IQRL")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    report = run_backtest(args.model_path, days=args.days)
    print(report)


if __name__ == "__main__":
    main()
