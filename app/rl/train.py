"""Entrenamiento del agente PPO/A2C."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ..data_history import HistoricalDataManager, HistoryRequest
from ..iq_client import IQOptionClient
from ..utils.logger import get_logger
from ..utils.storage import TradeStorage
from .. import load_config
from .env import EnvConfig, IQOptionMultiAssetEnv
from .policy import MultiTemporalCnnLstmPolicy

ALGORITHMS = {"PPO": PPO, "A2C": A2C}


def make_env(history: Dict[Tuple[str, int], pd.DataFrame], config: EnvConfig):
    def _init():
        env = IQOptionMultiAssetEnv(config=config, history=history)
        return env

    return _init


def prepare_history(
    manager: HistoricalDataManager,
    assets: List[str],
    timeframes: List[int],
    window_size: int,
    days: int,
) -> Dict[Tuple[str, int], pd.DataFrame]:
    history: Dict[Tuple[str, int], pd.DataFrame] = {}
    for asset in assets:
        for tf in timeframes:
            df = manager.load(asset, tf)
            if days:
                limit = days * 24 * 60 // max(1, tf // 60)
                df = df.tail(limit + window_size + 1)
            if len(df) < window_size + 10:
                raise ValueError(
                    f"Datos insuficientes para {asset} {tf}s. Ejecuta download_data.sh"
                )
            history[(asset, tf)] = df
    return history


def train_agent(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entrena el agente IQRL")
    parser.add_argument("--assets", type=str, default=None, help="Activos separados por coma")
    parser.add_argument("--tfs", type=str, default=None, help="Timeframes en segundos separados por coma")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--algo", type=str, default="PPO", choices=list(ALGORITHMS))
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    if args is None:
        args = parser.parse_args()

    cfg = load_config()
    assets = args.assets.split(",") if args.assets else cfg["assets"]
    timeframes = [int(tf) for tf in (args.tfs.split(",") if args.tfs else cfg["timeframes"])]
    model_cfg = cfg["model"]
    total_steps = args.total_steps or int(model_cfg["total_steps"])
    window_size = int(cfg.get("window_size", 60))

    logger = get_logger("train")
    logger.info("Iniciando entrenamiento con %s", assets)

    data_dir = cfg["storage"]["data_dir"]
    client = IQOptionClient(
        email=os.getenv("IQOPTION_EMAIL", ""),
        password=os.getenv("IQOPTION_PASSWORD", ""),
        account_type=os.getenv("IQOPTION_ACCOUNT_TYPE", "PRACTICE"),
    )
    history_manager = HistoricalDataManager(client, data_dir=data_dir)
    history = prepare_history(history_manager, assets, timeframes, window_size, args.days)

    env_config = EnvConfig(assets=assets, timeframes=timeframes, window_size=window_size)

    def env_fn():
        return IQOptionMultiAssetEnv(env_config, history)

    num_envs = int(model_cfg.get("num_envs", 1))
    vec_env = SubprocVecEnv([env_fn for _ in range(num_envs)]) if num_envs > 1 else DummyVecEnv([env_fn])

    algo_cls = ALGORITHMS[args.algo]
    algo_kwargs = dict(
        learning_rate=model_cfg.get("learning_rate", 3e-4),
        n_steps=model_cfg.get("n_steps", 2048),
        batch_size=model_cfg.get("batch_size", 256),
        gamma=model_cfg.get("gamma", 0.99),
        gae_lambda=model_cfg.get("gae_lambda", 0.95),
        clip_range=model_cfg.get("clip_range", 0.2),
        ent_coef=model_cfg.get("ent_coef", 0.0),
        vf_coef=model_cfg.get("vf_coef", 0.5),
        max_grad_norm=model_cfg.get("max_grad_norm", 0.5),
        tensorboard_log=("artifacts/tensorboard" if args.tensorboard else None),
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
    )

    model = algo_cls(MultiTemporalCnnLstmPolicy, vec_env, **algo_kwargs)
    model.set_random_seed(model_cfg.get("seed", 42))

    model.learn(total_timesteps=total_steps)
    artifacts = Path(cfg["storage"]["artifacts_dir"]) / "models"
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or artifacts / f"{args.algo.lower()}_latest.zip"
    model.save(model_path)
    logger.info("Modelo guardado en %s", model_path)


if __name__ == "__main__":
    train_agent()
