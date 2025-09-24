"""Run PPO model against the trading environment and report metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from stable_baselines3 import PPO

from app.rl.env import TradingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest PPO model")
    parser.add_argument("--model", type=str, default="models/ppo_latest.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--payout", type=float, default=0.8)
    parser.add_argument("--output", type=str, default="reports/backtest.json")
    return parser.parse_args()


def max_drawdown(equity_curve):
    equity = np.array(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    return float(np.max(drawdowns))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    env = TradingEnv(data_dir=args.data_dir, window_size=args.window_size, episode_length=args.episode_length, payout=args.payout)
    model = PPO.load(model_path)

    wins = 0
    trades = 0
    net_profit = 0.0
    win_payout = 0.0
    loss_amount = 0.0
    equity_curve = [0.0]
    per_asset_tf: Dict[Tuple[str, int], Dict[str, float]] = {}

    for _ in range(args.episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if action != 0:
                trades += 1
                key = (info["asset"], int(info["timeframe"]))
                stats = per_asset_tf.setdefault(key, {"trades": 0, "pnl": 0.0})
                stats["trades"] += 1
                stats["pnl"] += reward
                if reward > 0:
                    wins += 1
                    win_payout += reward
                elif reward < 0:
                    loss_amount += reward
            net_profit += reward
            equity_curve.append(net_profit)

    win_rate = wins / trades if trades else 0.0
    profit_factor = (win_payout / abs(loss_amount)) if loss_amount < 0 else float("inf") if win_payout > 0 else 0.0
    dd = max_drawdown(equity_curve)

    report = {
        "model": str(model_path),
        "episodes": args.episodes,
        "total_trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "max_drawdown": dd,
        "per_asset_timeframe": {
            f"{asset}_{tf}": stats for (asset, tf), stats in per_asset_tf.items()
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

