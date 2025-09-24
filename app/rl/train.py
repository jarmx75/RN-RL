"""CLI entry point for training PPO on the trading environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from app.rl.env import TradingEnv
from app.rl.policy import MultiTFExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for IQ Option trading")
    parser.add_argument("--steps", type=int, default=200_000, help="Total timesteps to train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4, help="Number of vectorized environments")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=288)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--payout", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per environment")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tensorboard-log", type=str, default="runs/ppo")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--features-dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def make_env_fn(data_dir: str, window_size: int, episode_length: int, payout: float):
    def _factory():
        return TradingEnv(data_dir=data_dir, window_size=window_size, episode_length=episode_length, payout=payout)

    return _factory


def main() -> None:
    args = parse_args()
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    vec_env = make_vec_env(
        make_env_fn(args.data_dir, args.window_size, args.episode_length, args.payout),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    eval_env = make_vec_env(
        make_env_fn(args.data_dir, args.window_size, args.episode_length, args.payout),
        n_envs=1,
        seed=args.seed + 1,
    )

    policy_kwargs = dict(
        features_extractor_class=MultiTFExtractor,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
    )

    tensorboard_log = args.tensorboard_log if args.tensorboard else None

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(models_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    model.learn(total_timesteps=args.steps, callback=eval_callback)
    model.save(models_dir / "ppo_latest.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

