import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from app.rl.env import TradingEnv
from app.rl.policy import SimpleCnnLstmExtractor
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensorboard", action="store_true")
    args = parser.parse_args()

    env = TradingEnv()
    policy_kwargs = dict(features_extractor_class=SimpleCnnLstmExtractor, features_extractor_kwargs=dict(features_dim=64))
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, seed=args.seed)
    model.learn(total_timesteps=args.steps)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_dummy.zip")

if __name__ == "__main__":
    main()