import gymnasium as gym
from app.rl.env import TradingEnv
from stable_baselines3 import PPO
import json
import os

def main():
    env = TradingEnv()
    model_path = "models/ppo_dummy.zip"
    if not os.path.isfile(model_path):
        print("No model found! Run train first.")
        return
    model = PPO.load(model_path)
    obs, _ = env.reset()
    wins, trades, pf = 0, 0, 0
    rewards = []
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        if reward > 0:
            wins += 1
        trades += 1
        rewards.append(reward)
        if done:
            break
    win_rate = wins / trades if trades else 0
    profit_factor = sum([r for r in rewards if r > 0]) / abs(sum([r for r in rewards if r < 0])) if any([r < 0 for r in rewards]) else 0
    report = dict(win_rate=win_rate, profit_factor=profit_factor, trades=trades)
    os.makedirs("reports", exist_ok=True)
    with open("reports/backtest.json", "w") as f:
        json.dump(report, f)
    print("Backtest:", report)

if __name__ == "__main__":
    main()