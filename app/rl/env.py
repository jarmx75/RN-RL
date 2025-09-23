import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, window_size: int = 10):
        super().__init__()
        self.window_size = window_size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 5), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=NO_OP, 1=CALL, 2=PUT
        self.reset()

    def reset(self, seed=None, options=None):
        self.data = self._generate_data()
        self.idx = self.window_size
        obs = self.data[self.idx - self.window_size:self.idx]
        return obs, {}

    def _generate_data(self):
        price = 1.0
        data = []
        for _ in range(100):
            change = np.random.normal(0, 0.001)
            o = price
            c = price + change
            h = max(o, c) + abs(np.random.normal(0, 0.0005))
            l = min(o, c) - abs(np.random.normal(0, 0.0005))
            v = np.random.randint(1, 10)
            data.append([o, h, l, c, v])
            price = c
        return np.array(data, dtype=np.float32)

    def step(self, action):
        done = self.idx >= len(self.data) - 1
        reward = 0.0
        if action == 0:
            reward = 0.0
        elif action == 1: # CALL
            if self.data[self.idx][3] > self.data[self.idx - 1][3]:
                reward = 0.8
            else:
                reward = -1.0
        elif action == 2: # PUT
            if self.data[self.idx][3] < self.data[self.idx - 1][3]:
                reward = 0.8
            else:
                reward = -1.0
        self.idx += 1
        obs = self.data[self.idx - self.window_size:self.idx]
        return obs, reward, done, False, {}