import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrafficSignalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=20, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.steps = 0

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 20, size=(4,), dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        traffic = np.sum(self.state)
        reward = 0.9 if (traffic > 15 and action == 1) else 0.1
        done = self.steps >= 3
        return self.state, reward, done, False, {}

    def close(self):
        pass


class EmailSortEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        self.state = np.random.rand(10).astype(np.float32)
        self.correct = np.random.randint(0, 3)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        reward = 0.9 if action == self.correct else 0.1
        done = self.steps >= 3
        return self.state, reward, done, False, {}

    def close(self):
        pass


class MultiIntersectionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=50, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 50, size=(8,), dtype=np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        self.steps += 1
        best = np.argmax(self.state[:6])
        reward = 0.9 if action == best else 0.1
        done = self.steps >= 3
        return self.state, reward, done, False, {}

    def close(self):
        pass
