import gymnasium as gym
import numpy as np

# 1. Email Sorting Environment
class EmailSortEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)  # Work, Personal, Spam
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.observation_space.sample()  # always valid
        info = {}
        return state, info

    def step(self, action):
        reward = 1.0 if action == np.random.randint(0, 3) else 0.0
        done = True
        truncated = False
        info = {}
        next_state = self.observation_space.sample()
        return next_state, reward, done, truncated, info


# 2. Traffic Signal Environment
class TrafficSignalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)  # Red, Green, Orange
        self.observation_space = gym.spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.observation_space.sample()
        info = {}
        return state, info

    def step(self, action):
        reward = 1.0 if np.random.randint(0, 20) < 10 else 0.0
        done = True
        truncated = False
        info = {}
        next_state = self.observation_space.sample()
        return next_state, reward, done, truncated, info


# 3. Multi-Intersection Environment
class MultiIntersectionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(6)  # multiple signals
        self.observation_space = gym.spaces.Box(low=0, high=50, shape=(4,), dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.observation_space.sample()
        info = {}
        return state, info

    def step(self, action):
        cars = self.observation_space.sample()
        avg_density = np.mean(cars)
        reward = 1.0 if avg_density < 25 else 0.5 if avg_density < 40 else 0.0
        done = True
        truncated = False
        info = {}
        return cars, reward, done, truncated, info
