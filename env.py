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
        state = np.random.rand(10).astype(np.float32)
        info = {}  # must return info dict
        return state, info

    def step(self, action):
        reward = 1.0 if action == np.random.randint(0, 3) else 0.0
        done = True
        truncated = False
        info = {}
        return np.random.rand(10).astype(np.float32), reward, done, truncated, info


# 2. Traffic Signal Environment
class TrafficSignalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)  # Red, Green, Orange
        self.observation_space = gym.spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = np.array([np.random.randint(0, 20)], dtype=np.int32)
        info = {}
        return state, info

    def step(self, action):
        queue_length = np.random.randint(0, 20)
        reward = 1.0 if queue_length < 10 else 0.0
        done = True
        truncated = False
        info = {}
        return np.array([queue_length], dtype=np.int32), reward, done, truncated, info


# 3. Multi-Intersection Environment
class MultiIntersectionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(6)  # multiple signals
        self.observation_space = gym.spaces.Box(low=0, high=50, shape=(4,), dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = np.random.randint(0, 50, size=(4,), dtype=np.int32)
        info = {}
        return state, info

    def step(self, action):
        cars = np.random.randint(0, 50, size=(4,), dtype=np.int32)
        reward = 1.0 if np.mean(cars) < 25 else 0.5 if np.mean(cars) < 40 else 0.0
        done = True
        truncated = False
        info = {}
        return cars, reward, done, truncated, info
