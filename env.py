import gymnasium as gym
import numpy as np

# 1. Email Sorting Environment
class EmailSortEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 3 categories: Work, Personal, Spam
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = np.random.rand(10).astype(np.float32)
        info = {}  # must return info dict
        return state, info

    def step(self, action):
        # reward = 1 if correct classification, else 0
        reward = 1.0 if action == np.random.randint(0, 3) else 0.0
        done = True
        truncated = False
        info = {}
        next_state = np.random.rand(10).astype(np.float32)
        return next_state, reward, done, truncated, info


# 2. Traffic Signal Environment
class TrafficSignalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 3 actions: Red, Green, Orange
        self.action_space = gym.spaces.Discrete(3)
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
        next_state = np.array([queue_length], dtype=np.int32)
        return next_state, reward, done, truncated, info


# 3. Multi-Intersection Environment
class MultiIntersectionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 6 actions: controlling multiple signals
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=0, high=50, shape=(4,), dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = np.random.randint(0, 50, size=(4,), dtype=np.int32)
        info = {}
        return state, info

    def step(self, action):
        cars = np.random.randint(0, 50, size=(4,), dtype=np.int32)
        avg_density = np.mean(cars)
        if avg_density < 25:
            reward = 1.0
        elif avg_density < 40:
            reward = 0.5
        else:
            reward = 0.0
        done = True
        truncated = False
        info = {}
        return cars, reward, done, truncated, info
