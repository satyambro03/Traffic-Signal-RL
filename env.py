import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Easy Task: Email Sorting
class EmailSortEnv(gym.Env):
    def __init__(self):
        super(EmailSortEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Work, Personal, Spam
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.state = np.random.rand(10)
        return self.state, {}

    def step(self, action):
        reward = 1.0 if action == np.argmax(self.state) else 0.0
        done = True
        return self.state, reward, done, False, {}

# Medium Task: Single Traffic Signal
class TrafficSignalEnv(gym.Env):
    def __init__(self):
        super(TrafficSignalEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Red, Green, Orange
        self.observation_space = spaces.Box(low=0, high=20, shape=(4,), dtype=np.int32)

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 20, size=(4,))
        return self.state, {}

    def step(self, action):
        reward = 1.0 if action == np.argmin(self.state) else 0.0
        done = True
        return self.state, reward, done, False, {}

# Hard Task: Multi-Intersection Optimization
class MultiIntersectionEnv(gym.Env):
    def __init__(self):
        super(MultiIntersectionEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # multiple signals
        self.observation_space = spaces.Box(low=0, high=50, shape=(8,), dtype=np.int32)

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 50, size=(8,))
        return self.state, {}

    def step(self, action):
        reward = 1.0 if action == np.argmin(self.state) else 0.0
        done = True
        return self.state, reward, done, False, {}