import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ==========================================
# 1. Traffic Signal Environment
# ==========================================
class TrafficSignalEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0: Red, 1: Green

        self.state = None

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 10, size=(4,), dtype=np.float32)
        return self.state, {}

    def step(self, action):
        done = True
        truncated = False

        # Simple logic: if traffic high → green is good
        traffic_load = np.sum(self.state)

        if traffic_load > 15 and action == 1:
            reward = 0.9   # correct
        else:
            reward = 0.1   # wrong

        return self.state, reward, done, truncated, {}

    def close(self):
        pass


# ==========================================
# 2. Email Sorting Environment
# ==========================================
class EmailSortEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Work, 1: Personal, 2: Spam

        self.state = None
        self.correct_label = None

    def reset(self, seed=None, options=None):
        self.state = np.random.rand(3).astype(np.float32)
        self.correct_label = np.random.randint(0, 3)
        return self.state, {}

    def step(self, action):
        done = True
        truncated = False

        if action == self.correct_label:
            reward = 0.9   # correct
        else:
            reward = 0.1   # wrong

        return self.state, reward, done, truncated, {}

    def close(self):
        pass


# ==========================================
# 3. Multi Intersection Environment
# ==========================================
class MultiIntersectionEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=0, high=20, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

        self.state = None

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 20, size=(6,), dtype=np.float32)
        return self.state, {}

    def step(self, action):
        done = True
        truncated = False

        # Simple rule: choose index with max traffic
        best_action = np.argmax(self.state)

        if action == best_action:
            reward = 0.9   # correct
        else:
            reward = 0.1   # wrong

        return self.state, reward, done, truncated, {}

    def close(self):
        pass
