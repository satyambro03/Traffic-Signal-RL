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
        self.action_space = spaces.Discrete(2)

        self.state = None

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 10, size=(4,), dtype=np.float32)
        return self.state, {}

    def step(self, action):
        truncated = False
        done = True   # single-step episode

        traffic_load = np.sum(self.state)

        # RL logic: high traffic → green (1)
        if traffic_load > 15 and action == 1:
            reward = 0.9
        else:
            reward = 0.1

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
        self.action_space = spaces.Discrete(3)

        self.state = None
        self.correct_label = None

    def reset(self, seed=None, options=None):
        self.state = np.random.rand(3).astype(np.float32)
        self.correct_label = np.random.randint(0, 3)
        return self.state, {}

    def step(self, action):
        truncated = False
        done = True   # single-step

        if action == self.correct_label:
            reward = 0.9
        else:
            reward = 0.1

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
        truncated = False
        done = True   # single-step

        best_action = np.argmax(self.state)

        if action == best_action:
            reward = 0.9
        else:
            reward = 0.1

        return self.state, reward, done, truncated, {}

    def close(self):
        pass
