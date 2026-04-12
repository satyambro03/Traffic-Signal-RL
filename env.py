import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ==========================================
# Traffic Signal Environment
# ==========================================
class TrafficSignalEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 10, size=(4,), dtype=np.float32)
        return self.state, {}

    def step(self, action):
        traffic = np.sum(self.state)

        if traffic > 15 and action == 1:
            reward = 0.9
        else:
            reward = 0.1

        return self.state, reward, True, False, {}

    def close(self):
        pass


# ==========================================
# Email Sort Environment
# ==========================================
class EmailSortEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        self.state = np.random.rand(3).astype(np.float32)
        self.correct = np.random.randint(0, 3)
        return self.state, {}

    def step(self, action):
        if action == self.correct:
            reward = 0.9
        else:
            reward = 0.1

        return self.state, reward, True, False, {}

    def close(self):
        pass


# ==========================================
# Multi Intersection Environment
# ==========================================
class MultiIntersectionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=20, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 20, size=(6,), dtype=np.float32)
        return self.state, {}

    def step(self, action):
        best = np.argmax(self.state)

        if action == best:
            reward = 0.9
        else:
            reward = 0.1

        return self.state, reward, True, False, {}

    def close(self):
        pass
