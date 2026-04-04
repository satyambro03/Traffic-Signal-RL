import gym
from gym import spaces
import numpy as np

class EmailSortEnv(gym.Env):
    def __init__(self):
        super(EmailSortEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Work, Personal, Spam
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

    def reset(self):
        self.state = np.random.rand(10)
        return self.state

    def step(self, action):
        reward = 1.0 if action == np.argmax(self.state) else 0.0
        done = True
        return self.state, reward, done, {}
