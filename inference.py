import numpy as np
from env import EmailSortEnv   # ya TrafficSignalEnv

def run_baseline():
    env = EmailSortEnv()
    scores = []
    for _ in range(10):  # run 10 episodes
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # random action
            state, reward, done, info = env.step(action)
            scores.append(reward)
    print("Average score:", np.mean(scores))

if __name__ == "__main__":
    run_baseline()