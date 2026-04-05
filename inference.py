from env import EmailSortEnv, TrafficSignalEnv, MultiIntersectionEnv
import numpy as np

def run_baseline(env_class):
    env = env_class()
    scores = []
    for _ in range(10):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            scores.append(reward)
    print(env_class.__name__, "Average score:", np.mean(scores))

if __name__ == "__main__":
    run_baseline(EmailSortEnv)
    run_baseline(TrafficSignalEnv)
    run_baseline(MultiIntersectionEnv)