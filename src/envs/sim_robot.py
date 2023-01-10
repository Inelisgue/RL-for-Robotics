import numpy as np

class SimRobotEnv:
    def __init__(self):
        self.state = np.zeros(4)
        self.target = np.random.rand(2)

    def step(self, action):
        self.state[:2] += action
        reward = -np.linalg.norm(self.state[:2] - self.target)
        done = np.linalg.norm(self.state[:2] - self.target) < 0.1
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.zeros(4)
        self.target = np.random.rand(2)
        return self.state
