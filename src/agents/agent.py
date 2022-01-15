import numpy as np


class Agent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state: np.array):
        raise NotImplemented()

    def train(self, state: np.array, action: int, new_state: np.array, reward: int):
        raise NotImplemented()

    def training_done(self, episode: int, total_reward: float):
        raise NotImplemented()
