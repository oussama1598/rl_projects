import numpy as np


class Agent:
    def __init__(self, env, save_to: str):
        self.env = env
        self.save_to: str = save_to

    def load_model(self):
        raise NotImplemented()

    def save_model(self):
        raise NotImplemented()

    def get_action(self, state: np.array):
        raise NotImplemented()

    def train(self, state: np.array, action: int, new_state: np.array, reward: int, done: bool):
        raise NotImplemented()

    def training_done(self, episode: int, total_reward: float):
        raise NotImplemented()
