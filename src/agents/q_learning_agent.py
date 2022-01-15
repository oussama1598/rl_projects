import math
import random

import numpy as np

from src.agents.agent import Agent


class QLearningAgent(Agent):
    def __init__(self, env,
                 min_epsilon: float = 0.001,
                 min_learning_rate: float = 0.2,
                 discount_factor: float = 0.99):
        super().__init__(env)

        self.min_epsilon: float = min_epsilon
        self.min_learning_rate: float = min_learning_rate
        self.discount_factor: float = discount_factor

        self.max_size = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))
        self.num_buckets = self.max_size
        self.state_bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))

        self.decay = np.prod(self.max_size, dtype=float) / 10.0
        self.epsilon = self._get_epsilon_per_episode(0)
        self.learning_rate = self._get_learning_rate_per_episode(0)

        self.Q = np.zeros(self.max_size + (self.env.action_space.n,), dtype=float)

        self.rewards = []
        self.epsilons = []

    def _get_epsilon_per_episode(self, episode: int):
        return max(self.min_epsilon, min(0.8, 1.0 - math.log10((episode + 1) / self.decay)))

    def _get_learning_rate_per_episode(self, episode: int):
        return max(self.min_learning_rate, min(0.8, 1.0 - math.log10((episode + 1) / self.decay)))

    def get_action(self, state: np.array):
        i, j = state.astype(int)

        return int(np.argmax(self.Q[i, j])) if random.uniform(0, 1) > self.epsilon else self.env.action_space.sample()

    def train(self, state: np.array, action: int, new_state: np.array, reward: float, done: bool):
        state_i, state_j = state.astype(int)
        new_state_i, new_state_j = new_state.astype(int)

        self.Q[state_i, state_j, action] *= (1 - self.learning_rate)
        self.Q[state_i, state_j, action] += self.learning_rate * (
                reward + self.discount_factor * np.max(self.Q[new_state_i, new_state_j]))

    def training_done(self, episode: int, total_reward: float):
        self.epsilon = self._get_epsilon_per_episode(episode)
        self.learning_rate = self._get_learning_rate_per_episode(episode)

        self.epsilons.append(self.epsilon)
        self.rewards.append(total_reward)
