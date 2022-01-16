import os
import random
from collections import deque

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam

from src.agents.agent import Agent


class DeepQLearningAgent(Agent):
    def __init__(self, env,
                 save_to: str,
                 min_epsilon: float = 0.01,
                 min_learning_rate: float = 0.2,
                 discount_factor: float = 0.95):
        super().__init__(env, save_to)

        self.min_epsilon: float = min_epsilon
        self.min_learning_rate: float = min_learning_rate
        self.discount_factor: float = discount_factor

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.decay = 0.995
        self.epsilon = 1.0
        self.learning_rate = 0.001

        self.target_update_counter = 0

        self.memory = deque(maxlen=2000)
        self.model = self._build_model()

        self.load_model()

        self.rewards = []
        self.epsilons = []

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def load_model(self):
        if os.path.isfile(self.save_to):
            self.model.load_weights(self.save_to)

            self.epsilon = self.min_epsilon

    def save_model(self):
        self.model.save(self.save_to)

    def get_action(self, state: np.array):
        if random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.model.predict(state)[0])

        return self.env.action_space.sample()

    def train(self, state: np.array, action: int, new_state: np.array, reward: int, done: bool):
        self.memory.append((state, action, reward, new_state, done))

    def training_done(self, episode: int, total_reward: float):
        if len(self.memory) < 32:
            return

        sample_batch = random.sample(self.memory, 32)

        for state, action, reward, next_state, done in sample_batch:
            target = reward

            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon:
            self.epsilon *= self.decay

        self.epsilons.append(self.epsilon)
        self.rewards.append(total_reward)
