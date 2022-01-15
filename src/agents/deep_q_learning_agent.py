import math
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
                 min_epsilon: float = 0.01,
                 min_learning_rate: float = 0.2,
                 discount_factor: float = 0.95):
        super().__init__(env)

        self.min_epsilon: float = min_epsilon
        self.min_learning_rate: float = min_learning_rate
        self.discount_factor: float = discount_factor

        self.weight_backup = 'cartpole_weight.h5'
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.decay = 0.995
        self.epsilon = self.min_epsilon
        self.learning_rate = 0.001

        self.target_update_counter = 0

        self.memory = deque(maxlen=2000)
        self.model = self._build_model()

        self.rewards = []
        self.epsilons = []

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=(None, self.state_size), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)

        return model

    def save_model(self):
        self.model.save(self.weight_backup)

    def get_action(self, state: np.array):
        if random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.model.predict(np.array([state])).flatten())

        return self.env.action_space.sample()

    def train(self, state: np.array, action: int, new_state: np.array, reward: int, done: bool):
        self.memory.append((state, action, reward, new_state, done))

    def training_done(self, episode: int, total_reward: float):
        if len(self.memory) < 32:
            return

        sample_batch = random.sample(self.memory, 32)

        current_states = np.array([frame[0] for frame in sample_batch])
        current_qs = self.model.predict(current_states)

        new_states = np.array([frame[3] for frame in sample_batch])
        new_qs = self.model.predict(new_states)

        training_X = []
        training_Y = []

        for i, (state, action, reward, next_state, done) in enumerate(sample_batch):
            target = reward

            if not done:
                target = reward + self.discount_factor * np.amax(new_qs[i][0])

            target_q = current_qs[i]
            target_q[0][action] = target

            training_X.append(state)
            training_Y.append(target_q)

        self.model.fit(np.array(training_X), np.array(training_Y), batch_size=32, verbose=False)

        if self.epsilon > self.epsilon:
            self.epsilon *= self.decay

        self.epsilons.append(self.epsilon)
        self.rewards.append(total_reward)
