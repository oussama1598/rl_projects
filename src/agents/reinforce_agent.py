import os
import random

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam

from src.agents.agent import Agent


class ReinforceAgent(Agent):
    def __init__(self, env,
                 save_to: str):
        super().__init__(env, save_to)

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.gamma = 0.99
        self.alpha = 1e-4
        self.learning_rate = 0.01

        self.model = self._build_model()

        self.load_model()

        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.discounted_rewards = []

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _one_hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_size)
        action_encoded[action] = 1

        return action_encoded

    def _get_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_total_return = 0

        for reward in rewards[::-1]:
            cumulative_total_return = (cumulative_total_return * self.gamma) + reward
            discounted_rewards.insert(0, cumulative_total_return)

        mean_rewards = np.mean(discounted_rewards)
        std_rewards = np.std(discounted_rewards)
        norm_discounted_rewards = (discounted_rewards -
                                   mean_rewards) / (std_rewards + 1e-7)  # avoiding zero div

        return norm_discounted_rewards

    def load_model(self):
        if os.path.isfile(self.save_to):
            self.model.load_weights(self.save_to)

    def save_model(self):
        self.model.save(self.save_to)

    def get_action(self, state):
        state = state.reshape([1, state.shape[0]])

        action_probability_distribution = self.model.predict(state).flatten()
        action_probability_distribution /= np.sum(action_probability_distribution)

        action = np.random.choice(self.action_size, 1,
                                  p=action_probability_distribution)[0]

        return action, action_probability_distribution

    def train(self, state: np.array, action: int, action_prob: np.array, reward: int, done: bool):
        encoded_action = self._one_hot_encode_action(action)

        self.gradients.append(encoded_action - action_prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)

    def training_done(self, episode: int, total_reward: float):
        states = np.vstack(self.states)

        # get Y
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        discounted_rewards = self._get_discounted_rewards(rewards)
        gradients *= discounted_rewards
        gradients = self.alpha * np.vstack([gradients]) + self.probs

        history = self.model.train_on_batch(states, gradients)

        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

        return history
