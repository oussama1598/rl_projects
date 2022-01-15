from time import sleep

import gym
import numpy as np

from src.agents.agent import Agent


class LearnableEnvironment:
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)

    def run(self, agent: Agent,
            num_episodes: int = 150,
            max_epochs: int = 1000,
            verbose: bool = False
            ):
        self.env.render()

        for episode in range(num_episodes):
            obv = self.env.reset()
            state, total_reward = np.array([*obv]), 0

            for t in range(max_epochs):
                action = agent.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                agent.train(state, action, new_state, reward)

                state = np.array([*new_state])

                if done:
                    if verbose:
                        print(f'Episode {episode + 1} finished after {t} time steps with total {total_reward}')

                    break

            agent.training_done(episode, total_reward)

    def test(self, agent: Agent,
             num_episodes: int = 1,
             max_epochs: int = 10
             ):
        self.env.render()

        for episode in range(num_episodes):
            state = self.env.reset()

            for t in range(max_epochs):
                sleep(1)
                action = agent.get_action(state)
                new_state, reward, done, _ = self.env.step(action)

                self.env.render()

                state = new_state
