from time import sleep
import gym_maze
import numpy as np

from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SarsaAgent
from src.games.game import Game


class SarsaMazeGame(Game):
    def __init__(self):
        super().__init__('maze-random-10x10-plus-v0')

        self.agent = SarsaAgent(
            self.env,
            'models/sarsa_maze.pkl'
        )

    def run(self,
            num_episodes: int = 150,
            max_epochs: int = 1000,
            verbose: bool = False
            ):
        self.env.render()
        try:
            for episode in range(num_episodes):
                obv = self.env.reset()
                state, total_reward = np.array([*obv]), 0

                for t in range(max_epochs):
                    action = self.agent.get_action(state)
                    new_state, reward, done, _ = self.env.step(action)
                    new_action = self.agent.get_action(new_state)

                    total_reward += reward

                    self.agent.train(state, action, new_state, reward, done, new_action=new_action)

                    state = np.array([*new_state])

                    if done:
                        if verbose:
                            print(f'Episode {episode + 1} finished after {t} time steps with total {total_reward}')

                        break

                self.agent.training_done(episode, total_reward)
        finally:
            self.agent.save_model()

    def test(self,
             num_episodes: int = 1,
             max_epochs: int = 10
             ):
        self.env.render()

        for episode in range(num_episodes):
            state = self.env.reset()

            for t in range(max_epochs):
                sleep(1)
                action = self.agent.get_action(state)
                new_state, reward, done, _ = self.env.step(action)

                self.env.render()

                state = new_state
