import numpy as np

from src.agents.deep_q_learning_agent import DeepQLearningAgent
from src.games.game import Game


class DQCartpoleGame(Game):
    def __init__(self, save_to: str = 'models/cartpole.h5', save_model: bool = True):
        super().__init__('CartPole-v1', save_model)

        self.agent = DeepQLearningAgent(
            self.env,
            save_to
        )

    def run(self,
            num_episodes: int = 1000,
            max_epochs: int = 1000,
            verbose: bool = False
            ):

        try:
            for episode in range(num_episodes):
                state, total_reward, done = np.reshape(self.env.reset(),
                                                       [1, self.env.observation_space.shape[0]]), 0, False

                while not done:
                    action = self.agent.get_action(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
                    total_reward += reward

                    self.agent.train(state, action, next_state, reward, done)

                    state = next_state

                if verbose:
                    print(f'Episode {episode + 1} finished with total {total_reward}')

                self.agent.training_done(episode, total_reward)
        finally:
            if self.save_model:
                self.agent.save_model()
