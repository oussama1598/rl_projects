import numpy as np

from src.agents.reinforce_agent import ReinforceAgent
from src.games.game import Game


class RPGCartpoleGame(Game):
    def __init__(self, save_to: str = 'models/maze.pkl'):
        super().__init__('CartPole-v1')

        self.agent = ReinforceAgent(
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
                state, total_reward, done = self.env.reset(), 0, False

                while not done:
                    action, prob = self.agent.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)

                    self.agent.train(state, action, prob, reward, done)

                    state = next_state
                    total_reward += reward

                    if done:
                        if episode % 1 == 0:
                            self.agent.training_done(episode, total_reward)

                if verbose:
                    print(f'Episode {episode + 1} finished with total {total_reward}')

        finally:
            self.agent.save_model()
