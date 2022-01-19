from typing import Optional

import gym

from src.agents.agent import Agent


class Game:
    def __init__(self, env_name: str, save_model: bool):
        self.env = gym.make(env_name)

        self.agent: Optional[Agent] = None
        self.save_model: bool = save_model
