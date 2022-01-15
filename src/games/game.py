from typing import Optional

import gym

from src.agents.agent import Agent


class Game:
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)

        self.agent: Optional[Agent] = None
