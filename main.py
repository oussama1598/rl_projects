import matplotlib.pyplot as plt

import gym_maze.gym_maze
from src.agents.q_learning_agent import QLearningAgent
from src.learnable_environment import LearnableEnvironment

env = LearnableEnvironment(
    env_name='maze-random-10x10-plus-v0'
)
agent = QLearningAgent(env)

env.run(
    agent,
    verbose=True
)
env.test(agent)
