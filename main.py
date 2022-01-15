import gym_maze
from src.games.maze_game import MazeGame

game = MazeGame(
    env_name='maze-random-10x10-plus-v0'
)
game.run(
    verbose=True
)
game.test()
