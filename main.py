# from src.games.dq_cartpole_game import DQCartpoleGame

# from src.games.maze_game import MazeGame
from src.games.rpg_cartpole_game import RPGCartpoleGame

game = RPGCartpoleGame()
game.run(
    verbose=True
)
# game.test()
