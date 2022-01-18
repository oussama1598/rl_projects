# from src.games.dq_cartpole_game import DQCartpoleGame

# from src.games.maze_game import MazeGame
#from src.games.rpg_cartpole_game import RPGCartpoleGame
from src.games.sarsa_maze_game import SarsaMazeGame

game = SarsaMazeGame()
game.run(
    verbose=True
)
# game.test()
