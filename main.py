from src.games.dq_cartpole_game import DQCartpoleGame
from src.games.maze_game import MazeGame
from src.games.rpg_cartpole_game import RPGCartpoleGame
from src.games.sarsa_maze_game import SarsaMazeGame

game1 = MazeGame()
game1.run(verbose=True)

game2 = DQCartpoleGame()
game2.run(verbose=True)

game3 = SarsaMazeGame()
game3.run(verbose=True)

game4 = RPGCartpoleGame()
game4.run(verbose=True)
