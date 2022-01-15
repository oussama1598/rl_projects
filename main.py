# import gym_maze

from src.games.cartpole_game import CartpoleGame

# from src.games.maze_game import MazeGame

game = CartpoleGame()
game.run(
    verbose=True
)
# game.test()
