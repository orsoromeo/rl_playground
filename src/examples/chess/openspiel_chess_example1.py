# from open_spiel.python import games  # pylint: disable=unused-import
# from open_spiel.python.mfg import games as mfg_games  # pylint: disable=unused-import
# import pyspiel

# for game in pyspiel.registered_games():
#     print(game)


"""Python spiel example."""

import random
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel
import chess
import chess.svg

FLAGS = flags.FLAGS

# Game strings can just contain the name or the name followed by parameters
# and arguments, e.g. "breakthrough(rows=6,columns=6)"
flags.DEFINE_string("game_string", "chess", "Game string")
NUM_MOVES = 10


def main(_):
  games_list = pyspiel.registered_games()
  print("Registered games:")
  print(games_list)

  action_string = None

  print("Creating game: " + FLAGS.game_string)
  game = pyspiel.load_game(FLAGS.game_string)

  # Create the initial state
  state = game.new_initial_state()

  # Print the initial state
  def save_board(fen_state):
    board = chess.Board(str(state))
    boardsvg = chess.svg.board(board=board)
    f = open("BoardVisualisedFromFEN_"+str(i)+".SVG", "w")
    f.write(boardsvg)
    f.close()

  for i in range(0, NUM_MOVES):
    action = random.choice(state.legal_actions(state.current_player()))
    action_string = state.action_to_string(state.current_player(), action)
    print("Player ", state.current_player(), ", randomly sampled action: ",
          action_string)
    state.apply_action(action)
    save_board(state)

  # Game is now done. Print utilities for each player
  returns = state.returns()
  for pid in range(game.num_players()):
    print("Utility for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
  app.run(main)