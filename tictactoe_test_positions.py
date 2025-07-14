# tictactoe_test_positions.py
# Ten TicTacToe board positions, each one move from victory for either X (1) or O (2).
# Each position is a 3x3 numpy array, and the winning move is provided.
import numpy as np

test_positions = [
    # X to win (row)
    (np.array([[1, 1, 0], [0, 2, 2], [0, 0, 0]]), 2),  # X plays at (0,2)
    (np.array([[2, 2, 0], [1, 1, 0], [0, 0, 0]]), 5),  # X plays at (1,2)
    # O to win (row)
    (np.array([[2, 2, 0], [1, 1, 0], [0, 0, 0]]), 2),  # O plays at (0,2)
    (np.array([[1, 1, 0], [2, 2, 0], [0, 0, 0]]), 5),  # O plays at (1,2)
    # X to win (col)
    (np.array([[1, 2, 0], [1, 2, 0], [0, 0, 0]]), 6),  # X plays at (2,0)
    (np.array([[2, 1, 0], [2, 1, 0], [0, 0, 0]]), 7),  # X plays at (2,1)
    # O to win (col)
    (np.array([[2, 1, 0], [2, 1, 0], [0, 0, 0]]), 6),  # O plays at (2,0)
    (np.array([[1, 2, 0], [1, 2, 0], [0, 0, 0]]), 7),  # O plays at (2,1)
    # X to win (diag)
    (np.array([[1, 2, 0], [0, 1, 2], [0, 0, 0]]), 8),  # X plays at (2,2)
    # O to win (anti-diag)
    (np.array([[1, 1, 2], [0, 2, 0], [0, 0, 0]]), 6),  # O plays at (2,0)
]

# Each tuple: (board, winning_action)
# winning_action is the cell index (0-8) where the winning move should be played.

