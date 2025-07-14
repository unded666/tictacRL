import numpy as np

class Policy:
    def select_action(self, board, player):
        """Given the current board and player, return an action (0-8)."""
        raise NotImplementedError

class RandomPolicy(Policy):
    def select_action(self, board, player):
        available = np.where(board.flatten() == 0)[0]
        return np.random.choice(available)

