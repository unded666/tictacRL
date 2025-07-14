import gym
from gym import spaces
import numpy as np

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.action_space = spaces.Discrete(9)  # 9 cells
        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 3), dtype=np.int8)
        self.reset()
        self.record = []

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.record = []
        return self.board.copy()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0 or self.done:
            return self.board.copy(), -10, True, {'invalid': True}
        self.board[row, col] = self.current_player
        self.record.append((self.current_player, action))
        reward, self.done = self._check_game_status()
        info = {'player': self.current_player}
        self.current_player = 2 if self.current_player == 1 else 1
        return self.board.copy(), reward, self.done, info

    def _check_game_status(self):
        for player in [1, 2]:
            for i in range(3):
                if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                    return (1 if player == 1 else -1), True
            if all([self.board[i, i] == player for i in range(3)]) or all([self.board[i, 2 - i] == player for i in range(3)]):
                return (1 if player == 1 else -1), True
        if not (self.board == 0).any():
            return 0, True  # Draw
        return 0, False

    def render(self, mode='human'):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print('\n'.join(' '.join(symbols[cell] for cell in row) for row in self.board))

    def get_record(self):
        return self.record

    def visualize_game(self, record=None):
        if record is None:
            record = self.record
        board = np.zeros((3, 3), dtype=np.int8)
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for move_num, (player, action) in enumerate(record):
            row, col = divmod(action, 3)
            board[row, col] = player
            print(f"Move {move_num+1}: Player {player} -> Cell {action}")
            print('\n'.join(' '.join(symbols[cell] for cell in row) for row in board))
            print()

