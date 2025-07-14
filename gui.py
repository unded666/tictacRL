import tkinter as tk
from tictactoe_env import TicTacToeEnv

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("TicTacToe Manual Tester")
        self.env = TicTacToeEnv()
        self.buttons = []
        self.status_label = tk.Label(master, text="Player 1's turn (X)")
        self.status_label.grid(row=3, column=0, columnspan=3)
        self.reset_button = tk.Button(master, text="Reset", command=self.reset)
        self.reset_button.grid(row=4, column=0, columnspan=3)
        for i in range(3):
            row = []
            for j in range(3):
                btn = tk.Button(master, text="", width=6, height=3,
                                command=lambda r=i, c=j: self.make_move(r, c))
                btn.grid(row=i, column=j)
                row.append(btn)
            self.buttons.append(row)
        self.reset()

    def make_move(self, row, col):
        if self.env.done:
            return
        action = row * 3 + col
        obs, reward, done, info = self.env.step(action)
        self.update_board()
        if 'invalid' in info:
            self.status_label.config(text="Invalid move! Try again.")
            return
        if done:
            if reward == 1:
                self.status_label.config(text="Player 1 (X) wins!")
            elif reward == -1:
                self.status_label.config(text="Player 2 (O) wins!")
            else:
                self.status_label.config(text="Draw!")
        else:
            player = self.env.current_player
            self.status_label.config(text=f"Player {player}'s turn ({'X' if player == 1 else 'O'})")

    def update_board(self):
        symbols = {0: '', 1: 'X', 2: 'O'}
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text=symbols[self.env.board[i, j]])

    def reset(self):
        self.env.reset()
        self.update_board()
        self.status_label.config(text="Player 1's turn (X)")

if __name__ == "__main__":
    root = tk.Tk()
    gui = TicTacToeGUI(root)
    root.mainloop()

