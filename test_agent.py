import numpy as np
from tictactoe_env import TicTacToeEnv
from policy import DQNPolicy, BenchmarkRandomPolicy
from tictactoe_test_positions import test_positions

MODEL_PATH = "dqn_agent1.pth"

def test_agent_on_positions(agent, test_positions):
    env = TicTacToeEnv()
    correct = 0
    # Store and set epsilon to 0 if agent has epsilon attribute
    has_epsilon = hasattr(agent, 'epsilon')
    if has_epsilon:
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0
    for idx, (board, winning_action) in enumerate(test_positions):
        env.board = board.copy()
        # Infer current player: X=1, O=2
        x_count = np.sum(env.board == 1)
        o_count = np.sum(env.board == 2)
        if x_count == o_count:
            env.current_player = 1  # X to play
        else:
            env.current_player = 2  # O to play
        env.done = False
        action = agent.select_action(env.board, env.current_player)
        print(f"Test {idx+1}: Agent chose {action}, expected {winning_action} (Player {env.current_player})")
        if action == winning_action:
            correct += 1
    if has_epsilon:
        agent.epsilon = old_epsilon  # Restore epsilon
    print(f"\nAgent chose the correct winning move {correct}/" + str(len(test_positions)) + " times.")
    return correct

if __name__ == "__main__":
    MODEL_PATH2 = "dqn_agent2.pth"
    MODEL_PATH3 = "dqn_agent3.pth"
    old_agent = DQNPolicy()
    old_agent.load(MODEL_PATH2)
    new_agent = DQNPolicy()
    newest_agent = DQNPolicy()
    newest_agent.load(MODEL_PATH3)

    random_agent = BenchmarkRandomPolicy()
    test_agent_on_positions(newest_agent, test_positions)

