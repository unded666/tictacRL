import numpy as np
from tictactoe_env import TicTacToeEnv
from policy import DQNPolicy, RandomPolicy
import torch

NUM_EPISODES = 10000
TARGET_UPDATE = 50
FILE_NAME = "dqn_agent3.pth"

if __name__ == "__main__":
    env = TicTacToeEnv()
    agent1 = DQNPolicy(epsilon_start=0.4, epsilon_end=0.05)
    agent2 = DQNPolicy(epsilon_start=0.4, epsilon_end=0.05)  # You can use DQNPolicy() for both for full self-play

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        transitions = []
        current_agent = agent1
        other_agent = agent2
        player = 1
        while not done:
            action = current_agent.select_action(state, player)
            next_state, reward, done, info = env.step(action)
            # For DQN, reward is from the perspective of the acting agent
            transitions.append((state.flatten(), action, reward, next_state.flatten(), done, player))
            state = next_state
            # Swap agents and player
            current_agent, other_agent = other_agent, current_agent
            player = 2 if player == 1 else 1
        # Store transitions for both agents
        for s, a, r, ns, d, p in transitions:
            if p == 1:
                agent1.store(s, a, r, ns, d)
            else:
                if isinstance(agent2, DQNPolicy):
                    agent2.store(s, a, -r, ns, d)  # Invert reward for player 2
        agent1.update()
        if isinstance(agent2, DQNPolicy):
            agent2.update()
        if episode % TARGET_UPDATE == 0:
            agent1.update_target()
            if isinstance(agent2, DQNPolicy):
                agent2.update_target()
        if episode % 500 == 0:
            print(f"Episode {episode}")
    agent1.save(FILE_NAME)
    print(f"Training complete. Model saved as {FILE_NAME}")

