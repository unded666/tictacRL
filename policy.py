import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class Policy:
    def select_action(self, board, player):
        """Given the current board and player, return an action (0-8)."""
        raise NotImplementedError

class RandomPolicy(Policy):
    def select_action(self, board, player):
        available = np.where(board.flatten() == 0)[0]
        return np.random.choice(available)

class BenchmarkRandomPolicy(Policy):
    """
    A random policy for benchmarking. Always selects a random valid move.
    """
    def select_action(self, board, player):
        available = np.where(board.flatten() == 0)[0]
        return int(np.random.choice(available))

class DQN(nn.Module):
    def __init__(self, input_dim=9, output_dim=9):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNPolicy(Policy):
    def __init__(self, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.steps = 0

    def select_action(self, board, player):
        state = torch.FloatTensor(board.flatten()).unsqueeze(0).to(self.device)
        available = np.where(board.flatten() == 0)[0]
        if random.random() < self.epsilon:
            return np.random.choice(available)
        with torch.no_grad():
            q_values = self.model(state).cpu().numpy().flatten()
        # Mask invalid actions
        q_values[[i for i in range(9) if i not in available]] = -np.inf
        return int(np.argmax(q_values))

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
