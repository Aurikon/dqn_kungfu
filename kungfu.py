import gymnasium as gym
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import tracemalloc
tracemalloc.start()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    print("Using cuda")

class DeepQLearning():

    class DQN(nn.Module):
        def __init__(self, input_shape, output_shape):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

            self.fc1 = nn.Linear(22528, 512)
            self.fc2 = nn.Linear(512, output_shape)


        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x) 


    def __init__(self, env_id, max_steps, episodes):
        self.env = gym.make(env_id)#, render_mode='human')
        self.env.metadata['render_fps'] = 30

        self.max_steps = 100000
        self.episodes = 1000000
        self.epsilon = 1
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.95
        self.gamma = 0.95

        self.dqn = self.DQN(self.env.observation_space.shape, 14).to(device)
        self.target_dqn = self.DQN(self.env.observation_space.shape, 14).to(device)

        self.batch_size = 32

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)

        self.memory = deque(maxlen=10000)


    def __del__(self):
        self.env.close()

    def best_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            q_values = self.dqn(state)
        return torch.argmax(q_values).item()

    def epsilon_greedy_policy(self, state, epsilon):
        random_num = random.uniform(0, 1)
        if (random_num > epsilon):
            return self.best_action(state)
        else:
            return self.env.action_space.sample()

    def update_q(self, train_set):
        states, actions, rewards, new_states, terminate = zip(*train_set)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        new_states = torch.tensor(np.array(new_states), dtype=torch.float32).to(device)
        terminate = torch.tensor(terminate, dtype=torch.float32).to(device)

        q_values = self.dqn(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_q_values = self.target_dqn(new_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - terminate))

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        epsilon = self.epsilon
        for episode in range(self.episodes):

            snapshot1 = tracemalloc.take_snapshot()
            stats1 = snapshot1.statistics('lineno')
            # for stat in stats1[:5]:  # Show top 5 memory-consuming lines
            #     print(stat)

            episode_reward = 0
            state, info = self.env.reset(seed=42)
            state = np.transpose(state, (2, 0, 1))
            for step in range(self.max_steps):
                action = self.epsilon_greedy_policy(state, epsilon)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                new_state = np.transpose(new_state, (2, 0, 1))
                self.memory.append((state, action, reward, new_state, terminated or truncated))
                if terminated or truncated:
                    break


                if len(self.memory) > self.batch_size:
                    train_set = random.sample(self.memory, self.batch_size)
                    self.update_q(train_set)

                state = new_state

                episode_reward += reward

            epsilon = max(self.epsilon_min, self.epsilon_decay * epsilon)

            print("Reward:", episode_reward)
            snapshot2 = tracemalloc.take_snapshot()
            stats2 = snapshot2.statistics('lineno')
            # for stat in stats2[:5]:  # Show top 5 memory-consuming lines
            #     print(stat)

            
            if episode % 100:
                torch.save(self.dqn.state_dict(), "weights.pth")


    def debug(self):
        state, _ = self.env.reset(seed=42)
        state = np.transpose(state, (2, 0, 1))
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.dqn.forward(state)

    def load(self):
        if os.path.exists("weights.pth"):
            self.dqn.load_state_dict(torch.load("weights.pth", weights_only=False))
            print(self.dqn)
            self.dqn.eval()

    def play_single_episode(self):
        epsilon = self.epsilon
        state, info = self.env.reset(seed=42)
        state = np.transpose(state, (2, 0, 1))
        for step in range(max_steps):
            action = self.epsilon_greedy_policy(state, epsilon)
            new_state, reward, terminated, truncated, info = self.env.step(action)
        
            if terminated or truncated:
                break

        state = new_state

        



agent = DeepQLearning("ALE/KungFuMaster-v5", 10000, 5000)
agent.load()
agent.train()
# agent.play_single_episode()
