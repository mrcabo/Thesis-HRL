import random
import math
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from thesis_hrl.utils import ReplayMemory, Transition
from household_env.envs.house_env import Tasks


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 50)
        self.fc2 = nn.Linear(50, 25)
        self.head = nn.Linear(25, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x


class QLearning:
    def __init__(self, obs_space, action_space):
        # Hyper-parameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # TODO: necesario aqui?
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        # Others
        self.steps_done = 0
        self.obs_space = obs_space
        self.action_space = action_space

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax().view(1)
        else:
            return torch.tensor([random.randrange(self.action_space)], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # TODO pending..


env = gym.make('household_env:Household-v0')

tasks_list = [Tasks.TURN_ON_TV]
env.set_current_task(tasks_list[0])
env.reset()

if __name__ == "__main__":
    q_learning = QLearning(env.observation_space.shape[0], env.action_space.n)
    num_episodes = 200
    for i_episode in range(num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float, device=q_learning.device)
        for i in count():
            # env.render()
            # Select action and execute it
            action = q_learning.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            # Convert to tensors - TODO: I gotta see if we dont run out of GPU memory if buffer gets too big..
            next_state = torch.tensor(next_state, dtype=torch.float, device=q_learning.device)
            reward = torch.tensor([reward], dtype=torch.float, device=q_learning.device)
            done = torch.tensor([done], dtype=torch.bool, device=q_learning.device)

            q_learning.memory.push(state, action, next_state, reward, done)
            state = next_state

            # Perform one step of the optimization (on the target network)
            q_learning.optimize_model()
            print(f"Action taken: {action}")
            print("hello")
            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break
            # time.sleep(0.1)
