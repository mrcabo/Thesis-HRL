import random
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from thesis_hrl.utils import ReplayMemory, Transition


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
        self.EPS_DECAY = 2000
        self.TARGET_UPDATE = 10
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_net = QNetwork(obs_space, action_space).to(self.device)
        print(f"Policy net:\n {self.policy_net}")
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
        if self.steps_done % 500 == 0:
            print(f"Epsilon threshold: {eps_threshold}")
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax().view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: not s,
                                                batch.done)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for (s, d) in zip(batch.next_state, batch.done) if not d])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # TODO: necessary?
        self.optimizer.step()

    def save_models(self, path):
        # Saves weights of the network to a file
        policy_path = path / 'policy_net.pt'
        target_path = path / 'target_net.pt'
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.target_net.state_dict(), target_path)

    def load_models(self, path):
        # Loads weights of the network from a file
        policy_path = path / 'policy_net.pt'
        target_path = path / 'target_net.pt'
        self.policy_net.load_state_dict(torch.load(policy_path))
        self.target_net.load_state_dict(torch.load(target_path))


def plot_info(data, title=None, labels=None, fig_num=None):
    plt.figure(fig_num)
    plt.clf()
    if title is not None:
        plt.title(title)
    if labels is not None:
        xlabel, ylabel = labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.plot(data)
    plt.savefig(title)
