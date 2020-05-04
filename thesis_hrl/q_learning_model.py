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
        self.fc = nn.Sequential(
            nn.Linear(state_size, 1500),
            nn.ReLU(),
            nn.Linear(1500, 500),
            nn.ReLU(),
            nn.Linear(500, action_size)
        )

    def forward(self, x):
        qval = self.fc(x)
        return qval


class QLearning:
    def __init__(self, obs_space, action_space, batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05,
                 eps_decay=500., target_update=10, lr=1e-2):
        # Hyper-parameters
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = int(target_update)
        self.LEARNING_RATE = lr
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # TODO: necesario aqui?
        # self.optimizer = optim.Adam(self.policy_net.parameters())
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.memory = ReplayMemory(10000)
        # Others
        self.steps_done = 0
        self.obs_space = obs_space
        self.action_space = action_space

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        if self.steps_done % 500 == 0 and eps_threshold > 0.6:
            print(f"Epsilon threshold: {eps_threshold}")
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(0)[1].view(1, 1)
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
        # TODO:This line below can return an empty tensor and crashes (happened for batch_size=4)
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
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)  # TODO: necessary?
        self.optimizer.step()

    def print_hyperparam(self):
        print(str("#" * 5 + "  HYPER-PARAMETERS  " + "#" * 5))
        print(f"Batch size: {self.BATCH_SIZE}")
        print(f"Gamma: {self.GAMMA}")
        print(f"Epsilon start: {self.EPS_START}")
        print(f"Epsilon end: {self.EPS_END}")
        print(f"Epsilon decay: {self.EPS_DECAY}")
        print(f"Target net update freq.: {self.TARGET_UPDATE}")
        print(f"Learning rate.: {self.LEARNING_RATE}")
        print(f"Network:\n {self.policy_net}")
        print("#" * 30)

    def set_hyperparam(self, **kwargs):
        variables = vars(self)
        for key, value in kwargs.items():
            if key in variables:
                variables[key] = value

    def get_param_suffix(self):
        s = (f"_{self.BATCH_SIZE}" + f"_{self.GAMMA}" + f"_{self.EPS_START}" +
             f"_{self.EPS_END}" + f"_{self.EPS_DECAY}" + f"_{self.TARGET_UPDATE}" +
             f"_{self.LEARNING_RATE}")
        return s

    def save_models(self, path):
        # Saves weights of the network to a file
        policy_path = path / ('policy_net' + self.get_param_suffix() + '.pt')
        target_path = path / ('target_net' + self.get_param_suffix() + '.pt')
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.target_net.state_dict(), target_path)

    def load_models(self, path, w_suffix):
        # Loads weights of the network from a file
        policy_path = path / ('policy_net_' + w_suffix + '.pt')
        target_path = path / ('target_net_' + w_suffix + '.pt')
        self.policy_net.load_state_dict(torch.load(policy_path))
        self.target_net.load_state_dict(torch.load(target_path))


def plot_info(data, path, title=None, labels=None, fig_num=None):
    plt.figure(fig_num)
    plt.clf()
    if title is not None:
        plt.title(title)
    if labels is not None:
        xlabel, ylabel = labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.plot(data)
    plt.savefig(path)
