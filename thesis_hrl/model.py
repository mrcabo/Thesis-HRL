import json
import random
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml

from thesis_hrl.utils import ReplayMemory, Transition
from thesis_hrl.config import CONF_DIR


def reset_weights(m):
    """
    Re-initializes the weights of the MLP.
    Args:
        m: Layer that will be re-initialized. Only nn.Linear layers will be reset.
    """
    if type(m) == nn.Linear:
        m.reset_parameters()


class MLP(nn.Module):
    def __init__(self, state_size, action_size, **kwargs):
        h = kwargs.get('hidden_size', (1500, 500))
        if len(h) != 2 or h[0] <= 0 or h[1] <= 0:
            raise Exception("A tuple of size 2 must be provided with non-negative values")
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, h[0]),
            nn.ReLU(),
            nn.Linear(h[0], h[1]),
            nn.ReLU(),
            nn.Linear(h[1], action_size)
        )

    def forward(self, x):
        qval = self.fc(x)
        return qval


class Policy:
    def __init__(self, obs_space, action_space, device, **kwargs):
        self.policy_net = MLP(obs_space, action_space, **kwargs).to(device)
        self.target_net = MLP(obs_space, action_space, **kwargs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=kwargs.get('lr'))

    def reset(self):
        self.policy_net.apply(reset_weights)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self):
        pass

    def optimize_model(self):
        pass


class HRLDQN:
    def __init__(self, obs_space, action_space, **kwargs):
        # Hyper-parameters
        self.BATCH_SIZE = kwargs.get('batch_size', 32)
        self.GAMMA = kwargs.get('gamma', 0.99)
        self.EPS_START = kwargs.get('eps_start', 1)
        self.EPS_END = kwargs.get('eps_end', 0.1)
        self.EPS_DECAY = kwargs.get('eps_decay', 2e5)
        self.TARGET_UPDATE = int(kwargs.get('target_update', 1e4))
        self.LEARNING_RATE = kwargs.get('lr', 0.00025)
        self.MASTER_ER_LENGTH = int(kwargs.get('master_ER', 1000000))
        self.SUB_ER_LENGTH = int(kwargs.get('sub_ER', 1000000))
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.master_policy = Policy(obs_space, action_space, self.device,
                                    hidden_size=kwargs.get('master_hidden_size', (1500, 500)),
                                    lr=self.LEARNING_RATE)
        self.sub_policies = [Policy(obs_space, action_space, self.device,
                                    hidden_size=kwargs.get('sub_hidden_size', (1500, 500)),
                                    lr=self.LEARNING_RATE) for _ in range(kwargs.get('n_sub_policies', 3))]
        self.loss = nn.MSELoss()
        self.master_ER = ReplayMemory(self.MASTER_ER_LENGTH)
        self.sub_ER = ReplayMemory(self.SUB_ER_LENGTH)
        # Others
        self.steps_done = 0
        self.obs_space = obs_space
        self.action_space = action_space

    def print_hyperparam(self):
        print(str("#" * 5 + "  HYPER-PARAMETERS  " + "#" * 5))
        print(f"Batch size: {self.BATCH_SIZE}")
        print(f"Gamma: {self.GAMMA}")
        print(f"Epsilon start: {self.EPS_START}")
        print(f"Epsilon end: {self.EPS_END}")
        print(f"Epsilon decay: {self.EPS_DECAY}")
        print(f"Target net update freq.: {self.TARGET_UPDATE}")
        print(f"Learning rate.: {self.LEARNING_RATE}")
        print(f"Experience replay length.: {self.ER_LENGTH}")
        print(f"Network:\n {self.policy_net}")
        print("#" * 30)


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


if __name__ == '__main__':
    path = CONF_DIR / 'hyperparam.yaml'
    with open(path) as file:
        hyperparam = yaml.full_load(file)

    foo = HRLDQN(10, 5, **hyperparam)
    print(foo.master_policy.policy_net)
    print(foo.sub_policies[0].policy_net)
