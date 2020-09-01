import json
import random
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from household_env.envs.house_env import Tasks

from thesis_hrl.utils import ReplayMemory, Transition


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
        # Net
        self.policy_net = MLP(obs_space, action_space, **kwargs).to(device)
        self.target_net = MLP(obs_space, action_space, **kwargs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=kwargs.get('lr'))
        # Others
        self.device = device
        self.steps_done = 0
        self.updates_done = 0
        self.action_space = action_space
        self.EPS_START = kwargs.get('eps_start', 1)
        self.EPS_END = kwargs.get('eps_end', 0.1)
        self.EPS_DECAY = kwargs.get('eps_decay', 2e5)

    def reset(self):
        self.policy_net.apply(reset_weights)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1. * self.steps_done / self.EPS_DECAY)
        # if self.steps_done % 500 == 0 and eps_threshold > 0.2:
        #     print(f"Epsilon threshold: {eps_threshold}")
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space)]], device=self.device, dtype=torch.long)

    def optimize_model(self, memory, batch_size, gamma, loss_func):
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
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
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss = loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)  # TODO: necessary?
        self.optimizer.step()

    def print_hyperparam(self):
        print(f"Epsilon start: {self.EPS_START}")
        print(f"Epsilon end: {self.EPS_END}")
        print(f"Epsilon decay: {self.EPS_DECAY}")
        print(f"Network:\n {self.policy_net}")


class HRLDQN:
    def __init__(self, obs_space, action_space, **kwargs):
        # Hyper-parameters
        self.BATCH_SIZE = kwargs.get('batch_size', 32)
        self.M_GAMMA = kwargs.get('master_gamma', 0.99)
        self.S_GAMMA = kwargs.get('sub_gamma', 0.99)
        self.M_TARGET_UPDATE = int(kwargs.get('master_target_update', 1e4))
        self.S_TARGET_UPDATE = int(kwargs.get('sub_target_update', 1e4))
        self.M_LEARNING_RATE = kwargs.get('master_lr', 0.00025)
        self.S_LEARNING_RATE = kwargs.get('sub_lr', 0.00025)
        # Model
        self.master_ERs = {}  # Creates a dictionary containing an ER for each task
        self.task_ERs = {}  # Creates a dictionary containing an ER for each task
        for t in Tasks:
            self.master_ERs[t.name] = ReplayMemory(int(kwargs.get('master_ER', 1e3)))
            self.task_ERs[t.name] = ReplayMemory(int(kwargs.get('task_ER', 1e5)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_sub_policies = kwargs.get('n_sub_policies', 3)
        self.master_policy = Policy(obs_space, n_sub_policies, self.device,
                                    lr=self.M_LEARNING_RATE,
                                    hidden_size=kwargs.get('master_hidden_size', (1500, 500)),
                                    eps_start=kwargs.get('master_eps_start'),
                                    eps_end=kwargs.get('master_eps_end'),
                                    eps_decay=kwargs.get('master_eps_decay'),
                                    **kwargs)
        self.sub_policies = [Policy(obs_space, action_space, self.device,
                                    lr=self.S_LEARNING_RATE,
                                    hidden_size=kwargs.get('sub_hidden_size', (1500, 500)),
                                    eps_start=kwargs.get('sub_eps_start'),
                                    eps_end=kwargs.get('sub_eps_end'),
                                    eps_decay=kwargs.get('sub_eps_decay'),
                                    **kwargs) for _ in range(n_sub_policies)]
        self.loss = nn.MSELoss()
        # Others
        self.obs_space = obs_space
        self.action_space = action_space
        self.prev_trained = []  # List of previously trained tasks

    def print_model(self):
        print("#" * 5 + "  MODEL DESCRIPTION  " + "#" * 5)
        # General ones
        print("*" * 3 + "  Shared values  " + "*" * 3)
        print(f"Batch size: {self.BATCH_SIZE}")
        print(f"Task ER length.: {self.task_ERs[list(self.task_ERs.keys())[0]].capacity}")

        # Master policy
        print("*" * 3 + "  Master policy  " + "*" * 3)
        print(f"ER length.: {self.master_ERs[list(self.master_ERs.keys())[0]].capacity}")
        print(f"Learning rate.: {self.M_LEARNING_RATE}")
        print(f"Gamma: {self.M_GAMMA}")
        print(f"Target net update freq.: {self.M_TARGET_UPDATE}")
        self.master_policy.print_hyperparam()

        # Sub-policies
        print("*" * 3 + "  Sub policies  " + "*" * 3)
        print(f"Learning rate.: {self.S_LEARNING_RATE}")
        print(f"Gamma: {self.S_GAMMA}")
        print(f"Target net update freq.: {self.S_TARGET_UPDATE}")
        # for i in range(len(self.sub_policies)): # For now i think all sub policies equal so not necessary..
        self.sub_policies[0].print_hyperparam()

        print("#" * 31)

    def optimize_master(self, memory):
        self.master_policy.optimize_model(memory, self.BATCH_SIZE, self.M_GAMMA, self.loss)

    def optimize_sub(self, memory, idx):
        if idx < 0 or idx > len(self.sub_policies):
            raise IndexError(f"Index must be between 0 and {len(self.sub_policies)}")
        else:
            self.sub_policies[idx].optimize_model(memory, self.BATCH_SIZE, self.S_GAMMA, self.loss)

    def optimize_all_subs(self, memory):
        for sub_policy in self.sub_policies:
            sub_policy.optimize_model(memory, self.BATCH_SIZE, self.S_GAMMA, self.loss)

    def testing_mode(self):
        self.master_policy.policy_net.eval()
        self.master_policy.target_net.eval()
        for policy in self.sub_policies:
            policy.policy_net.eval()
            policy.target_net.eval()

    def save_model(self, results_path):
        """
        Saves weights of every MLP.

        Args:
            results_path (Path): Path to the results directory

        Returns:

        """
        # Saves master policy
        policy_path = results_path / 'master_policy_net.pt'
        target_path = results_path / 'master_target_net.pt'
        torch.save(self.master_policy.policy_net.state_dict(), policy_path)
        torch.save(self.master_policy.target_net.state_dict(), target_path)
        # Saves sub_policies
        for i, sub_policy in enumerate(self.sub_policies):
            policy_path = results_path / ('sub_policy_net' + str(i) + '.pt')
            target_path = results_path / ('sub_target_net' + str(i) + '.pt')
            torch.save(sub_policy.policy_net.state_dict(), policy_path)
            torch.save(sub_policy.target_net.state_dict(), target_path)

    def load_model(self, path):
        """
        Loads the weights from the specified path. Only works if weights are all stored in the same directory and
        with the same naming style as in the save_model function.

        Args:
            path (Path): Path to the directory containing all the weights.

        Returns:

        """
        # Loads master policy
        policy_path = path / 'master_policy_net.pt'
        target_path = path / 'master_target_net.pt'
        self.master_policy.policy_net.load_state_dict(torch.load(policy_path))
        self.master_policy.target_net.load_state_dict(torch.load(target_path))
        # Saves sub_policies
        for i, sub_policy in enumerate(self.sub_policies):
            policy_path = path / ('sub_policy_net' + str(i) + '.pt')
            target_path = path / ('sub_target_net' + str(i) + '.pt')
            sub_policy.policy_net.load_state_dict(torch.load(policy_path))
            sub_policy.target_net.load_state_dict(torch.load(target_path))

    def save_task_memories(self, path):
        for key, memoy in self.task_ERs.items():
            pathfile = path / (key + '.pickle')
            torch.save(memoy, pathfile)
        for key, memoy in self.master_ERs.items():
            pathfile = path / ('M_' + key + '.pickle')
            torch.save(memoy, pathfile)

    def load_task_memories(self, path):
        for key in self.task_ERs.keys():
            pathfile = path / (key + '.pickle')
            self.task_ERs[key] = torch.load(pathfile)


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
