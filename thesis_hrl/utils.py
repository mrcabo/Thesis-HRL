import argparse
from collections import namedtuple

import numpy as np
import random
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def normalize_values(state, inv=False):
    NORM_VEC = [19., 19., 1., 1., 1., 1., 1., 8., 8., 8., 8.]
    device = state.device
    norm_vec = torch.tensor(NORM_VEC, device=device)
    if inv:
        return state * norm_vec
    else:
        return state / norm_vec


def parse_arguments():
    parser = argparse.ArgumentParser(description='Grasping detection system')
    parser.add_argument('--num_episodes', type=int, default=500,
                        help='Number of episodes that will be run')
    parser.add_argument('--test', type=str, default=None,
                        help='Suffix of the weights that will be loaded during testing (default: train)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='Gamma or discount factor')
    parser.add_argument('--eps_decay', type=float, default=500,
                        help='Epsilon decay')
    parser.add_argument('--eps_end', type=float, default=0.05,
                        help='Epsilon final value')
    parser.add_argument('--target_update', type=float, default=100,
                        help='Target net update freq.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    args = parser.parse_args()
    return (args.num_episodes, args.test, args.batch_size, args.gamma,
            args.eps_decay, args.eps_end, args.target_update, args.lr)
