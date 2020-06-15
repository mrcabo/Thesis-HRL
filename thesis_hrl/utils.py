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
    NORM_VEC = [19., 19.] + [1.] * 14
    device = state.device
    norm_vec = torch.tensor(NORM_VEC, device=device)
    if inv:
        return state * norm_vec
    else:
        return state / norm_vec


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--hyperparam', type=str, default='',
                        help='Name of the hyperparamer YAML file to be used. It should be placed in the config '
                             'directory.')
    parser.add_argument('--test', type=str, default='',
                        help='Path to the directory containing all the saved weights.')
    args = parser.parse_args()
    return args
