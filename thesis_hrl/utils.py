import argparse
import pickle
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

    def reset(self):
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
    NORM_VEC = [19., 19.] + [1.] * 27
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
    parser.add_argument('--weights', type=str, default='',
                        help='Experiment folder name e.g. "hyperparam_01".')
    args = parser.parse_args()
    return args


def save_list_to_disk(my_list, pathname):
    with open(pathname, 'wb') as filehandle:
        pickle.dump(my_list, filehandle)


def load_list_from_disk(pathname):
    with open(pathname, 'rb') as filehandle:
        my_list = pickle.load(filehandle)
    return my_list
