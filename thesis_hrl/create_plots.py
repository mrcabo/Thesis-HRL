from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from thesis_hrl.utils import load_list_from_disk


def plot(mean, std, filename, title='', ylabel='y'):
    t = np.arange(np.size(stacked_arrays, 1))
    fig, ax = plt.subplots(1)
    # ax.plot(t, mean, lw=2, label='mean population 1', color='blue')
    ax.plot(t, mean, color='tab:blue')
    ax.fill_between(t, mean + std, mean - std, facecolor='tab:blue', alpha=0.5)
    ax.set_title(title)
    # ax.legend(loc='lower right')
    ax.set_xlabel('Steps')
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(style='sci', scilimits=(0, 5))
    ax.grid()
    plt.tight_layout()
    plt.savefig(filename)


def trunc(data_arr):
    # Truncate array to match min len list
    min_len = len(data_arr[0])
    for i in range(1, 3):
        if len(data_arr[i]) < min_len:
            min_len = len(data_arr[i])
    # min_len = 10000 plot only 10000 episodes..
    for i in range(3):
        data_arr[i] = data_arr[i][:min_len]
    return data_arr


experiment = 'hyperparam_sqn_23'
path = Path('/home/diego/Coding/Thesis-HRL/results/temp')

# Loading data
ep_rewards = []
cum_rewards = []
for i in range(3):
    pathname = path / ('_'.join([experiment, str(i)])) / 'ep_rewards.pickle'
    rewards = np.array(load_list_from_disk(pathname))
    ep_rewards.append(rewards)
    cum_rewards.append(np.cumsum(rewards))

if 'sqn' not in experiment:
    ep_rewards = trunc(ep_rewards)
    cum_rewards = trunc(cum_rewards)

# Calculate episode reward mean-std
stacked_arrays = np.dstack((ep_rewards[0], ep_rewards[1], ep_rewards[2]))
mean = stacked_arrays.mean(axis=2).reshape(-1)
std = stacked_arrays.std(axis=2).reshape(-1)
filename = path / 'Episode_rewards.png'
title = r'Episode reward $\mu\pm\sigma$'
plot(mean, std, filename, title, ylabel='Episode reward')

# Calculate cum. reward mean-std
stacked_arrays = np.dstack((cum_rewards[0], cum_rewards[1], cum_rewards[2]))
mean = stacked_arrays.mean(axis=2).reshape(-1)
std = stacked_arrays.std(axis=2).reshape(-1)
filename = path / 'Cumulative_rewards.png'
title = r'Cumulative reward $\mu\pm\sigma$'
plot(mean, std, filename, title, ylabel='Cumulative reward')
