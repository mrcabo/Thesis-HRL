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
    ax.grid()
    plt.savefig(filename)


experiment = 'hyperparam_sqn_23'
path = Path('/home/diego/Coding/Thesis-HRL/results/temp')

# Loading data
ep_rewards = []
cum_rewards = []
for i in range(3):
    pathname = path / ('_'.join([experiment, str(i)])) / 'ep_rewards.pickle'
    rewards = np.array(load_list_from_disk(pathname))
    # cum_reward = [rewards[0]]
    # for val in rewards[1:]:
    #     cum_reward.append(val + cum_reward[-1])
    ep_rewards.append(rewards)
    cum_rewards.append(np.cumsum(rewards))

# Calculate episode reward mean-std
stacked_arrays = np.dstack((ep_rewards[0], ep_rewards[1], ep_rewards[2]))
mean = stacked_arrays.mean(axis=2).reshape(-1)
std = stacked_arrays.std(axis=2).reshape(-1)
filename = path / 'Episode_rewards.png'
title = r'Episode reward empirical $\mu$ and $\pm \sigma$'
plot(mean, std, filename, title, ylabel='Episode reward')

# Calculate cum. reward mean-std
stacked_arrays = np.dstack((cum_rewards[0], cum_rewards[1], cum_rewards[2]))
mean = stacked_arrays.mean(axis=2).reshape(-1)
std = stacked_arrays.std(axis=2).reshape(-1)
filename = path / 'Cumulative_rewards.png'
title = r'Cumulative reward empirical $\mu$ and $\pm \sigma$'
plot(mean, std, filename, title, ylabel='Cumulative reward')
