import random
from itertools import count
from pathlib import Path

import gym
import numpy as np
import torch
import yaml
from household_env.envs.house_env import Tasks

from thesis_hrl.single_policy_model import QLearning, plot_info
from thesis_hrl.utils import parse_arguments, normalize_values
from thesis_hrl.config import CONF_DIR
from thesis_hrl.training import plot_and_save
from thesis_hrl.single_policy_train import train

if __name__ == '__main__':
    args = parse_arguments()  # vars(args)  # Turns it into a dictionary.

    hyperparam_pathname = CONF_DIR / args.hyperparam
    with open(hyperparam_pathname) as file:
        hyperparam = yaml.full_load(file)

    # Make sure output exists
    results_path = (hyperparam_pathname.parents[2] / 'results')  # For now..
    results_path = results_path / hyperparam_pathname.stem
    if not results_path.exists():
        Path.mkdir(results_path, parents=True)

    env = gym.make('household_env:Household-v0')
    new_tasks = [Tasks.MAKE_PANCAKES]
    tasks_list = [Tasks.MAKE_PASTA, Tasks.MAKE_TEA, Tasks.MAKE_SOUP, Tasks.CLEAN_STOVE] + new_tasks
    env.set_current_task(tasks_list[0])

    my_model = QLearning(env.observation_space.shape[0], env.action_space.n, **hyperparam)
    my_model.print_hyperparam()
    # Load pre-trained policies in other tasks
    if not args.weights:
        raise Exception("The necessary argument --weights must indicate the name "
                        "of a real directory that contains the pre-trained weights.")
    my_model.load_model(results_path.parent / args.weights)
    train(env, my_model, tasks_list, results_path, new_tasks, **hyperparam)

    env.close()
