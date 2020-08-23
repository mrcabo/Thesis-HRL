from pathlib import Path

import gym
import yaml
from household_env.envs.house_env import Tasks

from thesis_hrl.config import CONF_DIR
from thesis_hrl.model import HRLDQN
from thesis_hrl.utils import parse_arguments
from thesis_hrl.training import train

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

    # NOTE: pre-trained weights should be copied to results_path

    env = gym.make('household_env:Household-v0')
    tasks_list = [Tasks.MAKE_PASTA]
    my_model = HRLDQN(env.observation_space.shape[0], env.action_space.n, **hyperparam)
    my_model.print_model()

    # Load pre-trained policies in other tasks
    if not args.weights:
        raise Exception("The necessary argument --weights must indicate the name of a real directory that contains "
                        "the pre-trained weights.")
    my_model.load_model(results_path.parent / args.weights)
    my_model.prev_trained = [Tasks.MAKE_TEA, Tasks.MAKE_SOUP]
    # Train model for the new task
    train(env, my_model, tasks_list, results_path, **hyperparam)

    env.close()
