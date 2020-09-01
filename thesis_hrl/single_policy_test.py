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


def test(env, model, task_list, results_path, **kwargs):
    model.load_model(results_path)
    model.testing_mode()
    env.render()
    n_episodes = kwargs.get("test_episodes", 10)
    successful_episodes = 0
    with torch.no_grad():
        for e in range(n_episodes):
            print(f"Episode {e}")
            state = env.reset()
            chosen_task = random.choice(task_list)
            print(f"Chosen task: {chosen_task.name}")  # DEBUG
            state = env.set_current_task(chosen_task)
            state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
            ep_reward = 0
            for t in count():
                action = model.policy_net(state).max(0)[1].view(1, 1)
                # print(f"Action taken: {action.item()}")
                next_state, reward, done, _ = env.step(action.item())
                ep_reward += reward
                next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
                state = next_state
                env.render()
                if done:
                    if ep_reward > 90:
                        successful_episodes += 1
                    print(f"Episode reward: {ep_reward}")
                    break

    print('Testing complete')
    print(f"{successful_episodes}/{n_episodes} successful episodes. "
          f"{((successful_episodes / n_episodes) * 100)}% success rate")


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
    tasks_list = [Tasks.MAKE_SOUP]
    env.set_current_task(tasks_list[0])

    my_model = QLearning(env.observation_space.shape[0], env.action_space.n, **hyperparam)
    my_model.print_hyperparam()
    test(env, my_model, tasks_list, results_path, **hyperparam)

    env.close()
