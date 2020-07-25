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


def train(env, model, task_list, results_path, **kwargs):
    filename_ep_reward = results_path / 'Episode rewards.png'
    filename_cum_reward = results_path / 'Cumulative rewards.png'
    ep_rewards = []
    for i_episode in range(kwargs.get('num_episodes')):
        state = env.reset()
        chosen_task = random.choice(task_list)
        print(f"Chosen task: {chosen_task.name}")  # DEBUG
        state = env.set_current_task(chosen_task)
        state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
        ep_reward = 0
        for t in count():
            # Select action and execute it
            action = model.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            ep_reward += reward
            next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
            reward = torch.tensor([reward], dtype=torch.float, device=model.device)
            done = torch.tensor([done], dtype=torch.bool, device=model.device)

            model.memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, done)
            state = next_state

            # Perform one step of the optimization (on the target network)
            model.optimize_model()
            if done:
                if ep_reward > 90:
                    print(f"Success in ep. {i_episode}!! Ep. reward: {ep_reward}")
                ep_rewards.append(ep_reward)
                # plot_info(ep_rewards, 'Episode rewards', ('N. episode', 'Reward'))
                break
            # Update the target network, copying all weights and biases in DQN
            if model.steps_done % model.TARGET_UPDATE == 0:
                model.target_net.load_state_dict(model.policy_net.state_dict())

        if i_episode % 100 == 0:
            print(f"Episode {i_episode}")
            plot_and_save(model, ep_rewards, results_path, filename_ep_reward, filename_cum_reward)

    cum_r = plot_and_save(model, ep_rewards, results_path, filename_ep_reward, filename_cum_reward)
    print('Training complete')
    print(f"Cumulative reward: {cum_r}")


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
    train(env, my_model, tasks_list, results_path, **hyperparam)

    env.close()
