import random
import time
from collections import deque
from datetime import timedelta
from itertools import count
from pathlib import Path

import gym
import numpy as np
import torch
import yaml
from household_env.envs.house_env import Tasks

from thesis_hrl.config import CONF_DIR
from thesis_hrl.model import HRLDQN, plot_info
from thesis_hrl.utils import parse_arguments, normalize_values, save_list_to_disk


def plot_and_save(model, cycle_rewards, results_path, filename_ep_reward, filename_cum_reward):
    plot_info(np.array(cycle_rewards), filename_ep_reward, 'Cycle rewards', ('Cycle', 'Reward'), fig_num=1)
    # Cumulative reward
    cum_reward = [cycle_rewards[0]]
    for val in cycle_rewards[1:]:
        cum_reward.append(val + cum_reward[-1])
    plot_info(cum_reward, filename_cum_reward, 'Cumulative reward', ('Cycle', 'Reward'), fig_num=2)
    model.save_model(results_path)
    return cum_reward[-1]


def train(env, model, task_list, results_path, **kwargs):
    start_time = time.time()
    total_timesteps = 0
    successful_runs = deque(maxlen=100)
    timer_flag = True  # Indicates if the experiment already achieved a X % success rate. Avoids repeating measurement
    success_threshold = kwargs.get('success_threshold')
    filename_ep_reward = results_path / 'Episode rewards.png'
    filename_cum_reward = results_path / 'Cumulative rewards.png'
    ep_rewards = []
    for i_episode in range(kwargs.get('num_episodes')):
        # Sample a task and initialize environment
        chosen_task = random.choice(task_list)
        print(f"Chosen task: {chosen_task.name}")  # DEBUG
        model.master_policy.reset()
        # Train on ERs
        if len(model.master_ERs[chosen_task.name]) > 0:
            for i in range(kwargs.get('train_iters_M')):
                model.optimize_master(model.master_ERs[chosen_task.name])
                model.master_policy.updates_done += 1
                if model.master_policy.updates_done % model.M_TARGET_UPDATE == 0:
                    model.master_policy.target_net.load_state_dict(model.master_policy.policy_net.state_dict())
            for i in range(kwargs.get('train_iters')):
                idx = model.master_ERs[chosen_task.name].sample(1)[0].action.item()
                model.optimize_sub(model.task_ERs[chosen_task.name], idx)
                model.sub_policies[idx].updates_done += 1
                for policy in model.sub_policies:
                    if policy.updates_done % model.S_TARGET_UPDATE == 0:
                        policy.target_net.load_state_dict(policy.policy_net.state_dict())
        # Get more experiences
        env.reset()
        state = env.set_current_task(chosen_task)
        state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
        ep_reward = 0
        for t in range(kwargs.get('collect_exp')):
            master_action = model.master_policy.select_action(state)  # Chooses a policy
            primitive_action = model.sub_policies[master_action.item()].select_action(state)
            next_state, reward, done, _ = env.step(primitive_action.item())
            total_timesteps += 1
            # print(f"Primitive action taken: {policy_action.item()}")
            ep_reward += reward
            next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
            reward = torch.tensor([reward], dtype=torch.float, device=model.device)
            done = torch.tensor([done], dtype=torch.bool, device=model.device)
            model.master_ERs[chosen_task.name].push(state.unsqueeze(0), master_action,
                                                    next_state.unsqueeze(0), reward, done)
            if not model.master_policy.rand_action:
                # Only adding experience to replay if master actions are following the optimal policy
                model.task_ERs[chosen_task.name].push(state.unsqueeze(0), primitive_action,
                                                      next_state.unsqueeze(0), reward, done)
            state = next_state
            if done:
                successful_runs.append(1 if reward > 0 else 0)
                if timer_flag and (sum(successful_runs) >= success_threshold):
                    time_elapsed = timedelta(seconds=time.time() - start_time)
                    print(f"Time elapsed until {success_threshold} success rate: {time_elapsed}")
                    print(f"Timesteps taken until {success_threshold} success rate: {total_timesteps}")
                    timer_flag = not timer_flag
                state = env.reset()
                state = env.set_current_task(chosen_task)
                state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
                ep_rewards.append(ep_reward)
                ep_reward = 0

        if i_episode % 500 == 0:
            print(f"Episode {i_episode}")
            plot_and_save(model, ep_rewards, results_path, filename_ep_reward, filename_cum_reward)

    cum_r = plot_and_save(model, ep_rewards, results_path, filename_ep_reward, filename_cum_reward)
    save_list_to_disk(ep_rewards, results_path / 'ep_rewards.pickle')
    model.save_task_memories(results_path)
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
    tasks_list = [Tasks.MAKE_PASTA, Tasks.MAKE_TEA, Tasks.MAKE_SOUP, Tasks.MAKE_OMELETTE, Tasks.CLEAN_STOVE,
                  Tasks.MAKE_PANCAKES]

    my_model = HRLDQN(env.observation_space.shape[0], env.action_space.n, **hyperparam)
    my_model.print_model()
    train(env, my_model, tasks_list, results_path, **hyperparam)

    env.close()
