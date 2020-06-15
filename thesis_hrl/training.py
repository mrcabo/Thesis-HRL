import random
from itertools import count
from pathlib import Path

import gym
import numpy as np
import torch
import yaml
from household_env.envs.house_env import Tasks

from thesis_hrl.model import HRLDQN, plot_info
from thesis_hrl.utils import parse_arguments, normalize_values
from thesis_hrl.config import CONF_DIR


def train(env, model, task_list, results_path, **kwargs):
    filename_ep_reward = results_path / ('Episode rewards' + model.get_param_suffix() + '.png')
    filename_cum_reward = results_path / ('Cumulative rewards' + model.get_param_suffix() + '.png')
    filename_avg_reward = results_path / ('Average rewards' + model.get_param_suffix() + '.png')
    cycle_rewards = []
    W = kwargs.get('warmup_period')
    U = kwargs.get('joint_period')

    for i_cycle in range(kwargs.get('num_cycles')):
        state = env.reset()
        state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
        cycle_reward = 0
        debug_master_actions = []
        # Sample task from the task distribution (possibility)
        env.set_current_task(random.choice(task_list))  # TODO: delete this if we change between tasks during learning
        # Warmup period
        for w in range(W):
            action = model.master_policy.select_action(state)
            debug_master_actions.append(action.item())
            next_state, reward, done, _ = env.step(action.item())
            cycle_reward += reward
            next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
            reward = torch.tensor([reward], dtype=torch.float, device=model.device)
            done = torch.tensor([done], dtype=torch.bool, device=model.device)
            model.master_ER.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, done)
            state = next_state

            model.optimize_master()
            if done:
                cycle_rewards.append(cycle_reward)
                break
            # Update the target network, copying all weights and biases in DQN
            if model.master_policy.steps_done % model.TARGET_UPDATE == 0:
                model.target_net.load_state_dict(model.policy_net.state_dict())

        # Joint update period
        for u in range(U):
            pass

        if i_cycle % 100 == 0:
            print(f"Episode {i_cycle}")
            plot_info(np.array(cycle_rewards), filename_ep_reward, 'Cycle rewards', ('Cycle', 'Reward'), fig_num=1)
            # Cumulative reward
            cum_reward = [cycle_rewards[0]]
            for val in cycle_rewards[1:]:
                cum_reward.append(val + cum_reward[-1])
            plot_info(cum_reward, filename_cum_reward, 'Cumulative reward', ('Episode', 'Reward'), fig_num=2)
            model.save_model(results_path)


def test():
    pass


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
    env.set_current_task(tasks_list[0])  # TODO: delete this if we change between tasks during learning

    my_model = HRLDQN(env.observation_space.shape[0], env.action_space.n, **hyperparam)
    my_model.print_model()

    if args.test:
        test()
    else:
        train(env, my_model, tasks_list, results_path, **hyperparam)

    env.close()
