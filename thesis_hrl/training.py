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
    filename_ep_reward = results_path / 'Episode rewards.png'
    filename_cum_reward = results_path / 'Cumulative rewards.png'
    cycle_rewards = []
    W = kwargs.get('warmup_period')
    U = kwargs.get('joint_period')

    for i_cycle in range(kwargs.get('num_cycles')):
        state = env.reset()
        state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
        cycle_reward = 0
        # Sample task from the task distribution (possibility)
        model.master_policy.reset()  # Reset :math:`\theta`
        chosen_task = random.choice(task_list)
        env.set_current_task(chosen_task)
        # Warmup period
        for w in range(W):
            policy_idx = model.master_policy.select_action(state)  # Chooses a policy
            policy_action = model.sub_policies[policy_idx].policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
            # policy_action = model.sub_policies[policy_idx].select_action(state)  # TODO: maybe this works better??
            next_state, reward, done, _ = env.step(policy_action.item())
            # print(f"Primitive action taken: {policy_action.item()}")
            cycle_reward += reward
            next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
            reward = torch.tensor([reward], dtype=torch.float, device=model.device)
            done = torch.tensor([done], dtype=torch.bool, device=model.device)
            model.master_ER.push(state.unsqueeze(0), policy_idx, next_state.unsqueeze(0), reward, done)
            state = next_state

            model.optimize_master()
            # env.render()
            if done:
                # This ensures that W updates will be performed always.
                if w < W:  # The last one won't be reset so it can continue to gather info in next phase.
                    state = env.reset()
                    state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
                    env.set_current_task(chosen_task)
                continue
            # Update the target network, copying all weights and biases in DQN
            if model.master_policy.steps_done % model.TARGET_UPDATE == 0:
                model.target_net.load_state_dict(model.policy_net.state_dict())

        # Joint update period
        for u in range(U):
            policy_idx = model.master_policy.select_action(state)  # Chooses a policy
            policy_action = model.sub_policies[policy_idx].select_action(state)
            next_state, reward, done, _ = env.step(policy_action.item())
            # print(f"Primitive action taken: {policy_action.item()}")
            cycle_reward += reward
            next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
            reward = torch.tensor([reward], dtype=torch.float, device=model.device)
            done = torch.tensor([done], dtype=torch.bool, device=model.device)
            model.master_ER.push(state.unsqueeze(0), policy_idx, next_state.unsqueeze(0), reward, done)
            model.sub_ER.push(state.unsqueeze(0), policy_action, next_state.unsqueeze(0), reward, done)
            state = next_state

            model.optimize_master()
            model.optimize_sub(policy_idx)
            # TODO:we could as well update all policies? after all they share a ER.
            # for i, _ in enumerate(model.sub_policies):
            #     model.optimize_sub(i)
            # env.render()
            if done:
                state = env.reset()
                state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
                env.set_current_task(chosen_task)
                continue
            # Update the target network, copying all weights and biases in DQN
            if model.master_policy.steps_done % model.TARGET_UPDATE == 0:
                model.target_net.load_state_dict(model.policy_net.state_dict())
            for policy in model.sub_policies:
                if policy.steps_done % model.TARGET_UPDATE == 0:
                    policy.target_net.load_state_dict(policy.policy_net.state_dict())

        # print(f"Current state is: {env.env.states}")
        cycle_rewards.append(cycle_reward)
        if i_cycle % 100 == 0:
            print(f"Episode {i_cycle}")
            plot_info(np.array(cycle_rewards), filename_ep_reward, 'Cycle rewards', ('Cycle', 'Reward'), fig_num=1)
            # Cumulative reward
            cum_reward = [cycle_rewards[0]]
            for val in cycle_rewards[1:]:
                cum_reward.append(val + cum_reward[-1])
            plot_info(cum_reward, filename_cum_reward, 'Cumulative reward', ('Cycle', 'Reward'), fig_num=2)
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
