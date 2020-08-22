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


def plot_and_save(model, cycle_rewards, results_path, filename_ep_reward, filename_cum_reward, filename_avg_reward):
    plot_info(np.array(cycle_rewards), filename_ep_reward, 'Cycle rewards', ('Cycle', 'Reward'), fig_num=1)
    # Cumulative reward
    cum_reward = [cycle_rewards[0]]
    for val in cycle_rewards[1:]:
        cum_reward.append(val + cum_reward[-1])
    plot_info(cum_reward, filename_cum_reward, 'Cumulative reward', ('Cycle', 'Reward'), fig_num=2)
    # Avg reward over last 10 eps
    avg_reward = []
    for i in range(len(cycle_rewards) + 1):
        if i > 9:
            avg_reward.append(np.mean(cycle_rewards[i - 10:i]))
    plot_info(avg_reward, filename_avg_reward, 'Average reward', ('Cycle', 'Reward'), fig_num=3)

    model.save_model(results_path)
    return cum_reward[-1]


def train_master(env, model, task_list, results_path, **kwargs):
    filename_ep_reward = results_path / 'Episode rewards.png'
    filename_cum_reward = results_path / 'Cumulative rewards.png'
    filename_avg_reward = results_path / 'Average rewards.png'
    ep_rewards = []

    # Reset master policy
    model.master_policy.reset()

    for i_episode in range(kwargs.get('retrain_master_episodes')):
        state = env.reset()
        chosen_task = random.choice(task_list)
        print(f"Chosen task: {chosen_task.name}")  # DEBUG
        state = env.set_current_task(chosen_task)
        state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
        ep_reward = 0
        for t in count():
            master_action = model.master_policy.select_action(state)  # Chooses a policy
            primitive_action = model.sub_policies[master_action.item()].policy_net(state).max(0)[1].view(1, 1)
            next_state, reward, done, _ = env.step(primitive_action.item())
            ep_reward += reward
            next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
            reward = torch.tensor([reward], dtype=torch.float, device=model.device)
            done = torch.tensor([done], dtype=torch.bool, device=model.device)

            model.master_ER.push(state.unsqueeze(0), master_action, next_state.unsqueeze(0), reward, done)
            state = next_state

            # Perform one step of the optimization (on the target network)
            model.optimize_master()
            # Update the target network, copying all weights and biases in DQN
            if model.master_policy.steps_done % model.M_TARGET_UPDATE == 0:
                model.master_policy.target_net.load_state_dict(model.master_policy.policy_net.state_dict())

            if done:
                if ep_reward > 90:
                    print(f"Success in ep. {i_episode}!! Ep. reward: {ep_reward}")
                ep_rewards.append(ep_reward)
                # plot_info(ep_rewards, 'Episode rewards', ('N. episode', 'Reward'))
                break

        if i_episode % 100 == 0:
            print(f"Episode {i_episode}")
            plot_and_save(model, ep_rewards, results_path, filename_ep_reward, filename_cum_reward, filename_avg_reward)

    cum_r = plot_and_save(model, ep_rewards, results_path, filename_ep_reward, filename_cum_reward, filename_avg_reward)
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
    tasks_list = [Tasks.MAKE_TEA]
    env.set_current_task(tasks_list[0])  # TODO: delete this if we change between tasks during learning

    my_model = HRLDQN(env.observation_space.shape[0], env.action_space.n, **hyperparam)
    my_model.print_model()

    # Load pre-trained policies in other tasks
    if not args.weights:
        raise Exception("The necessary argument --weights must indicate the name of a real directory that contains "
                        "the pre-trained weights.")
    my_model.load_model(results_path.parent / args.weights)

    train_master(env, my_model, tasks_list, results_path, **hyperparam)

    env.close()
