import random
from itertools import count
from pathlib import Path

import gym
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from household_env.envs.house_env import Tasks

from thesis_hrl.model import HRLDQN, plot_info
from thesis_hrl.utils import parse_arguments, normalize_values
from thesis_hrl.config import CONF_DIR


class PolicyUsage:
    def __init__(self, num_subpolicies):
        # Only record first 100 timesteps
        self.usage_list = []
        self.window_size = 40
        for _ in range(num_subpolicies):
            self.usage_list.append(np.zeros(self.window_size))

    def record(self, action, timestep):
        if timestep < self.window_size:
            self.usage_list[action][timestep] += 1

    def plot_usage(self, path, title=None, labels=None, fig_num=None):
        plt.figure(fig_num)
        if title is not None:
            plt.title(title)
        if labels is not None:
            xlabel, ylabel = labels
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

        norm = self.usage_list[0]
        for i in range(1, len(self.usage_list)):
            norm = norm + self.usage_list[i]
        norm = np.where(norm != 0, norm, 1)  # So to not divide by 0
        for i, data in enumerate(self.usage_list):
            plt.plot(data / norm, label=(f"Sub-policy_{i}"))
        plt.legend()
        plt.savefig(path)


def train_master(model, task_name, train_iters_M):
    for _ in range(train_iters_M):
        model.optimize_master(model.master_ERs[task_name])
        model.master_policy.updates_done += 1
        if model.master_policy.updates_done % model.M_TARGET_UPDATE == 0:
            model.master_policy.target_net.load_state_dict(model.master_policy.policy_net.state_dict())


def test(env, model, task_list, results_path, **kwargs):
    model.testing_mode()
    # env.render()
    n_episodes = kwargs.get("test_episodes", 10)
    successful_episodes = 0
    usage = PolicyUsage(kwargs.get("n_sub_policies"))
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
                master_action = model.master_policy.policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
                usage.record(master_action.item(), t)
                action = model.sub_policies[master_action.item()].policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
                # print(f"Action taken: {action.item()}")
                next_state, reward, done, _ = env.step(action.item())
                ep_reward += reward
                next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
                state = next_state
                # env.render()
                if done:
                    if ep_reward > 90:
                        successful_episodes += 1
                    print(f"Episode reward: {ep_reward}")
                    break

    graph_name = 'Sub-policy usage'
    filename = results_path / (graph_name + '.png')
    usage.plot_usage(filename, graph_name, ('Timestep', 'Usage'), fig_num=0)
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

    env = gym.make('household_env:Household-v0')
    tasks_list = [Tasks.CLEAN_STOVE]
    env.set_current_task(tasks_list[0])  # TODO: delete this if we change between tasks during learning

    my_model = HRLDQN(env.observation_space.shape[0], env.action_space.n, **hyperparam)
    my_model.print_model()
    # Load pre-trained policies in other tasks
    if not args.weights:
        raise Exception("The necessary argument --weights must indicate the name of a real directory that contains "
                        "the pre-trained weights.")
    my_model.load_model(results_path.parent / args.weights)
    my_model.load_task_memories(results_path.parent / args.weights)
    train_master(my_model, tasks_list[0].name, hyperparam.get('train_iters_M'))

    test(env, my_model, tasks_list, results_path, **hyperparam)

    env.close()
