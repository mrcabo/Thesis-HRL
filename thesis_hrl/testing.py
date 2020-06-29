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


class PolicyUsage:
    def __init__(self, num_subpolicies):
        # Only record first 100 timesteps
        self.usage_list = []
        for _ in range(num_subpolicies):
            self.usage_list.append(np.zeros(100))

    def record(self, action, timestep):
        if timestep < 100:
            self.usage_list[action][timestep] += 1


def test(env, model, task_list, results_path, **kwargs):
    model.load_model(results_path)
    model.testing_mode()
    env.render()
    n_episodes = kwargs.get("test_episodes", 10)
    successful_episodes = 0
    usage = PolicyUsage(kwargs.get("n_sub_policies"))
    with torch.no_grad():
        for _ in range(n_episodes):
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

    for i in range(kwargs.get("n_sub_policies")):
        graph_name = ('Sub-policy_' + str(i) + ' usage')
        filename = results_path / (graph_name + '.png')
        plot_info(usage.usage_list[i] / n_episodes, filename, graph_name, ('Timestep', 'Usage'), fig_num=i)
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
    tasks_list = [Tasks.MAKE_SOUP, Tasks.MAKE_TEA]
    env.set_current_task(tasks_list[0])  # TODO: delete this if we change between tasks during learning

    my_model = HRLDQN(env.observation_space.shape[0], env.action_space.n, **hyperparam)
    my_model.print_model()

    test(env, my_model, tasks_list, results_path, **hyperparam)

    env.close()
