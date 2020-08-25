import random
from pathlib import Path

import gym
import torch
import yaml
from household_env.envs.house_env import Tasks

from thesis_hrl.config import CONF_DIR
from thesis_hrl.model import HRLDQN
from thesis_hrl.utils import parse_arguments, normalize_values
from thesis_hrl.training import train, plot_and_save


def train_new_task(env, model, task_list, results_path, cycle_rewards, **kwargs):
    filename_ep_reward = results_path / 'Episode rewards.png'
    filename_cum_reward = results_path / 'Cumulative rewards.png'
    W = kwargs.get('warmup_period')
    J = kwargs.get('joint_period')

    for i_cycle in range(kwargs.get('remind_freq')):
        # Sample a task and initialize environment
        state = env.reset()
        chosen_task = random.choice(task_list)
        print(f"Chosen task: {chosen_task.name}")  # DEBUG
        state = env.set_current_task(chosen_task)
        state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
        cycle_reward = 0
        # Reset master policy
        model.master_policy.reset()
        model.master_ER.reset()

        # Warmup period
        for w in range(W):
            master_action = model.master_policy.select_action(state)  # Chooses a policy
            primitive_action = model.sub_policies[master_action.item()].policy_net(state).max(0)[1].view(1, 1)
            next_state, reward, done, _ = env.step(primitive_action.item())
            # print(f"Primitive action taken: {policy_action.item()}")
            cycle_reward += reward
            next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
            reward = torch.tensor([reward], dtype=torch.float, device=model.device)
            done = torch.tensor([done], dtype=torch.bool, device=model.device)
            model.master_ER.push(state.unsqueeze(0), master_action, next_state.unsqueeze(0), reward, done)
            state = next_state

            model.optimize_master()
            # env.render()
            if done:
                # This ensures that W updates will be performed always.
                state = env.reset()
                state = env.set_current_task(chosen_task)
                state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
            # Update the target network, copying all weights and biases in DQN
            if model.master_policy.steps_done % model.M_TARGET_UPDATE == 0:
                model.master_policy.target_net.load_state_dict(model.master_policy.policy_net.state_dict())

        # Joint update period
        for j in range(J):
            master_action = model.master_policy.select_action(state)  # Chooses a policy
            primitive_action = model.sub_policies[master_action.item()].select_action(state)
            next_state, reward, done, _ = env.step(primitive_action.item())
            # print(f"Primitive action taken: {policy_action.item()}")
            cycle_reward += reward
            next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
            reward = torch.tensor([reward], dtype=torch.float, device=model.device)
            done = torch.tensor([done], dtype=torch.bool, device=model.device)
            model.master_ER.push(state.unsqueeze(0), master_action, next_state.unsqueeze(0), reward, done)
            model.task_ERs[chosen_task.name].push(state.unsqueeze(0), primitive_action, next_state.unsqueeze(0), reward,
                                                  done)

            state = next_state

            model.optimize_master()
            model.optimize_sub(model.task_ERs[chosen_task.name], master_action.item())
            if done:
                state = env.reset()
                state = env.set_current_task(chosen_task)
                state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
            # Update the target network, copying all weights and biases in DQN
            if model.master_policy.steps_done % model.M_TARGET_UPDATE == 0:
                model.master_policy.target_net.load_state_dict(model.master_policy.policy_net.state_dict())
            for policy in model.sub_policies:
                if policy.steps_done % model.S_TARGET_UPDATE == 0:
                    policy.target_net.load_state_dict(policy.policy_net.state_dict())

        cycle_rewards.append(cycle_reward)
        if i_cycle % 10 == 0:
            print(f"Cycle {i_cycle}")
            plot_and_save(model, cycle_rewards, results_path, filename_ep_reward, filename_cum_reward)


def remind_old_tasks(model):
    for task in model.prev_trained:
        model.optimize_all_subs(model.task_ERs[task.name])


def train(env, model, task_list, results_path, **kwargs):
    filename_ep_reward = results_path / 'Episode rewards.png'
    filename_cum_reward = results_path / 'Cumulative rewards.png'
    cycle_rewards = []
    for i_cycle in range(kwargs.get('num_cycles')):
        train_new_task(env, model, task_list, results_path, cycle_rewards, **kwargs)
        remind_old_tasks(model)
    cum_r = plot_and_save(model, cycle_rewards, results_path, filename_ep_reward, filename_cum_reward)
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
    my_model.load_task_memories(results_path.parent / args.weights)
    my_model.prev_trained = [Tasks.MAKE_TEA, Tasks.MAKE_SOUP]
    # Train model for the new task
    train(env, my_model, tasks_list, results_path, **hyperparam)

    env.close()
