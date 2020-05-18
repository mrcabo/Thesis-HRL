import time
from itertools import count
from pathlib import Path

import numpy as np
import gym
import torch

from thesis_hrl.utils import normalize_values, parse_arguments
from thesis_hrl.q_learning_model import plot_info, QLearning
from household_env.envs.house_env import Tasks


def train(num_episodes, env, model, path_to_output):
    filename_ep_reward = path_to_output / ('Episode rewards' + model.get_param_suffix() + '.png')
    filename_cum_reward = path_to_output / ('Cumulative rewards' + model.get_param_suffix() + '.png')
    filename_avg_reward = path_to_output / ('Average rewards' + model.get_param_suffix() + '.png')
    ep_rewards = []
    avg_rewards = []
    for i_episode in range(num_episodes):
        state = env.reset()
        state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
        ep_reward = 0
        debug_actions = []
        for t in count():
            # if i_episode % 100 == 0:
            #     env.render()
            # Select action and execute it
            action = model.select_action(state)
            debug_actions.append(action.item())
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
                if ep_reward > 0:
                    print(f"Success in ep. {i_episode}!! Ep. reward: {ep_reward}")
                    print(f"Number of steps: {t}, actions taken: {debug_actions}")
                ep_rewards.append(ep_reward)
                avg_rewards.append(np.mean(ep_rewards[-100:]))
                # plot_info(ep_rewards, 'Episode rewards', ('N. episode', 'Reward'))
                break
            # Update the target network, copying all weights and biases in DQN
            if model.steps_done % model.TARGET_UPDATE == 0:
                model.target_net.load_state_dict(model.policy_net.state_dict())

        if i_episode % 100 == 0:
            print(f"Episode {i_episode}")
            plot_info(np.array(ep_rewards), filename_ep_reward, 'Episode rewards', ('Episode', 'Reward'), fig_num=1)
            # Cumulative reward
            cum_reward = [ep_rewards[0]]
            for val in ep_rewards[1:]:
                cum_reward.append(val + cum_reward[-1])
            plot_info(cum_reward, filename_cum_reward, 'Cumulative reward', ('Episode', 'Reward'), fig_num=2)
            # Avg. reward over the last 100 episodes
            plot_info(avg_rewards, filename_avg_reward, 'Average rewards', ('Episode', 'Reward'), fig_num=3)
            model.save_models(path_to_output)

    plot_info(np.array(ep_rewards), filename_ep_reward, 'Episode rewards', ('Episode', 'Reward'), fig_num=1)
    cum_reward = [ep_rewards[0]]
    for val in ep_rewards[1:]:
        cum_reward.append(val + cum_reward[-1])
    plot_info(cum_reward, filename_cum_reward, 'Cumulative reward', ('Episode', 'Reward'), fig_num=2)

    # Save model
    model.save_models(path_to_output)
    print('Training complete')


def test(num_episodes, env, model, path_to_output, weights_suffix):
    model.load_models(path_to_output, weights_suffix)
    model.policy_net.eval()
    model.target_net.eval()
    env.render()
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
            ep_reward = 0
            for t in count():
                action = model.policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
                # print(f"Action taken {action}")
                next_state, reward, done, _ = env.step(action.item())
                ep_reward += reward
                next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
                state = next_state
                env.render()
                if done:
                    print(f"Episode reward: {ep_reward}")
                    break
                # time.sleep(0.1)
    print('Testing complete')


if __name__ == '__main__':
    num_episodes, test_path, batch_size, gamma, eps_decay, eps_start, eps_end, target_update, lr, memory = parse_arguments()

    path_to_output = Path.cwd() / 'results'
    # Make sure output exists
    if not path_to_output.exists():
        Path.mkdir(path_to_output, parents=True)

    env = gym.make('household_env:Household-v0')
    tasks_list = [Tasks.MAKE_BED]
    env.set_current_task(tasks_list[0])

    q_learning = QLearning(env.observation_space.shape[0], env.action_space.n,
                           batch_size=batch_size, gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                           target_update=target_update, lr=lr, memory=memory)
    q_learning.print_hyperparam()

    if test_path:
        test(num_episodes, env, q_learning, path_to_output, test_path)
    else:
        train(num_episodes, env, q_learning, path_to_output)

    env.close()
