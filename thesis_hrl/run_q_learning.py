from itertools import count
from pathlib import Path

import numpy as np
import gym
import torch

from thesis_hrl.utils import normalize_values, parse_arguments
from thesis_hrl.q_learning_model import plot_info, QLearning
from household_env.envs.house_env import Tasks


def train(num_episodes, env, model, path_to_weights):
    ep_rewards = []
    for i_episode in range(num_episodes):
        print(f"Episode {i_episode}")
        state = env.reset()
        # state = normalize_values(torch.tensor(state, dtype=torch.float, device=model.device))
        state = torch.tensor(state, dtype=torch.float, device=model.device)
        ep_reward = 0
        for t in count():
            # env.render()
            # Select action and execute it
            action = model.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            ep_reward += reward
            # next_state = normalize_values(torch.tensor(next_state, dtype=torch.float, device=model.device))
            next_state = torch.tensor(next_state, dtype=torch.float, device=model.device)
            reward = torch.tensor([reward], dtype=torch.float, device=model.device)
            done = torch.tensor([done], dtype=torch.bool, device=model.device)

            model.memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, done)
            state = next_state

            # Perform one step of the optimization (on the target network)
            model.optimize_model()
            if done:
                # if reward > 0:
                if ep_reward > 100:
                    print(f"Success!! Ep. reward: {ep_reward}")
                ep_rewards.append(ep_reward)
                # plot_info(ep_rewards, 'Episode rewards', ('N. episode', 'Reward'))
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % model.TARGET_UPDATE == 0:
            model.target_net.load_state_dict(model.policy_net.state_dict())
        if i_episode % 100 == 0:
            plot_info(np.array(ep_rewards), 'Episode rewards', ('Episode', 'Reward'))

    plot_info(np.array(ep_rewards), 'Episode rewards', ('Episode', 'Reward'))
    cum_reward = [ep_rewards[0]]
    for val in ep_rewards[1:]:
        cum_reward.append(val + cum_reward[-1])
    plot_info(cum_reward, 'Cumulative reward', ('Episode', 'Reward'))

    # Save model
    model.save_models(path_to_weights)
    print('Training complete')


def test(num_episodes, env, model, path_to_weights):
    model.load_models(path_to_weights)
    model.policy_net.eval()
    model.target_net.eval()
    env.render()
    for _ in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=model.device)
        ep_reward = 0
        for t in count():
            with torch.no_grad():
                action = model.policy_net(state).argmax().view(1, 1)
            next_state, reward, done, _ = env.step(action.item())
            ep_reward += reward
            next_state = torch.tensor(next_state, dtype=torch.float, device=model.device)
            state = next_state
            env.render()
            if done:
                print(f"Episode reward: {ep_reward}")
                break
    print('Testing complete')


if __name__ == '__main__':
    num_episodes, mode, path_to_weights = parse_arguments()
    path_to_weights = Path(path_to_weights) if path_to_weights != "" else Path.cwd()

    # env = gym.make('household_env:Household-v0')
    env = gym.make('CartPole-v0')
    env.reset()

    tasks_list = [Tasks.TURN_ON_TV]
    # env.set_current_task(tasks_list[0])
    q_learning = QLearning(env.observation_space.shape[0], env.action_space.n)

    if mode == "train":
        train(num_episodes, env, q_learning, path_to_weights)
    elif mode == "test":
        test(num_episodes, env, q_learning, path_to_weights)
    else:
        print(f"Only valid modes are 'train' or 'test'. User input was {mode}")

    env.close()
