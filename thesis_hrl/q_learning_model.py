import random
import math
from itertools import count

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from thesis_hrl.utils import ReplayMemory, Transition
from household_env.envs.house_env import Tasks


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 50)
        self.fc2 = nn.Linear(50, 25)
        self.head = nn.Linear(25, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x


class QLearning:
    def __init__(self, obs_space, action_space):
        # Hyper-parameters
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_net = QNetwork(obs_space, action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # TODO: necesario aqui?
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        # Others
        self.steps_done = 0
        self.obs_space = obs_space
        self.action_space = action_space

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).argmax().view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: not s,
                                                batch.done)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for (s, d) in zip(batch.next_state, batch.done) if not d])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # TODO: necessary?
        self.optimizer.step()


def plot_info(data, title=None, labels=None):
    plt.figure()
    plt.clf()
    if title is not None:
        plt.title(title)
    if labels is not None:
        xlabel, ylabel = labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    plt.plot(data)
    plt.show()


env = gym.make('household_env:Household-v0')

tasks_list = [Tasks.TURN_ON_TV]
env.set_current_task(tasks_list[0])
env.reset()

if __name__ == "__main__":
    q_learning = QLearning(env.observation_space.shape[0], env.action_space.n)
    num_episodes = 200
    ep_rewards = []
    for i_episode in range(num_episodes):
        print(f"Episode {i_episode}")
        state = torch.tensor(env.reset(), dtype=torch.float, device=q_learning.device)
        ep_reward = 0
        for t in count():
            env.render()
            # Select action and execute it
            action = q_learning.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            ep_reward += reward
            # Convert to tensors - TODO: I gotta see if we dont run out of GPU memory if buffer gets too big..
            next_state = torch.tensor(next_state, dtype=torch.float, device=q_learning.device)
            reward = torch.tensor([reward], dtype=torch.float, device=q_learning.device)
            done = torch.tensor([done], dtype=torch.bool, device=q_learning.device)

            # TODO: same unsqueeze for reward and done? maybe?
            q_learning.memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, done)
            state = next_state

            # Perform one step of the optimization (on the target network)
            q_learning.optimize_model()
            if done:
                ep_rewards.append(ep_reward)
                # plot_info(ep_rewards, 'Episode rewards', ('N. episode', 'Reward'))
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % q_learning.TARGET_UPDATE == 0:
            q_learning.target_net.load_state_dict(q_learning.policy_net.state_dict())

    plot_info(np.array(ep_rewards), 'Episode rewards', ('Episode', 'Reward'))
    cum_reward = [ep_rewards[0]]
    for val in ep_rewards[1:]:
        cum_reward.append(val + cum_reward[-1])
    plot_info(cum_reward, 'Cumulative reward', ('Episode', 'Reward'))
    print('Complete')
    env.close()
