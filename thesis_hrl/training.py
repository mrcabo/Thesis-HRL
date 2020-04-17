from household_env.envs.house_env import Tasks
import gym
import time

tasks_list = [Tasks.TURN_ON_TV]

env = gym.make('household_env:Household-v0')
env.reset()
env.set_current_task(tasks_list[0])

for e in range(10):
    print(f"Epoch: {e}")
    env.render()
    next_state, aux, done, info = env.step(env.action_space.sample())  # take a random action
    print(f"next_state: {next_state}")
    print(f"Reward: {aux}")
    if done:
        break
    time.sleep(0.1)
