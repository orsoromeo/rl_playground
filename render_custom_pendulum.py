import gym
import gym_custom_pendulum
import torch

env = gym.make('gym_custom_pendulum/Pendulum-v0')
env.reset()

done = False
while not done:
    # env.render()
    action = env.action_space.sample()
    print("random action", action)
    observation, reward, truncated, terminated, _ = env.step(action)
    print(observation)
    done = truncated or terminated