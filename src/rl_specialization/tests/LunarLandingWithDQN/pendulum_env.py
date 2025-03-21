import gymnasium as gym
import numpy as np

class DiscretizedPendulumEnvironment():
    def __init__(self, discrete_action_num=3, render=False, max_episode_steps=1000):
        env_name = "Pendulum-v1"
        if render:
            self.env = gym.make(env_name, render_mode='human', max_episode_steps=max_episode_steps)
        else:
            self.env = gym.make(env_name, max_episode_steps=max_episode_steps)
        self.action_min = self.env.action_space.low
        self.action_max = self.env.action_space.high
        self.discrete_action_num = discrete_action_num
        
    def cartesian_to_polar(self, x, y):
        if x < 0:
            if y > 0:
                theta = np.pi - np.arcsin(y)
            else:
                theta = - np.pi - np.arcsin(y)
        else:   
            theta = np.arcsin(y)
        return theta
        
    def reset(self):
        state, info = self.env.reset()
        (x, y, theta_d) = state
        theta = self.cartesian_to_polar(x,y)        
        return (theta, theta_d), info
        
    def step(self, discrete_action):
        action_range = self.action_max - self.action_min
        assert(action_range>0)
        normalized_action = discrete_action / (self.discrete_action_num - 1)
        continuous_action = self.action_min + action_range*normalized_action
        # print("actions", discrete_action, continuous_action[0])
        state, reward, terminated, truncated, info = self.env.step(continuous_action)
        x, y, thetad = state
        return (self.cartesian_to_polar(x,y), thetad), reward, terminated, truncated, info