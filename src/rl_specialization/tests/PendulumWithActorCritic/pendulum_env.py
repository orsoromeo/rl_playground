import gymnasium as gym
import numpy as np

class DiscretizedPendulumEnvironment():
    def __init__(self, discrete_action_num=3, render=False):
        env_name = "Pendulum-v1"
        if render:
            self.env = gym.make(env_name, render_mode='human')
        else:
            self.env = gym.make(env_name)
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
        
    def env_start(self):
        state, info = self.env.reset()
        (x, y, theta_d) = state
        theta = self.cartesian_to_polar(x,y)        
        return (theta, theta_d), info
        
    def env_step(self, discrete_action):
        action_range = self.action_max - self.action_min
        assert(action_range>0)
        continuous_action = self.action_min + action_range*discrete_action 
        state, reward, terminated, truncated, info = self.env.step(continuous_action)
        x, y, thetad = state
        return (self.cartesian_to_polar(x,y), thetad), reward, terminated, truncated, info