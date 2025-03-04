#!/usr/bin/env python
# coding: utf-8

import os
os.environ["XDG_SESSION_TYPE"] = "xcb"

# import jdc
import numpy as np
import matplotlib.pyplot as plt
from RLGlue.rl_glue import RLGlue
from Agent import BaseAgent 
from Environment import BaseEnvironment  
from manager import Manager
from itertools import product
from tqdm import tqdm

def isInBounds(x, y, width, height):
    return (x < height) and (y < width) and (x >= 0) and (y >= 0)

class CliffWalkEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        reward = None
        state = None # See Aside
        termination = None
        self.reward_state_term = (reward, state, termination)
        
        self.grid_h = env_info.get("grid_height", 4) 
        self.grid_w = env_info.get("grid_width", 12)
        
        self.start_loc = (self.grid_h - 1, 0)
        self.goal_loc = (self.grid_h - 1, self.grid_w - 1)
        
        self.cliff = [(self.grid_h - 1, i) for i in range(1, (self.grid_w - 1))]

    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state from the environment.
        """
        reward = 0
        # agent_loc will hold the current location of the agent
        self.agent_loc = self.start_loc
        # state is the one dimensional state representation of the agent location.
        state = self.state(self.agent_loc)
        termination = False
        self.reward_state_term = (reward, state, termination)

        return self.reward_state_term[1]


    # Fill in the code for action UP and implement the logic for reward and termination.
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """

        x, y = self.agent_loc

        # UP
        if action == 0:
            x = x - 1

        # LEFT
        elif action == 1:
            y = y - 1

        # DOWN
        elif action == 2:
            x = x + 1

        # RIGHT
        elif action == 3:
            y = y + 1

        # Uh-oh
        else: 
            raise Exception(str(action) + " not in recognized actions [0: Up, 1: Left, 2: Down, 3: Right]!")

        if not isInBounds(x, y, self.grid_w, self.grid_h):
            x, y = self.agent_loc

        # assign the new location to the environment object
        self.agent_loc = (x, y)

        # by default, assume -1 reward per step and that we did not terminate
        reward = -1
        terminal = False

        if (x == self.grid_h - 1) and (y == self.grid_w - 1):
            terminal = True
        elif (x == self.grid_h - 1) and (y>0) and (y<self.grid_w - 1):
            reward = -100
            self.agent_loc = self.start_loc

        self.reward_state_term = (reward, self.state(self.agent_loc), terminal)
        return self.reward_state_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        self.agent_loc = self.start_loc
    
    # helper method
    def state(self, loc):
        return self.grid_w * loc[0] + loc[1]

class TDAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.policy = agent_info.get("policy")
        self.discount = agent_info.get("discount")
        self.step_size = agent_info.get("step_size")
        self.values = np.zeros((self.policy.shape[0],))
        
    def agent_start(self, state):
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state
        return action

    def agent_step(self, reward, state):
        target = reward + self.discount*self.values[state]
        self.values[self.last_state] += self.step_size * (target - self.values[self.last_state])
        action = self.rand_generator.choice(range(self.policy.shape[1]), p=self.policy[state])
        self.last_state = state

        return action

    def agent_end(self, reward):
        self.values[self.last_state] += self.step_size * (reward - self.values[self.last_state])

    def agent_cleanup(self):        
        """Cleanup done after the agent ends."""
        self.last_state = None

    def agent_message(self, message):
        if message == "get_values":
            return self.values
        else:
            raise Exception("TDAgent.agent_message(): Message not understood!")

def run_experiment(env_info, agent_info,num_episodes=5000, experiment_name=None, plot_freq=100, true_values_file=None, value_error_threshold=1e-8):
    env = CliffWalkEnvironment
    agent = TDAgent
    rl_glue = RLGlue(env, agent)

    rl_glue.rl_init(agent_info, env_info)

    manager = Manager(env_info, agent_info, true_values_file=true_values_file, experiment_name=experiment_name)
    for episode in tqdm(range(1, num_episodes + 1)):
        rl_glue.rl_episode(0) # no step limit
        if episode % plot_freq == 0:
            values = rl_glue.agent.agent_message("get_values")
            manager.visualize(values, episode)

    values = rl_glue.agent.agent_message("get_values")
    return values

env_info = {"grid_height": 4, "grid_width": 12, "seed": 0}
agent_info = {"discount": 1, "step_size": 0.01, "seed": 0}

# The Optimal Policy that strides just along the cliff
policy = np.ones(shape=(env_info['grid_width'] * env_info['grid_height'], 4)) * 0.25
policy[36] = [1, 0, 0, 0]
for i in range(24, 35):
    policy[i] = [0, 0, 0, 1]
policy[35] = [0, 0, 1, 0]

agent_info.update({"policy": policy})

true_values_file = "optimal_policy_value_fn.npy"
_ = run_experiment(env_info, agent_info, num_episodes=500, experiment_name="Policy Evaluation on Optimal Policy",
                   plot_freq=50, true_values_file=true_values_file)
