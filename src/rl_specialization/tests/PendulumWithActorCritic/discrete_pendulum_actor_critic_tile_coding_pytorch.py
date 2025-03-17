import numpy as np
import matplotlib.pyplot as plt

import os
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pendulum_env import DiscretizedPendulumEnvironment
import tiles3 as tc
import time

class PendulumTileCoderPyTorch(nn.Module):
    def __init__(self, num_actions, iht_size=4096, num_tilings=32, num_tiles=8):
        super(PendulumTileCoderPyTorch, self).__init__()
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht_size = iht_size 
        self.iht = tc.IHT(self.iht_size)

        self.layer = nn.Linear(iht_size, self.num_actions)
        
    def get_tiles(self, angle, ang_vel):
        """
        Takes in an angle and angular velocity from the pendulum environment
        and returns a numpy array of active tiles.
        
        Arguments:
        angle -- float, the angle of the pendulum between -np.pi and np.pi
        ang_vel -- float, the angular velocity of the agent between -2*np.pi and 2*np.pi
        
        returns:
        tiles -- np.array, active tiles
        
        """
        angle_scaled = 0
        ang_vel_scaled = 0
        
        min_angle = -np.pi
        min_vel = -np.pi*2    
        angle_range = np.pi*2
        vel_range = 4*np.pi
        
        angle_scaled = (angle+min_angle)/angle_range*self.num_tiles
        ang_vel_scaled = (ang_vel+min_vel)/vel_range*self.num_tiles
        
        tiles = tc.tileswrap(self.iht, self.num_tilings, [angle_scaled, ang_vel_scaled], wrapwidths=[self.num_tiles, False])
                    
        return np.array(tiles)

    def one_hot(self, state, num_states):
        """
        Given num_state and a state, return the one-hot encoding of the state
        """
        # Create the one-hot encoding of state
        # one_hot_vector is a numpy array of shape (1, num_states)

        one_hot_vector = np.zeros(num_states)
        # print("state", len(state), state)
        for s in state:
            # print("s",s)
            one_hot_vector[s] = 1
        # print("one hot", one_hot_vector)
        return one_hot_vector

    def forward(self, state):
        angle, ang_vel = state 
        active_tiles = self.get_tiles(angle, ang_vel)
        x = self.one_hot(active_tiles, self.iht_size)
        x = torch.tensor(x, dtype=torch.float32)
        return self.layer(x)

class ActorCriticSoftmaxAgent(): 
    def __init__(self):
        self.rand_generator = None

        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None

        self.tc = None

        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None

        self.actions = None

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
    
    def agent_init(self, agent_info={}):
        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed")) 

        iht_size = agent_info.get("iht_size")
        num_tilings = agent_info.get("num_tilings")
        num_tiles = agent_info.get("num_tiles")
        num_actions = agent_info.get("num_actions")

        # initialize self.tc to the tile coder we created
        # self.tc = PendulumTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        # set step-size accordingly (we normally divide actor and critic step-size by num. tilings (p.217-218 of textbook))
        self.actor_step_size = agent_info.get("actor_step_size")/num_tilings
        self.critic_step_size = agent_info.get("critic_step_size")/num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")
        self.actions = list(range(num_actions))

        # Set initial values of average reward, actor weights, and critic weights
        # We initialize actor weights to three times the iht_size. 
        # Recall this is because we need to have one set of weights for each of the three actions.
        self.avg_reward = 0.0
        self.actor_w = PendulumTileCoderPyTorch(num_actions=num_actions, iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)
        self.critic_w = PendulumTileCoderPyTorch(num_actions=1, iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        self.critic_optimizer = optim.Adam(self.critic_w.parameters(), lr=self.critic_step_size, amsgrad=True)
        self.actor_optimizer = optim.Adam(self.actor_w.parameters(), lr=self.actor_step_size, amsgrad=True)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
    
    def agent_policy(self, state):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder
            
        Returns:
            The action selected according to the policy
        """
        
        # compute softmax probability
        softmax_prob = torch.nn.Softmax()(self.actor_w(state))
        
        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob.cpu().data.numpy())
        
        # save softmax_prob as it will be useful later when updating the Actor
        self.softmax_prob = softmax_prob
        
        return chosen_action

    # def select_greedy_action(self, state):
    #     angle, ang_vel = state
    #     active_tiles = self.tc.get_tiles(angle, ang_vel)
    #     softmax_prob = compute_softmax_prob(self.actor_w, active_tiles)
    #     return np.argmax(softmax_prob)

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        # angle, ang_vel = state
        # active_tiles = self.tc.get_tiles(angle, ang_vel)
        current_action = self.agent_policy(state)

        self.last_action = current_action
        # self.prev_tiles = np.copy(active_tiles)
        self.prev_state = state

        return self.last_action


    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on 
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """

        # angle, ang_vel = state

        # active_tiles = self.tc.get_tiles(angle, ang_vel)
        # print("angle, ang vel", angle, ang_vel)
        # print("Tiles", active_tiles)
        current_state_value = self.critic_w(state)
        prev_state_value = self.critic_w(self.prev_state)
        target = reward - self.avg_reward + current_state_value
        delta = target - prev_state_value

        ### update average reward using Equation (2) (1 line)
        self.avg_reward += self.avg_reward_step_size * delta

        optimize_critic(state, reward, self.prev_state, self.avg_reward, self.critic_optimizer, self.critic_w)
        optimize_actor(delta, state, self.last_action, self.actor_optimizer, self.actor_w)

        current_action = self.agent_policy(state)
        # ----------------

        # self.prev_tiles = active_tiles
        self.prev_state = state
        self.last_action = current_action

        return self.last_action


    def agent_message(self, message):
        if message == 'get avg reward':
            return self.avg_reward

    def simulate(self):
        env = DiscretizedPendulumEnvironment(render=True)
        state, info = env.env_start()
        done = False
        i=0
        while not done:
          action = self.select_greedy_action(state)
          state, reward, terminated, truncated, _ = env.env_step(action.item())
          done = terminated or truncated
          i+=1
          time.sleep(0.01)

def optimize_critic(state, reward, prev_state, avg_reward, optimizer, critic_net):

    current_state_value = critic_net(state)
    prev_state_value = critic_net(prev_state)
    target = reward - avg_reward + current_state_value
    # delta = target - prev_state_value

    smoothL1 = nn.SmoothL1Loss()
    loss = smoothL1(prev_state_value, target)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_value_(network.parameters(), 100)
    optimizer.step()

def optimize_actor(delta, state, action, optimizer, actor_net):

    preferences = actor_net(state)
    prob = torch.nn.Softmax()(preferences)[action]
    delta = torch.tensor(delta, dtype=torch.float32)
    loss = -torch.log(prob)*delta

    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_value_(network.parameters(), 100)
    optimizer.step()

# Define function to run experiment
def run_experiment(environment_parameters, agent_parameters, experiment_parameters):
    environment = DiscretizedPendulumEnvironment()
    agent = ActorCriticSoftmaxAgent()

    # sweep agent parameters
    for num_tilings in agent_parameters['num_tilings']:
        for num_tiles in agent_parameters["num_tiles"]:
            for actor_ss in agent_parameters["actor_step_size"]:
                for critic_ss in agent_parameters["critic_step_size"]:
                    for avg_reward_ss in agent_parameters["avg_reward_step_size"]:
                        
                        env_info = {}
                        agent_info = {"num_tilings": num_tilings,
                                      "num_tiles": num_tiles,
                                      "actor_step_size": actor_ss,
                                      "critic_step_size": critic_ss,
                                      "avg_reward_step_size": avg_reward_ss,
                                      "num_actions": agent_parameters["num_actions"],
                                      "iht_size": agent_parameters["iht_size"]}            
            
                        # results to save
                        return_per_step = np.zeros((experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
                        exp_avg_reward_per_step = np.zeros((experiment_parameters["num_runs"], experiment_parameters["max_steps"]))

                        # using tqdm we visualize progress bars 
                        for run in tqdm(range(1, experiment_parameters["num_runs"]+1)):
                            env_info["seed"] = run
                            agent_info["seed"] = run
                
                            agent.agent_init(agent_info)
                            last_state, info = environment.env_start()
                            last_action = agent.agent_start(last_state)

                            observation = (last_state, last_action)

                            num_steps = 0
                            total_return = 0.
                            return_arr = []

                            # exponential average reward without initial bias
                            exp_avg_reward = 0.0
                            exp_avg_reward_ss = 0.01
                            exp_avg_reward_normalizer = 0

                            while num_steps < experiment_parameters['max_steps']:
                                num_steps += 1
                                print("steps", num_steps)
                                
                                state, reward, terminated, truncated, _ = environment.env_step(last_action)

                                last_action = agent.agent_step(reward, state)
                                
                                total_return += reward
                                return_arr.append(reward)

                                exp_avg_reward_normalizer = exp_avg_reward_normalizer + exp_avg_reward_ss * (1 - exp_avg_reward_normalizer)
                                ss = exp_avg_reward_ss / exp_avg_reward_normalizer
                                exp_avg_reward += ss * (reward - exp_avg_reward)
                                
                                return_per_step[run-1][num_steps-1] = total_return
                                exp_avg_reward_per_step[run-1][num_steps-1] = exp_avg_reward
                                                        
                        if not os.path.exists('results'):
                            os.makedirs('results')
                
                        save_name = "ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_avg_reward_ss_{}".format(num_tilings, num_tiles, actor_ss, critic_ss, avg_reward_ss)
                        total_return_filename = "results/{}_total_return.npy".format(save_name)
                        exp_avg_reward_filename = "results/{}_exp_avg_reward.npy".format(save_name)

                        np.save(total_return_filename, return_per_step)
                        np.save(exp_avg_reward_filename, exp_avg_reward_per_step)
                        

#### Run Experiment

# Experiment parameters
experiment_parameters = {
    "max_steps" : 3000,
    "num_runs" : 1
}

# Environment parameters
environment_parameters = {}

# Agent parameters
# Each element is an array because we will be later sweeping over multiple values
# actor and critic step-sizes are divided by num. tilings inside the agent
agent_parameters = {
    "num_tilings": [32],
    "num_tiles": [8],
    "actor_step_size": [2**(-2)],
    "critic_step_size": [2**1],
    "avg_reward_step_size": [2**(-6)],
    "num_actions": 3,
    "iht_size": 4096
}

run_experiment(environment_parameters, agent_parameters, experiment_parameters)

avg_rewards_filename = "results/ActorCriticSoftmax_tilings_32_tiledim_8_actor_ss_0.25_critic_ss_2_avg_reward_ss_0.015625_exp_avg_reward.npy"
tot_returns_filename = "results/ActorCriticSoftmax_tilings_32_tiledim_8_actor_ss_0.25_critic_ss_2_avg_reward_ss_0.015625_total_return.npy"
avg_rewards = np.load(avg_rewards_filename)
tot_returns = np.load(tot_returns_filename)

for run in avg_rewards:
    plt.plot(run, label='avg reward')
plt.legend()
plt.show()

for run in tot_returns:
    plt.plot(run, label='total return')
plt.legend()
plt.show()

agent.simulate()