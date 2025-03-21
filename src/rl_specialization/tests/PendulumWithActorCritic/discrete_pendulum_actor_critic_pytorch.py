import numpy as np
import matplotlib.pyplot as plt

import os
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import namedtuple, deque

from pendulum_env import DiscretizedPendulumEnvironment
import time
import random

class ReplayBuffer(object):

    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.minibatch_size = batch_size

    def append(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        samples = random.sample(self.memory, self.minibatch_size)
        return samples

    def size(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class Actor(nn.Module):
    def __init__(self, network_config):
        super(Actor, self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")

        self.layer1 = nn.Linear(self.state_dim, self.num_hidden_units)
        self.layer2 = nn.Linear(self.num_hidden_units, self.num_hidden_units)
        self.layer3 = nn.Linear(self.num_hidden_units, self.num_actions)

        self.softmax = torch.nn.Softmax()

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        temperature=1.0
        return self.softmax(self.layer3(x)/temperature)

class Critic(nn.Module):
    def __init__(self, network_config):
        super(Critic, self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")

        self.layer1 = nn.Linear(self.state_dim, self.num_hidden_units)
        self.layer2 = nn.Linear(self.num_hidden_units, self.num_hidden_units)
        self.layer3 = nn.Linear(self.num_hidden_units, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ActorCriticSoftmaxAgent(): 
    def __init__(self, agent_info={}):
        # set random seed for each run
        self.rand_generator = np.random.RandomState() 

        num_actions = agent_info["network_config"].get("num_actions")

        # initialize self.tc to the tile coder we created
        # self.tc = PendulumTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        # set step-size accordingly (we normally divide actor and critic step-size by num. tilings (p.217-218 of textbook))
        self.actor_step_size = agent_info.get("actor_step_size")[0]
        self.critic_step_size = agent_info.get("critic_step_size")[0]
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")[0]
        self.actions = list(range(num_actions))

        # Set initial values of average reward, actor weights, and critic weights
        # We initialize actor weights to three times the iht_size. 
        # Recall this is because we need to have one set of weights for each of the three actions.
        self.avg_reward = 0.0
        self.actor_w = Actor(agent_info['network_config'])
        self.critic_w = Critic(agent_info['network_config'])

        self.critic_optimizer = optim.Adam(self.critic_w.parameters(), lr=self.critic_step_size)
        self.actor_optimizer = optim.Adam(self.actor_w.parameters(), lr=self.actor_step_size)

        capacity = 10000
        minibatch_size = 128
        self.replay_buffer = ReplayBuffer(capacity, minibatch_size)


    def agent_policy(self, state):
        
        # compute softmax probability
        softmax_prob = self.actor_w(state)
        
        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        # print("probs",softmax_prob.cpu().data.numpy()[0])
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob.cpu().data.numpy()[0])
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
        action = self.agent_policy(state)
        return action

    def agent_step(self, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on 
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            experiences = self.replay_buffer.sample()
            batch = Transition(*zip(*experiences))
            prev_state_batch = torch.cat(batch.state)
            prev_action_batch = torch.cat(batch.action).reshape(self.replay_buffer.minibatch_size, 1)
            reward_batch = torch.cat(batch.reward).reshape(self.replay_buffer.minibatch_size, 1)
            next_state_batch = torch.cat(batch.next_state)
            # Estimate target update
            current_state_value = self.critic_w(next_state_batch)
            prev_state_value = self.critic_w(prev_state_batch)
            target = reward_batch - self.avg_reward + current_state_value
            td_error = target - prev_state_value
            td_error = torch.tensor(td_error, dtype=torch.float32, requires_grad=True)
            # print("value", current_state_value.shape, td_error.shape, reward_batch.shape)

            ### update average reward using Equation (2) (1 line)
            self.avg_reward += self.avg_reward_step_size * td_error

            optimize_critic(target, prev_state_value, self.critic_optimizer, self.critic_w)
            optimize_actor(target, prev_state_value, prev_state_batch, prev_action_batch, self.actor_optimizer, self.actor_w)

        current_action = self.agent_policy(state)
        # ----------------

        return current_action

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

def optimize_critic(target, prev_value, optimizer, critic_net):
    # delta = target - prev_value
    # loss = delta**2
    # print(target.shape, prev_value.shape)
    mse_loss = nn.MSELoss()
    loss = mse_loss(target, prev_value)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_value_(network.parameters(), 100)
    optimizer.step()

def optimize_actor(target, prev_state_value, state, action, optimizer, actor_net):
    # print("action", action, actor_net(state))
    # print("prob", state.shape, action.shape)
    # prob = actor_net(state).gather(1, action)
    probs = actor_net(state)
    p = Categorical(probs)
    delta = target - prev_state_value
    loss = - (p.log_prob(action)*delta.detach()).mean()
    # loss = -torch.log(prob)*delta.detach()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Define function to run experiment
def run_experiment(environment_parameters, agent_parameters, experiment_parameters):
    environment = DiscretizedPendulumEnvironment()
    total_return_filename = list()
    exp_avg_reward_filename = list()

    # sweep agent parameters
    for actor_ss in agent_parameters["actor_step_size"]:
        for critic_ss in agent_parameters["critic_step_size"]:
            for avg_reward_ss in agent_parameters["avg_reward_step_size"]:
                
                env_info = {}
                agent_info = {
                              "actor_step_size": actor_ss,
                              "critic_step_size": critic_ss,
                              "avg_reward_step_size": avg_reward_ss,
                              "num_actions": agent_parameters["network_config"].get("num_actions")
                              }
    
                # results to save
                return_per_step = np.zeros((experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
                exp_avg_reward_per_step = np.zeros((experiment_parameters["num_runs"], experiment_parameters["max_steps"]))
                # using tqdm we visualize progress bars 
                for run in tqdm(range(1, experiment_parameters["num_runs"]+1)):
                    # env_info["seed"] = run
                    # agent_info["seed"] = run
        
                    agent = ActorCriticSoftmaxAgent(agent_parameters)
                    last_state, info = environment.env_start()
                    last_state = torch.tensor(last_state, dtype=torch.float32).unsqueeze(0)
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
                        # print("step", num_steps)
                        num_steps += 1
                        
                        # try:
                        action = agent.agent_step(last_state)
                        # except:
                        #     print("NAN!")
                        #     break                        

                        state, reward, terminated, truncated, _ = environment.env_step(last_action)
                        total_return += reward
                        return_arr.append(reward)

                        exp_avg_reward += reward - exp_avg_reward

                        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
                        last_action = torch.tensor(last_action, dtype=torch.int64).unsqueeze(0)
                        agent.replay_buffer.append(last_state, last_action, reward, state)
                        
                        return_per_step[run-1][num_steps-1] = total_return
                        exp_avg_reward_per_step[run-1][num_steps-1] = exp_avg_reward

                        last_action = action
                        last_state = state
                                                
                if not os.path.exists('results'):
                    os.makedirs('results')
        
                save_name = "ActorCriticSoftmax_actor_ss_{}_critic_ss_{}_avg_reward_ss_{}".format(actor_ss, critic_ss, avg_reward_ss)
                total_return_file = "results/{}_total_return.npy".format(save_name)
                total_return_filename.append(total_return_file)
                exp_avg_reward_file = "results/{}_exp_avg_reward.npy".format(save_name)
                exp_avg_reward_filename.append(exp_avg_reward_file)
                np.save(total_return_file, return_per_step)
                np.save(exp_avg_reward_file, exp_avg_reward_per_step)
                print(actor_ss, critic_ss, avg_reward_ss, total_return, exp_avg_reward)
    return total_return_filename, exp_avg_reward_filename
                        

#### Run Experiment

# Experiment parameters
experiment_parameters = {
    "max_steps" : 1500,
    "num_runs" : 1
}

# Environment parameters
environment_parameters = {}

# Agent parameters
# Each element is an array because we will be later sweeping over multiple values
# actor and critic step-sizes are divided by num. tilings inside the agent
actor_params = np.arange(-1, 2, step=1)
a_params = [2.0**float(n) for n in actor_params]

critic_params = np.arange(-5, -2, step=1)
c_params = [2.0**float(n) for n in critic_params]

rewards_params = np.arange(-10, -7, step=1)
r_params = [2.0**float(n) for n in rewards_params]

# print("params", actor_params, a_params)
agent_parameters = {
    'network_config': {
        'state_dim': 2,
        'num_hidden_units': 256,
        'num_hidden_layers': 2,
        'num_actions': 9
    },
    "actor_step_size": a_params,
    "critic_step_size": c_params,
    "avg_reward_step_size": r_params,
}

tot_returns_filename, avg_rewards_filename = run_experiment(environment_parameters, agent_parameters, experiment_parameters)

# avg_rewards_filename = "results/ActorCriticSoftmax_tilings_32_tiledim_8_actor_ss_0.25_critic_ss_2_avg_reward_ss_0.015625_exp_avg_reward.npy"
# tot_returns_filename = "results/ActorCriticSoftmax_tilings_32_tiledim_8_actor_ss_0.25_critic_ss_2_avg_reward_ss_0.015625_total_return.npy"

for returns_file in tot_returns_filename:
    tot_returns = np.load(returns_file)
    for run in tot_returns:
        plt.plot(run, label='Total returns')
plt.legend()
plt.show()


for avg_reward_file in avg_rewards_filename:
    avg_rewards = np.load(avg_reward_file)
    for run in avg_rewards:
        plt.plot(run, label='Avg reward')
plt.legend()
plt.show()

agent.simulate()