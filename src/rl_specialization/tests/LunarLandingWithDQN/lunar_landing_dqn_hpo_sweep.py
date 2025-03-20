import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wandb
import pprint

wandb.login()

rl_goal = "total_reward"

sweep_config = {
    'method': 'random'
    }

metric = {
    'name': rl_goal,
    'goal': 'maximize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'minibatch_sz': {
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      },
    'step_size': {
        'distribution': 'uniform',
        'min': 1e-5,
        'max': 1e-2,
      },
    'state_dim': {
        'value': 8},
    'num_hidden_units': {
        'value': 128},
    'num_actions': {
        'value': 4},
    'replay_buffer_size': {
        'value': 10000},
    'gamma': {
        'value': 0.99},
    'seed': {
        'value': 42},
    'num_episodes': {
        'value': 300}
    }

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")


class Agent():
    def __init__(self, config):
        self.name = "dqn_agent"
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size, config.minibatch_sz)
        self.network = ActionValueNetwork(config).to(device)
        self.target_net = ActionValueNetwork(config).to(device)
        self.target_net.load_state_dict(self.network.state_dict())
        step_size = config.step_size
        self.optimizer = optim.AdamW(self.network.parameters(), lr=step_size, amsgrad=True)
        # self.optimizer = optim.Adam(self.network.parameters(), lr = step_size, betas=(beta_m, beta_v), eps=epsilon)
        self.num_actions = config.num_actions
        self.discount = config.gamma
        
        self.rand_generator = np.random.RandomState(config.seed)
        
        self.eps = 1e-1

    def greedy_policy(self, state):
        with torch.no_grad():
            _, max_action_indices = self.network(state).max(dim=1, keepdims=True)
        return max_action_indices.cpu().data.numpy()[0][0]

    def eps_greedy_policy(self, state):
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action. 
        """
        if self.rand_generator.random() < self.eps:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.greedy_policy(state)

        return action

    # Work Required: No.
    def start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        eps_min = 1e-3
        decay=0.99
        self.eps = max(eps_min, self.eps*decay)

        # self.last_state = torch.tensor(state)
        return self.eps_greedy_policy(state)

    # Work Required: Yes. Fill in the action selection, replay-buffer update, 
    # weights update using optimize_network, and updating last_state and last_action (~5 lines).
    def step(self, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
                
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            # Get sample experiences from the replay buffer
            experiences = self.replay_buffer.sample()
            optimize_network(experiences, self.discount, self.optimizer, self.network, self.target_net)
                
        TAU = 0.005
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        action = self.eps_greedy_policy(state)
        
        return action

class ReplayBuffer(object):

    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.minibatch_size = batch_size

    def append(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.minibatch_size)

    def size(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'is_terminal'))

def compute_target_dqn(reward_batch, next_state_batch, terminated_batch, discount, current_q):
    with torch.no_grad():
        v_next_vec, indices = current_q(next_state_batch).max(dim=1)

    ones_vec = torch.ones_like(terminated_batch)
    target = reward_batch + discount * v_next_vec * (ones_vec - terminated_batch)
    return target.reshape(len(reward_batch), 1)

def optimize_network(experiences, discount, optimizer, network, target_net):
    """
    Args:
        experiences (Numpy array): The batch of experiences including the states, actions, 
                                   rewards, terminals, and next_states.
        discount (float): The discount factor.
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        target_net (ActionValueNetwork): The fixed network used for computing the targets, 
                                        and particularly, the action-values at the next-states.
    """
    
    batch = Transition(*zip(*experiences))

    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    is_terminal_batch = torch.cat(batch.is_terminal)
    batch_size = len(state_batch)  

    action_batch = torch.cat(batch.action)
    action_batch = action_batch.reshape((batch_size,1))
    state_action_values = network(state_batch).gather(1, action_batch)

    target = compute_target_dqn(reward_batch, next_state_batch, is_terminal_batch, discount, target_net)

    smoothL1 = nn.SmoothL1Loss()
    loss = smoothL1(state_action_values, target)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(network.parameters(), 100)
    optimizer.step()

class ActionValueNetwork(nn.Module):
    def __init__(self, network_config):
        super(ActionValueNetwork, self).__init__()
        self.state_dim = network_config.state_dim
        self.num_hidden_units = network_config.num_hidden_units
        self.num_actions = network_config.num_actions
        
        self.layer1 = nn.Linear(self.state_dim, self.num_hidden_units)
        self.layer2 = nn.Linear(self.num_hidden_units, self.num_hidden_units)
        self.layer3 = nn.Linear(self.num_hidden_units, self.num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def train(wandb_config=None):
    
        # Initialize a new wandb run
    with wandb.init(config=wandb_config):
        # this config will be set by Sweep Controller
        config = wandb.config

        env = gym.make("LunarLander-v3")

        agent = Agent(config)

        for episode in np.arange(config.num_episodes):
            num_steps = 0
            last_state, info = env.reset()
            last_state = torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)
            done = False
            total_reward = 0 

            while not done:

                action = agent.step(last_state)
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                reward = torch.tensor([reward], dtype=torch.float32, device=device)
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)
                terminated = torch.tensor(terminated, dtype=torch.float32, device=device).unsqueeze(0)
                action_tensor = torch.tensor([action], dtype=torch.int64, device=device)
                agent.replay_buffer.append(last_state, action_tensor, reward, new_state, terminated)

                total_reward += reward

                last_state = new_state
                last_action = action

                num_steps += 1

            print("Episode", episode, "Total reward", total_reward, "Steps", num_steps)
            wandb.log({rl_goal: total_reward, "episode": episode, "num_steps": num_steps})

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

wandb.agent(sweep_id, train, count=5)