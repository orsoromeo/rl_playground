import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions). 
                       The action-values computed by an action-value network.              
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """
    
    preferences = action_values.cpu().data.numpy() / tau
    max_preference = np.max(action_values.cpu().data.numpy()) / tau    
    
    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting 
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))
    
    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = None
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = None
    
    # your code here
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences)
    
    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting 
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    
    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = None
    
    # your code here
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    
    
    # squeeze() removes any singleton dimensions. It is used here because this function is used in the 
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in 
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()
    return action_probs

class Agent():
    def __init__(self):
        self.name = "expected_sarsa_agent"
        
    # Work Required: No.
    def init(self, agent_config):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer, 
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'], 
                                          agent_config['minibatch_sz'], agent_config.get("seed"))
        self.network = ActionValueNetwork(agent_config['network_config']).to(device)
        step_size = agent_config["optimizer_config"].get("step_size")
        print("Step size", step_size)
        self.optimizer = optim.AdamW(self.network.parameters(), lr = step_size, amsgrad=True)
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']
        
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        
        self.last_state = None
        self.last_action = None
        
        self.sum_rewards = 0
        self.episode_steps = 0

    # Work Required: No.
    def policy(self, state):
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action. 
        """
        action_values = self.network(state)
        probs_batch = softmax(action_values, self.tau)
        action = self.rand_generator.choice(self.num_actions, p=probs_batch.squeeze())
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
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = torch.tensor(state)
        self.last_action = self.policy(state)
        return self.last_action

    # Work Required: Yes. Fill in the action selection, replay-buffer update, 
    # weights update using optimize_network, and updating last_state and last_action (~5 lines).
    def step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        
        self.sum_rewards += reward
        self.episode_steps += 1

        action = self.policy(state)
        
        action_tensor = torch.tensor([action], dtype=torch.int64, device=device)
        self.replay_buffer.append(self.last_state, action_tensor, reward, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)
                
        self.last_state = state
        self.last_action = action
        
        return action

    # Work Required: Yes. Fill in the replay-buffer update and
    # update of the weights using optimize_network (~2 lines).
    def end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        
        # Set terminal state to an array of zeros
        state = torch.zeros(self.last_state.shape, device=device)

        # your code here
        action_tensor = torch.tensor([self.last_action], dtype=torch.int64, device=device)
        self.replay_buffer.append(self.last_state, action_tensor, reward, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau)
                
        
    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")

class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.              
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator. 
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, next_state):
        """
        Args:
            state (Numpy array): The state.              
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.           
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, next_state])

    def sample(self):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

def compute_target(reward_batch, next_state_batch, discount, current_q, tau):
    with torch.no_grad():
        next_state_values = current_q(next_state_batch)
    probs_mat = softmax(next_state_values, tau)
    v_next_vec = np.sum(probs_mat*next_state_values.cpu().data.numpy())
    target = reward_batch + discount * v_next_vec
    return torch.tensor(target)

def optimize_network(experiences, discount, optimizer, network, current_q, tau):
    """
    Args:
        experiences (Numpy array): The batch of experiences including the states, actions, 
                                   rewards, terminals, and next_states.
        discount (float): The discount factor.
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets, 
                                        and particularly, the action-values at the next-states.
    """
    
    batch = Transition(*zip(*experiences))

    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    batch_size = len(state_batch)  

    action_batch = torch.cat(batch.action)
    action_batch = action_batch.reshape((batch_size,1))
    state_action_values = network(state_batch)
    state_action_values = state_action_values.gather(1, action_batch)

    target = compute_target(reward_batch, next_state_batch, discount, current_q, tau)

    mse_loss = nn.MSELoss()
    loss = mse_loss(state_action_values, target)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class ActionValueNetwork(nn.Module):
    # Work Required: Yes. Fill in the layer_sizes member variable (~1 Line).
    def __init__(self, network_config):
        super(ActionValueNetwork, self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        
        self.layer1 = nn.Linear(self.state_dim, self.num_hidden_units)
        torch.nn.init.normal_(self.layer1.weight)
        self.layer2 = nn.Linear(self.num_hidden_units, self.num_actions)
        torch.nn.init.normal_(self.layer2.weight)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)
    
def run_experiment(num_episodes, agent_info):

    env = gym.make("LunarLander-v3", max_episode_steps=500)

    agent = Agent()
    agent.init(agent_info)
    rewards = torch.zeros(num_episodes)

    for episode in np.arange(num_episodes):
        num_steps = 0
        last_state, info = env.reset(seed=agent_info.get("seed"))
        last_state = torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        last_action = agent.start(last_state)
        total_reward = 0 

        while not done:

            new_state, reward, terminated, truncated, _ = env.step(last_action)
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            new_state = torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated
            
            action = agent.step(reward, last_state)
            
            total_reward += reward

            if done:
                agent.end(reward)
            # elif truncated:
            #     agent.step(reward, new_state)
            else:
                num_steps += 1

            last_state = new_state
            last_action = action

        print("Episode", episode, "Total reward", total_reward, "Steps", num_steps)
        rewards[episode] = total_reward
    return rewards.cpu().data.numpy()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

step_sizes = 3e-5 * np.power(2.0, np.array([-2, -1, 0, 1, 2, 3, 4, 5]))
num_tests = 1
num_episodes = 300

agent_info = {
         'network_config': {
             'state_dim': 8,
             'num_hidden_units': 256,
             'num_hidden_layers': 1,
             'num_actions': 4
         },
         'optimizer_config': {
             'step_size': 1e-3
         },
         'replay_buffer_size': 50000,
         'minibatch_sz': 8,
         'num_replay_updates_per_step': 4,
         'gamma': 0.99,
         'tau': 0.001,
         'seed': 0}

results = np.zeros((1, num_episodes))
for _ in np.arange(num_tests):
    results += run_experiment(num_episodes, agent_info)
results /= num_tests
plt.plot(results[0])

# results = np.zeros((len(step_sizes), num_episodes))
# for i, step_size in enumerate(step_sizes):
#     agent_info["optimizer_config"]["step_size"] = step_size
#     results[i, :] = run_experiment(num_episodes, agent_info)
# print(results.T)
# plt.plot(results.T)

plt.show()