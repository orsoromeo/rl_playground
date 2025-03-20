import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import os
import logging
import cv2
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

# Check versions
import importlib.metadata


print(f"torch version: {importlib.metadata.version('torch')}")
print(f"gymnasium version: {importlib.metadata.version('gymnasium')}")
# print(f"sb3 version: {importlib.metadata.version('stable-baselines3')}")
print(f"cv2 version: {importlib.metadata.version('opencv-python')}")
print(f"ax version: {importlib.metadata.version('ax-platform')}")

import wandb

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

wandb.login()


# Make wandb be quiet
os.environ["WANDB_SILENT"] = "true"
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)


def set_random_seeds(seed: int) -> None:
  """
  Seed the different random generators.
  """

  # Set seed for Python random, NumPy, and Torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Set deterministic operations for CUDA
  if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'], agent_config['minibatch_sz'])
        self.network = ActionValueNetwork(agent_config['network_config']).to(device)
        self.target_net = ActionValueNetwork(agent_config['network_config']).to(device)
        self.target_net.load_state_dict(self.network.state_dict())
        step_size = agent_config["optimizer_config"].get("step_size")
        beta_m = agent_config["optimizer_config"].get("beta_m")
        beta_v = agent_config["optimizer_config"].get("beta_v")
        epsilon = agent_config["optimizer_config"].get("epsilon")
        print("Step size", step_size)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=step_size, amsgrad=True)
        # self.optimizer = optim.Adam(self.network.parameters(), lr = step_size, betas=(beta_m, beta_v), eps=epsilon)
        self.num_actions = agent_config['network_config']['num_actions']
        self.discount = agent_config['gamma']
        
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        
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
        # if self.rand_generator.random() < self.eps:
        if self.rand_generator.random() < 0.1:
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
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        
        self.layer1 = nn.Linear(self.state_dim, self.num_hidden_units)
        self.layer2 = nn.Linear(self.num_hidden_units, self.num_hidden_units)
        self.layer3 = nn.Linear(self.num_hidden_units, self.num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def run_eval(num_eval_episodes, env, agent, video=None, msg=None):

    rewards = torch.zeros(num_eval_episodes)

    for episode in np.arange(num_eval_episodes):
        num_steps = 0
        last_state, info = env.reset()
        last_state = torch.tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        total_reward = 0 

        while not done:

            action = agent.greedy_policy(last_state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_state = torch.tensor(new_state, dtype=torch.float32, device=device).unsqueeze(0)

            total_reward += reward

            last_state = new_state
            last_action = action

            num_steps += 1
            if video:
                frame = env.render()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.putText(
                    frame,                        # Image
                    f"Random, episode {episode}",                          # Text to add
                    (10, 25),                     # Origin of text in image
                    cv2.FONT_HERSHEY_SIMPLEX,     # Font
                    1,                            # Font scale
                    (0, 0, 0),                    # Color
                    2,                            # Thickness
                    cv2.LINE_AA                   # Line type
                )
                video.write(frame)
        
        print("Evaluation episode", episode, "Total reward", total_reward, "Steps", num_steps)
        rewards[episode] = total_reward

    return rewards.cpu().data.numpy()

# Recorder settings
FPS = 30
FOURCC = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
VIDEO_FILENAME = "1-random.mp4"

# Try runnign a few episodes with the environment and random actions
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

agent_info_test = {
         'network_config': {
             'state_dim': 8,
             'num_hidden_units': 128,
             'num_hidden_layers': 2,
             'num_actions': 4
         },
         'optimizer_config': {
             'step_size': 1e-3,
         },
         'replay_buffer_size': 10000,
         'minibatch_sz': 128,
         'gamma': 0.99,
         'seed': 42}

agent = Agent()
agent.init(agent_info_test)
env = gym.make("LunarLander-v3", render_mode='rgb_array')
env.reset()
frame = env.render()
width = frame.shape[1]
height = frame.shape[0]

# Create recorder
# video = cv2.VideoWriter(VIDEO_FILENAME, FOURCC, FPS, (width, height))
# num_tests = 5

# test_rewards = run_eval(env=env, agent=agent, num_eval_episodes=num_tests, video=video)
# # print(f"Episode {ep} | Length: {ep_len}, Reward: {ep_rew}, Step time: {(step_time * 1000):.2e} ms")

# # Close the video writer
# video.release()

# class EvalAndSaveCallback(BaseCallback):
#     """
#     Evaluate and save the model every ``check_freq`` steps
#     """

#     # Constructor
#     def __init__(
#         self,
#         check_freq,
#         save_dir,
#         model_name="model",
#         replay_buffer_name=None,
#         steps_per_test=0,
#         num_tests=10,
#         step_offset=0,
#         verbose=1,
#     ):
#         super(EvalAndSaveCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.save_dir = save_dir
#         self.model_name = model_name
#         self.replay_buffer_name = replay_buffer_name
#         self.num_tests = num_tests
#         self.steps_per_test = steps_per_test
#         self.step_offset = step_offset
#         self.verbose = verbose

#     # Create directory for saving the models
#     def _init_callback(self):
#         if self.save_dir is not None:
#             os.makedirs(self.save_dir, exist_ok=True)

#     # Save and evaluate model at a set interval
#     def _on_step(self):
#         if self.n_calls % self.check_freq == 0: 
#             # Set actual number of steps (including offset)
#             actual_steps = self.step_offset + self.n_calls    
#             # Save model
#             model_path = os.path.join(self.save_dir, f"{self.model_name}_{str(actual_steps)}")
#             self.model.save(model_path)   
#             # Save replay buffer
#             if self.replay_buffer_name != None:
#                 replay_buffer_path = os.path.join(self.save_dir, f"{self.replay_buffer_name}")
#                 self.model.save_replay_buffer(replay_buffer_path)   
#                 # Evaluate the agent
#                 avg_ep_len, avg_ep_rew, avg_step_time = evaluate_agent(
#                     env,
#                     self.model,
#                     self.steps_per_test,
#                     self.num_tests
#                 )
#             if self.verbose:
#                 print(f"{str(actual_steps)} steps | average test length: {avg_ep_len}, average test reward: {avg_ep_rew}")  
#             # Log metrics to WandB
#             log_dict = {
#                 'avg_ep_len': avg_ep_len,
#                 'avg_ep_rew': avg_ep_rew,
#                 'avg_step_time': avg_step_time,
#                 'train/actor_loss': self.model.logger.name_to_value['train/n_updates'],
#                 'train/approx_k1': self.model.logger.name_to_value['train/approx_k1'],
#                 'train/clip_fraction': self.model.logger.name_to_value['train/clip_fraction'],
#                 'train/clip_range': self.model.logger.name_to_value['train/clip_range'],
#                 'train/critic_loss': self.model.logger.name_to_value['train/critic_loss'],
#                 'train/ent_coef': self.model.logger.name_to_value['train/ent_coef'],
#                 'train/ent_coef_loss': self.model.logger.name_to_value['train/ent_coef_loss'],
#                 'train/entropy_loss': self.model.logger.name_to_value['train/entropy_loss'],
#                 'train/explained_variance': self.model.logger.name_to_value['train/explained_variance'],
#                 'train/learning_rate': self.model.logger.name_to_value['train/learning_rate'],
#                 'train/loss': self.model.logger.name_to_value['train/loss'],
#                 'train/n_updates': self.model.logger.name_to_value['train/n_updates'],
#                 'train/policy_gradient_loss': self.model.logger.name_to_value['train/policy_gradient_loss'],
#                 'train/value_loss': self.model.logger.name_to_value['train/value_loss'],
#                 'train/std': self.model.logger.name_to_value['train/std'],
#             }
#             wandb.log(log_dict, commit=True, step=actual_steps) 
#         return True

# class WandBWriter(KVWriter):
#     """
#     Log metrics to Weights & Biases when called by .learn()
#     """ 
#     # Initialize run
#     def __init__(self, run, verbose=1):
#       super().__init__()
#       self.run = run
#       self.verbose = verbose    
      
#     # Write metrics to W&B project
#     def write(
#       self,
#       key_values: Dict[str, Any],
#       key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
#       step: int = 0,
#     ) -> None:
#         log_dict = {}   
#         # Go through each key/value pairs
#         for (key, value), (_, excluded) in zip(
#             sorted(key_values.items()), sorted(key_excluded.items())):    
#             if self.verbose >= 2:
#                 print(f"step={step} | {key} : {value} ({type(value)})") 
#             # Skip excluded items
#             if excluded is not None and "wandb" in excluded:
#                 continue    
#             # Log integers and floats
#             if isinstance(value, np.ScalarType):
#                 if not isinstance(value, str):
#                     wandb.log(data={key: value}, step=step)
#                     log_dict[key] = value 
#         # Print to console
#         if self.verbose >= 1:
#             print(f"Log for steps={step}")
#             print(f"--------------")
#             for (key, value) in sorted(log_dict.items()):
#                 print(f"  {key}: {value}")
#             print()

#     # Close the W&B run
#     def close(self) -> None:
#         self.run.finish()


def do_trial(settings, hparams):
    """
    Training loop used to evaluate a set of hyperparameters
    """ 
    # Set random seed
    set_random_seeds(settings['seed'])   
    # Create new W&B run
    config = {}
    dt = datetime.datetime.now(datetime.timezone.utc)
    dt = dt.replace(microsecond=0, tzinfo=None)
    run = wandb.init(
        project=settings['wandb_project'],
        name=str(dt),
        config=config,
        settings=wandb.Settings(silent=(not settings['verbose_wandb']))
    )   
    # Print run info
    if settings['verbose_trial'] > 0:
      print(f"WandB run ID: {run.id}")
      print(f"WandB run name: {run.name}")  
    # Log hyperparameters to W&B
    wandb.config.update(hparams)    
    # Set custom Logger with our custom writer
    # wandb_writer = WandBWriter(run, verbose=settings['verbose_log'])
    # loggers = Logger(
    #     folder=None,
    #     output_formats=[wandb_writer]
    # )   
    # Calculate derived hyperparameters
    n_steps = 2 ** hparams['steps_per_update_pow2']
    minibatch_size = (hparams['n_envs'] * n_steps) // (2 ** hparams['batch_size_div_pow2'])
    layer_1 = 2 ** hparams['layer_1_pow2']
    layer_2 = 2 ** hparams['layer_2_pow2']  
    # Set completed steps to checkpoint number (in filename) or 0 to start over
    # TODO: how to resume if trial is paused/cancelled
    completed_steps = 0 
    # Load or create new model
    if completed_steps != 0:
      model_path = os.path.join(settings['save_dir'], f"{settings['model_name']}_{str(completed_steps)}.zip")
      model = sb3.PPO.load(model_path, env)
      steps_to_complete = settings['total_steps'] - completed_steps
    else:
      model = sb3.PPO(
          'MlpPolicy',
          env,
          learning_rate=hparams['learning_rate'],
          n_steps=n_steps,
          batch_size=minibatch_size,
          gamma=hparams['gamma'],
          ent_coef=hparams['entropy_coef'],
          use_sde=hparams['use_sde'],
          sde_sample_freq=hparams['sde_freq'],
          policy_kwargs={'net_arch': [layer_1, layer_2]},
          verbose=settings['verbose_train'],
      )
      steps_to_complete = settings['total_steps']   
    # Set up checkpoint callback
    checkpoint_callback = EvalAndSaveCallback(
        check_freq=settings['checkpoint_freq'],
        save_dir=settings['save_dir'],
        model_name=settings['model_name'],
        replay_buffer_name=settings['replay_buffer_name'],
        steps_per_test=settings['steps_per_test'],
        num_tests=settings['tests_per_check'],
        step_offset=(settings['total_steps'] - steps_to_complete),
        verbose=settings['verbose_test'],
    )

    # Choo choo train
    model.learn(total_timesteps=steps_to_complete,
                callback=[checkpoint_callback]) 
    # Get dataframe of run metrics
    history = wandb.Api().run(f"{run.project}/{run.id}").history()  
    # Get index of evaluation with maximum reward
    max_idx = np.argmax(history.loc[:, 'avg_ep_rew'].values)    
    # Find number of steps required to produce that maximum reward
    max_rew_steps = history['_step'][max_idx]
    if settings['verbose_trial'] > 0:
      print(f"Steps with max reward: {max_rew_steps}")  
    # Load model with maximum reward from previous run
    model_path = os.path.join(settings['save_dir'], f"{settings['model_name']}_{str(max_rew_steps)}.zip")
    model = sb3.PPO.load(model_path, env)   
    # Evaluate the agent
    avg_ep_len, avg_ep_rew, avg_step_time = evaluate_agent(
        env,
        model,
        settings['steps_per_test'],
        settings['tests_per_check'],
    )   
    # Log final evaluation metrics to WandB run
    wandb.run.summary['Average test episode length'] = avg_ep_len
    wandb.run.summary['Average test episode reward'] = avg_ep_rew
    wandb.run.summary['Average test step time'] = avg_step_time 
    # Print final run metrics
    if settings['verbose_trial'] > 0:
      print('---')
      print(f"Best model: {settings['model_name']}_{str(max_rew_steps)}.zip")
      print(f"Average episode length: {avg_ep_len}")
      print(f"Average episode reward: {avg_ep_rew}")
      print(f"Average step time: {avg_step_time}")  
    # Close W&B run
    run.finish()    
    return avg_ep_rew


# Project settings that do not change
settings = {
    'wandb_project': "pendulum-ax-hpo",
    'model_name': "ppo-pendulum",
    'ax_experiment_name': "ppo-pendulum-experiment",
    'ax_objective_name': "avg_ep_rew",
    'replay_buffer_name': None,
    'save_dir': "checkpoints",
    'checkpoint_freq': 10_000,
    'steps_per_test': 100,
    'tests_per_check': 10,
    'total_steps': 100_000,
    'num_trials': 50,
    'seed': 42,
    'verbose_ax': False,
    'verbose_wandb': False,
    'verbose_train': 0,
    'verbose_log': 0,
    'verbose_test': 0,
    'verbose_trial': 1,
}

# Define the hyperparameters we want to optimize
hparams = [
  {
    'name': "n_envs",
    'type': "fixed",
    'value_type': "int",
    'value': 1,
  },
  {
    'name': "learning_rate",
    'type': "range",
    'value_type': "float",
    'bounds': [1e-5, 1e-2],
    'log_scale': True,
  },
  {
    'name': "steps_per_update_pow2",
    'type': "range",
    'value_type': "int",
    'bounds': [6, 12],    # Inclusive, 2**n between [64, 4096]
    'log_scale': False,
  },
  {
    'name': "batch_size_div_pow2",
    'type': "range",
    'value_type': "int",
    'bounds': [0, 3],    # Inclusive, 2**n between [1, 8]
    'log_scale': False,
  },
  {
    'name': "gamma",
    'type': "range",
    'value_type': "float",
    'bounds': [0.9, 0.99],
    'log_scale': False,
  },
  {
    'name': "entropy_coef",
    'type': "range",
    'value_type': "float",
    'bounds': [0.0, 0.1],
    'log_scale': False,
  },
  {
    'name': "use_sde",
    'type': "choice",
    'value_type': "bool",
    'values': [True, False],
    'is_ordered': False,
    'sort_values': False,
  },
  {
    'name': "sde_freq",
    'type': "range",
    'value_type': "int",
    'bounds': [-1, 8],
    'log_scale': False,
  },
  {
    'name': "layer_1_pow2",
    'type': "range",
    'value_type': "int",
    'bounds': [5, 8],    # Inclusive, 2**n between [32, 256]
    'log_scale': False,
  },
  {
    'name': "layer_2_pow2",
    'type': "range",
    'value_type': "int",
    'bounds': [5, 8],    # Inclusive, 2**n between [32, 256]
    'log_scale': False,
  },
]

# Set parameter constraints
parameter_constraints = []

# Create our environment
try:
  env.close()
except NameError:
  pass
env = gym.make('Pendulum-v1', render_mode='rgb_array')

# Construct path to Ax experiment snapshot file
ax_snapshot_path = os.path.join(settings['save_dir'], f"{settings['ax_experiment_name']}.json")

# Load experiment from snapshot if it exists, otherwise create a new one
if os.path.exists(ax_snapshot_path):
    print(f"Loading experiment from snapshot: {ax_snapshot_path}")
    ax_client = AxClient.load_from_json_file(ax_snapshot_path)
else:
    print(f"Creating new experiment. Snapshot to be saved at {ax_snapshot_path}.")
    ax_client = AxClient(
        random_seed=settings['seed'],
        verbose_logging=settings['verbose_ax']
    )
    ax_client.create_experiment(
        name=settings['ax_experiment_name'],
        parameters=hparams,
        # objective_name=settings['ax_objective_name'],
        # minimize=False,
        objectives={"hartmann6": ObjectiveProperties(minimize=False)},
        parameter_constraints=parameter_constraints,
    )
    
# Choo choo! Perform trials to optimize hyperparameters
while True:

    # Get next hyperparameters and end experiment if we've reached max trials
    next_hparams, trial_index = ax_client.get_next_trial()
    if trial_index >= settings['num_trials']:
        break 
    # Show that we're starting a new trial
    if settings['verbose_trial'] > 0:
        print(f"--- Trial {trial_index} ---") 
    # Perform trial
    avg_ep_rew = do_trial(settings, next_hparams)
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data=avg_ep_rew,
    )   
    # Save experiment snapshot
    ax_client.save_to_json_file(ax_snapshot_path)


def run_experiment(num_episodes, evaluation_iterations, agent_info, num_evaluations=20):

    env = gym.make("LunarLander-v3", render_mode='rgb_array')

    # Render the environment (render is not the observation!) and get width/height
    # env.reset()
    # frame = env.render()
    # width = frame.shape[1]
    # height = frame.shape[0]

    # Show frame
    # print("frame shape", frame.shape)
    # plt.imshow(frame)

    agent = Agent()
    agent.init(agent_info)
    rewards = torch.zeros(num_episodes)
    evaluation_returns = list()

    for episode in np.arange(num_episodes):
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
        rewards[episode] = total_reward

        if (episode + 1) % evaluation_iterations == 0:
            evaluation_returns.append(run_eval(num_evaluations, env, agent))

    return rewards.cpu().data.numpy(), evaluation_returns


num_tests = 1
num_episodes = 300
evaluation_iterations = 100
episodes_num_per_evaluation = 20

agent_info = {
         'network_config': {
             'state_dim': 8,
             'num_hidden_units': 128,
             'num_hidden_layers': 2,
             'num_actions': 4
         },
         'optimizer_config': {
             'step_size': 1e-3,
         },
         'replay_buffer_size': 10000,
         'minibatch_sz': 128,
         'gamma': 0.99,
         'seed': 0}

results = np.zeros((1, num_episodes))
for _ in np.arange(num_tests):
    training_res, evaluation_res = run_experiment(num_episodes, evaluation_iterations, agent_info, episodes_num_per_evaluation)
    for idx, run in enumerate(evaluation_res):
        plt.plot(run, label='Evaluation '+str(idx))
    plt.legend()
    plt.show()

    results += training_res

results /= num_tests
plt.plot(results[0], label='Training')
plt.legend()
plt.show()