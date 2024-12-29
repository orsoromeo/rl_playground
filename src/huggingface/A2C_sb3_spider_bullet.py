import gym

import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

"""## Environment 1: AntBulletEnv-v0 🕸

### Create the AntBulletEnv-v0
#### The environment 🎮
In this environment, the agent needs to use correctly its different joints to walk correctly.
You can find a detailled explanation of this environment here: https://hackmd.io/@jeffreymo/SJJrSJh5_#PyBullet
"""

env_id = "AntBulletEnv-v0"
# Create the env
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

"""The observation Space (from [Jeffrey Y Mo](https://hackmd.io/@jeffreymo/SJJrSJh5_#PyBullet)):

The difference is that our observation space is 28 not 29.

<img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit8/obs_space.png" alt="PyBullet Ant Obs space"/>

"""

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action"

env = make_vec_env(env_id, n_envs=4)

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

"""#### Solution"""

model = A2C(policy = "MlpPolicy",
            env = env,
            gae_lambda = 0.9,
            gamma = 0.99,
            learning_rate = 0.00096,
            max_grad_norm = 0.5,
            n_steps = 8,
            vf_coef = 0.4,
            ent_coef = 0.0,
            policy_kwargs=dict(
            log_std_init=-2, ortho_init=False),
            normalize_advantage=False,
            use_rms_prop= True,
            use_sde= True,
            verbose=1)

"""### Train the A2C agent 🏃
- Let's train our agent for 2,000,000 timesteps, don't forget to use GPU on Colab. It will take approximately ~25-40min
"""

model.learn(200)

# Save the model and  VecNormalize statistics when saving the agent
model.save("a2c-AntBulletEnv-v0")
env.save("vec_normalize.pkl")

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make("AntBulletEnv-v0")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# Load the agent
model = A2C.load("a2c-AntBulletEnv-v0")

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")