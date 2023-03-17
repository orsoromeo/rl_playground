import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


import gym

# We create our environment with gym.make("<name_of_the_environment>")
env_name="CartPole-v1"
env = gym.make(env_name)
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action

# Create the environment
env = make_vec_env(env_name, n_envs=16)

# SOLUTION
# We added some parameters to accelerate the training
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

# """## Train the PPO agent 🏃
model.learn(total_timesteps=1000)
# Save the model
model_name = "ppo-"+env_name
model.save(model_name)

# """## Evaluate the agent 📈
eval_env = gym.make(env_name)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

env = gym.make(env_name, render_mode ="human")
env.reset()
done=False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated