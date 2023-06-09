import numpy as np
import gym
import matplotlib.pyplot as plt
# Import and initialize Mountain Car Environment
def get_greedy_action(Q, state_adj):
     return np.argmax(Q[state_adj[0], state_adj[1]])

def simulate(Q):
     env = gym.make('MountainCar-v0', render_mode = "human")
     state = env.reset()
     state = state[0]
     done = False
     i=0
     while not done:
          print(state, env.observation_space.low)
          discretized_state = (state - env.observation_space.low)*np.array([10, 100])
          discretized_state = np.round(discretized_state, 0).astype(int)
          action = np.argmax(Q[discretized_state[0], discretized_state[1]])
          observation, reward, terminated, truncated, _ = env.step(action.item())
          state = observation
          print("iter", i, state)
          #state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
          done = terminated or truncated
          i+=1


env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
     # Determine size of discretized state space
     num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
     num_states = np.round(num_states, 0).astype(int) + 1

     # Initialize Q table
     Q = np.random.uniform(low = -1, high = 1,
     size = (num_states[0], num_states[1],
     env.action_space.n))

     # Initialize variables to track rewards
     reward_list = []
     ave_reward_list = []

     # Calculate episodic reduction in epsilon
     reduction = (epsilon - min_eps)/episodes

     # Run Q learning algorithm
     for i in range(episodes):
          print("Episode", i)
          # Initialize parameters
          done = False
          tot_reward, reward = 0,0
          state = env.reset()

          # Discretize state
          state_adj = (state[0] - env.observation_space.low)*np.array([10, 100])
          state_adj = np.round(state_adj, 0).astype(int)

          buffer = []

          while done != True:
               # Render environment for last five episodes
               if i >= (episodes - 20):
                    env.render()

               # Determine next action - epsilon greedy strategy
               if np.random.random() < 1 - epsilon:
                    action = np.argmax(Q[state_adj[0], state_adj[1]])
               else:
                    action = np.random.randint(0, env.action_space.n)

               # Get next state and reward
               state2, reward, done, truncated, info = env.step(action)
               # print("State", state2)
               # print("Reward", reward)
               # Discretize state2
               state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
               state2_adj = np.round(state2_adj, 0).astype(int)
               # print("Discrete state", state2_adj)

               #Allow for terminal states
               if done and state2[0] >= 0.5:
                    Q[state_adj[0], state_adj[1], action] = reward

               # Adjust Q value for current state
               else:
                    delta = learning*(reward + discount*np.max(Q[state2_adj[0], state2_adj[1]]) - Q[state_adj[0], state_adj[1],action])
                    Q[state_adj[0], state_adj[1],action] += delta

               # Update variables
               tot_reward += reward
               state_adj = state2_adj

          # Decay epsilon
          if epsilon > min_eps:
               epsilon -= reduction

          # Track rewards
          reward_list.append(tot_reward)

          if (i+1) % 100 == 0:
               ave_reward = np.mean(reward_list)
               ave_reward_list.append(ave_reward)
               reward_list = []

          if (i+1) % 100 == 0:
               print('Episode {} Average Reward: {}'.format(i+1, ave_reward))

     env.close()

     return ave_reward_list, Q

# Run Q-learning algorithm
rewards, q = QLearning(env, 0.2, 0.9, 0.8, 0, 1500)
# Plot Rewards
print("rewards", rewards)
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')
plt.show()

simulate(q)