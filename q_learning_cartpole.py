import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
env.reset()

high = env.observation_space.high
low = env.observation_space.low
max_vel = 3
max_ang_vel = 5
high[1] = max_vel
high[3] = max_ang_vel
low[1] = -max_vel
low[3] = -max_ang_vel
print("observation space max", high)
intervals=[10, 50, 100, 50]

def discretize_state(state):
     state[1] = low[1] if state[1] < low[1] else state[1]
     state[1] = high[1] if state[1] > high[1] else state[1]
     state[3] = low[3] if state[3] < low[3] else state[3]
     state[3] = high[3] if state[3] > high[3] else state[3]
     discretized_state = (state - low)*np.array(intervals)
     discretized_state = np.round(discretized_state, 0).astype(int)
     return discretized_state

     
# Import and initialize Mountain Car Environment
def get_greedy_action(Q, state_adj):
     return np.argmax(Q[state_adj[0], state_adj[1]])

def simulate(Q):
     env = gym.make('CartPole-v1', render_mode = "human")
     state = env.reset()
     state = state[0]
     done = False
     i=0
     while not done:
          print(state, low)
          discretized_state = discretize_state(state)
          action = np.argmax(Q[discretized_state[0], discretized_state[1], discretized_state[2], discretized_state[3]])
          observation, reward, terminated, truncated, _ = env.step(action.item())
          state = observation
          print("iter", i, state)
          #state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
          done = terminated or truncated
          i+=1

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
     # Determine size of discretized state space
     num_states = (high - low)*np.array(intervals)
     print("number of states before", num_states)
     num_states = np.round(num_states, 0).astype(int) + 1
     print("number of states", num_states)

     # Initialize Q table
     Q = np.random.uniform(low = -1, high = 1,
     size = (num_states[0], num_states[1],num_states[2], num_states[3],
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
          # print("State", state[0], low)
          state_adj = (state[0] - low)*np.array(intervals)
          state_adj = np.round(state_adj, 0).astype(int)

          while done != True:
               # Render environment for last five episodes
               if i >= (episodes - 20):
                    env.render()

               # Determine next action - epsilon greedy strategy
               if np.random.random() < 1 - epsilon:
                    action = np.argmax(Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3]])
               else:
                    action = np.random.randint(0, env.action_space.n)

               # Get next state and reward
               state2, reward, done, truncated, info = env.step(action)
               # print("State", state2)
               # print("Reward", reward)
               # Discretize state2
               # state2_adj = (state2 - low)*np.array([10, 50, 10, 50])
               state2_adj = discretize_state(state2)
               # print("Discrete state", state2_adj)

               #Allow for terminal states
               if done and state2[0] >= 0.5:
                    Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], action] = reward

               # Adjust Q value for current state
               else:
                    delta = learning*(reward + discount*np.max(Q[state2_adj[0], state2_adj[1], state2_adj[2], state2_adj[3]]) - Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3],action])
                    Q[state_adj[0], state_adj[1],state_adj[2], state_adj[3],action] += delta

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
rewards, q = QLearning(env, 0.2, 0.9, 0.8, 0, 100000)
# Plot Rewards
print("rewards", rewards)
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.jpg')
plt.show()

simulate(q)