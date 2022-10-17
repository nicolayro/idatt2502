import gym
import numpy as np

# Create environment 
print("--- CardPole-v1 ---") 
# Includes render_mode='human' for vizualization
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")

# Action Space:
#   0: push left
#   1: push right
print("action_space:", env.action_space)

# Observation Space:
#   0: cart position (-4.8, 4.8)
#   1: cart velocity (-inf, inf)
#   2: pole angle (rad)(~ -0.418, ~ 0.418)
#   3: pole angular velocity (-inf, inf)
print("observation_space:", env.observation_space.shape)

# Reward: Since the goal of the game is to stay alive as long as
# possible, +1 is given for each step taken

# Starting state: All observations are given a uniformly
# random value between (-0.05, 0.05)
# Initialize environment
observation, info = env.reset()
print("initial state:", observation, info)

# Episode end: 
#   1: pole angle is greater than 12 deg 
#   2: cart position is greater than 2.4 (means the cart reaches end of display)
#   3: episode length is greater than 500

class QLearn:
    alpha = 0.1 # learning rate
    gamma = 0.9 # discount factor
    epsilon = 0.1 # exploration rate

    buckets = []

    def __init__(self, observation_space, action_space, num_buckets = 1):
        # print(f"observations: {observation_space}")
        # print(f"actions: {action_space}")
        # high = observation_space.high
        # low = observation_space.low
        # obs_length = observation_space.shape[0]

        # for i in range(obs_length):
        #     self.buckets.append(np.linspace(low[i], high[i], num_buckets))

        self.buckets = [
		np.linspace(-4.8, 4.8, num_buckets),
		np.linspace(-4, 4, num_buckets),
		np.linspace(-.418, .418, num_buckets),
		np.linspace(-4, 4, num_buckets)
	    ]
        
        self.Q = np.zeros([num_buckets] * observation_space.shape[0] + [action_space.n])

    def learn(self, action, state, reward, new_state):
        state = self.quantize(state)
        new_state = self.quantize(new_state)
        self.Q[state][action] = self.bellmann(action, state, reward, new_state)


    def bellmann(self, action, state, reward, new_state):
        return (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]))

    # Some problems have a continuous observation_space. These problems
    # should be quantized (turn continuous values into descrete ones)
    def quantize(self, state):
        result = []
        for idx, bucket in enumerate(self.buckets):
            result.append(np.digitize(state[idx], bucket) - 1)
        return tuple(result)

    def policy(self, state):
        # Randoms 
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # Quantize state and fetch best guess
        state = self.Q[self.quantize(state)]
        return np.argmax(state)


model = QLearn(env.observation_space, env.action_space, 20)

result, best_result = 0, 0

for i in range(10000):
    best_result = max(result, best_result)

    result = 0
    observation, info = env.reset() 
    while True:
        action = model.policy(observation)

        old_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        model.learn(action, old_observation, reward, observation)

        result += reward

        if terminated or truncated:
            break


print(f"best result: {best_result}")
env.render()
env.close()
