import gym
import numpy as np

# Create environment print("--- CartPole-v1 ---") Includes render_mode='human' for vizualization
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")

print("\nDATA:")
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
print("initial_state:", observation, info)

# Episode end: 
#   1: pole angle is greater than 12 deg 
#   2: cart position is greater than 2.4 (means the cart reaches end of display)
#   3: episode length is greater than 500

class QLearn:
    alpha = 0.2 # learning rate
    gamma = 0.95 # discount factor
    epsilon = 0.1 # exploration rate

    def __init__(self, observation_space, action_space, buckets):
        self.buckets = buckets
        self.Q = np.zeros([len(buckets[0])] * observation_space.shape[0] + [action_space.n])

    def learn(self, action, state, reward, new_state):
        state = self.quantize(state)
        new_state = self.quantize(new_state)
        print(f"state: {state}")
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


print("\nPROCESSING:")
num_buckets = 20
buckets = [
     	np.linspace(-4.8, 4.8, num_buckets),
		np.linspace(-4, 4,num_buckets),
 		np.linspace(-.418, .418, num_buckets),
        np.linspace(-4, 4, num_buckets)
	]
model = QLearn(env.observation_space, env.action_space, buckets)

result, best_result, success = 0, 0, 0
best_q = np.empty(0)
for i in range(10000):
    # Print results so far
    if i % 1000 == 999:
        print(f"{i + 1} tries: best_result: {best_result}, number of successful runs: {success}")
        success = 0

    # Track best result
    best_result = max(result, best_result)
    result = 0

    observation, info = env.reset()
    while True:
        action = model.policy(observation)
        old_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        model.learn(action, old_observation, reward, observation)

        result += reward
       
        if info:
            print(f"info: {info}")

        if terminated:
            break

        if truncated:
            best_q = model.Q
            success += 1
            break

env.close()

print("\nFINISHED MODEL:")
# Display best Q in action
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
lives = 10
score = 0
while True:
    action = np.argmax(best_q[model.quantize(observation)])
    observation, reward, terminated, truncated, info = env.step(action)

    score += reward

    print(f'\rscore: {score}\tlives: {lives} ', end='')

    if info:
        print(f"info: {info}")

    if terminated:
        score = 0
        lives -= 1
        observation, info = env.reset()

    if lives < 1:
        break

print(f'\rscore: {score}\tlives: {lives} ', end='')
if (lives < 1):
    print("\nThe model lost...")
else:
    print("\nThe model succeded!")
