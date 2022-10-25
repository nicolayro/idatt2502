import numpy as np
import pygame

# Implementation of a Gridworld environment
# Gridworld consist of a grid that the agent can move around
# The action space of gridworld consist of four actions
#   0: move left
#   1: move up
#   2: move right
#   3: move down
# The observatino space of gridworld consist of two values:
#   0: x position
#   1: y position
class Gridworld:
 
    def __init__(self, x, y):
        self.observation_space = np.array([x, y])
        self.action_space = np.array([0, 1, 2, 3]) 
        self.agent_position = [0, 0]
        self.reward_position = [x - 1, y-1]

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        self.move_agent(action)

        if self.agent_position == self.reward_position:
            terminated = True
            reward = 1

        # return result
        return self.agent_position, reward, terminated, truncated


    def move_agent(self, action):
        # convert action to movement
        x, y = self.action_to_movement(action)

        # Check that move is valid
        new_pos_x = self.agent_position[0] + x
        new_pos_y = self.agent_position[1] + y
        if new_pos_x < 0 or new_pos_x > self.observation_space[0] - 1:
            return 
        if new_pos_y < 0 or new_pos_y > self.observation_space[1] - 1:
            return 
    
        # Actually move agent
        self.agent_position[0] = new_pos_x
        self.agent_position[1] = new_pos_y


    def action_to_movement(self, action):
        if action == 0:
            return [-1, 0]
        if action == 1:
            return [0, -1]
        if action == 2:
            return [1, 0]
        if action == 3:
            return [0, 1]

        print(f"invalid action: {action}")
        return [0, 0]

    def reset(self, random = False):
        # Move agent to start position
        if not random: 
            self.agent_position = [0, 0]
            return self.agent_position

        x = np.random.randint(0, self.observation_space[0])
        y = np.random.randint(0, self.observation_space[1])
        self.agent_position = [x, y]
        return self.agent_position


    def sample(self):
        random_action = np.random.randint(0, 4)
        return self.action_space[random_action]
     
class QLearn:

    def __init__(self, observation_space, action_space, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros([observation_space[0]] + [observation_space[1]] + [action_space.shape[0]])
        print(self.Q)
        self.Q = np.random.random(size=([observation_space[0]] + [observation_space[1]] + [action_space.shape[0]]))
        print(self.Q)

    def learn(self, action, state, reward, new_state):
        self.Q[state[0], state[1]][action] = self.bellmann(action, state, reward, new_state)

    def bellmann(self, action, state, reward, new_state):
        return (1 - self.alpha) * self.Q[state[0], state[1]][action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state[0], new_state[1]]))

    def policy(self, state):
        return np.argmax(self.Q[state[0], state[1]])



alpha = 0.05 # learning rate
gamma = 0.95 # discount factor
epsilon = 0.2 # Exploration rate

env = Gridworld(5, 5)
model = QLearn(env.observation_space, env.action_space, alpha, gamma)

observation = env.reset()

episodes = 100000
for i in range(episodes):
    observation = env.reset(True)

    if i % int(episodes/10) == 0:
        print(i)

    while True:
        # env.render()
        action = model.policy(observation)

        # Some moves are random
        if np.random.uniform(0, 1) <= epsilon:
            action = env.sample()

        old_observation = observation.copy()
        observation, reward, terminated, truncated = env.step(action)

        model.learn(action, old_observation, reward, observation)

        if terminated or truncated:
            break


# =============================================================================
# Pygame 
# =============================================================================

# Constants 
WHITE = (255, 255, 255)
BLACK = (0,0,0)
RED = (200,0,0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
BROWN  = (0, 255, 255)

WIDTH = 640
HEIGHT = 480
TILE_SIZE = 60

def update_pos(screen, pos):
    # update square
    rect = pygame.Rect(pos[0] * TILE_SIZE, pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(screen, RED, rect)

    # write debugging info
    font = pygame.font.SysFont("None", 24)
    text_pos = font.render(f"pos {pos}", True, BLACK)
    screen.blit(text_pos, (WIDTH - 2 * TILE_SIZE, TILE_SIZE))

    pygame.display.update()

def draw_grid(screen, q_table):
    screen.fill(WHITE)
    for x in range(env.observation_space[0]):
        for y in range(env.observation_space[1]):
            cell = pygame.Surface([TILE_SIZE, TILE_SIZE])
            cell.fill(WHITE)
            rect = pygame.Rect(0, 0, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(cell, BLACK, rect, 1)

            # Draw and orient arrow
            pygame.draw.polygon(cell, BLACK, ((0, 10), (0, 20), (20, 20), (20, 30), (30, 15), (20, 0), (20, 10)))
            oriented_cell = pygame.transform.rotate(cell, float(np.argmax(q_table[x, y])) * -90 + 180)
            screen.blit(oriented_cell, (x * TILE_SIZE, y * TILE_SIZE))

    # Draw termination state (goal)
    rect = pygame.Rect(env.reward_position[0] * TILE_SIZE , env.reward_position[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(screen, BLUE, rect)


# setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gridworld")

for i in range(20):
    observation = env.reset(True)

    while True:
        # Visuals
        pygame.time.delay(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        draw_grid(screen, model.Q)
        update_pos(screen, observation)

        # Actual model
        action = model.policy(observation)
        observation, reward, terminated, truncated = env.step(action)

        if terminated or truncated:
            break


print(model.Q)
pygame.quit()

