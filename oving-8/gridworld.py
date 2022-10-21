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
    # Pygame
    WHITE = (255, 255, 255)
    BLACK = (0,0,0)
    RED = (200,0,0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    PURPLE = (255, 0, 255)
    BROWN  = (0, 255, 255)
    TILE_SIZE = 40 


    def __init__(self, x, y):
        self.observation_space = np.array([x, y])
        self.action_space = np.array([0, 1, 2, 3]) 
        self.agent_position = [0, 0]
        self.reward_position = [x - 1, y-1]

        self.init_pygame()

   
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

    def reset(self):
        # Move agent to start position
        self.agent_position = [0, 0]
        return self.agent_position

    def sample(self):
        random_action = np.random.randint(0, 4)
        return self.action_space[random_action]

    def init_pygame(self):
        # pygame
        pygame.init()
        w = 640
        h = 480
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Gridworld")

    def fill_board(self):
        self.screen.fill(self.WHITE)
        for x in range(self.observation_space[0]):
            for y in range(self.observation_space[1]):
                image = pygame.Surface([self.TILE_SIZE, self.TILE_SIZE])
                image.fill(self.WHITE)
                rect = pygame.Rect(0, 0, self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(image, env.BLACK, rect, 1)
                self.screen.blit(image, (x * self.TILE_SIZE, y * self.TILE_SIZE))

        rect = pygame.Rect(self.reward_position[0] * self.TILE_SIZE , self.reward_position[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.BLUE, rect)

  
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.fill_board()

        # update square
        rect = pygame.Rect(self.agent_position[0], self.agent_position[1] * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.RED, rect)

        # write debugging info
        font = pygame.font.SysFont(None, 24)
        pos = font.render(f"pos {self.agent_position}", True, self.BLACK)
        self.screen.blit(pos, (400, 100))
        
        pygame.display.update()

    
class QLearn:
    alpha = 0.2 # learning rate
    gamma = 0.95 # discount factor

    def __init__(self, observation_space, action_space):
        self.Q = np.zeros([observation_space[0]] + [observation_space[1]] + [action_space.shape[0]])

    def learn(self, action, state, reward, new_state):
        self.Q[state[0], state[1]][action] = self.bellmann(action, state, reward, new_state)

    def bellmann(self, action, state, reward, new_state):
        return (1 - self.alpha) * self.Q[state[0], state[1]][action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state[0], new_state[1]]))

    def policy(self, state):
        return np.argmax(self.Q[state[0], state[1]])

env = Gridworld(2, 3)
model = QLearn(env.observation_space, env.action_space)
print(model.Q.shape)

observation = env.reset()
epsilon = 0.1 # Exploration rate

for i in range(100):
    observation = env.reset()

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



for i in range(100):
    observation = env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        env.screen.fill(env.WHITE)
        for x in range(env.observation_space[0]):
            for y in range(env.observation_space[1]):
                image = pygame.Surface([env.TILE_SIZE, env.TILE_SIZE])
                image.fill(env.WHITE)
                rect = pygame.Rect(0, 0, env.TILE_SIZE, env.TILE_SIZE)
                pygame.draw.rect(image, env.BLACK, rect, 1)

                pygame.draw.polygon(image, env.BLACK, ((0, 10), (0, 20), (20, 20), (20, 30), (30, 15), (20, 0), (20, 10)))
                pygame.transform.rotate(image, float(model.policy(observation) * 90))

                env.screen.blit(image, (x * env.TILE_SIZE, y * env.TILE_SIZE))

        rect = pygame.Rect(env.reward_position[0] * env.TILE_SIZE , env.reward_position[1] * env.TILE_SIZE, env.TILE_SIZE, env.TILE_SIZE)
        pygame.draw.rect(env.screen, env.BLUE, rect)

        action = model.policy(observation)

        old_observation = observation.copy()
        observation, reward, terminated, truncated = env.step(action)

        model.learn(action, old_observation, reward, observation)

        pygame.time.delay(100)
        if terminated or truncated:
            break

print(model.Q)
pygame.quit()

