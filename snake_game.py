import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# initialize pygame modules
pygame.init()
font = pygame.font.Font(None, 25)

# define directions using an enum for clarity
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# a simple named tuple to represent a point (x, y)
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# game constants
BLOCK_SIZE = 20
SPEED = 20 # you can adjust this for training vs. human playing

class SnakeGame:
    """
    the snake game environment for our reinforcement learning agent.
    """
    def __init__(self, w=640, h=480):
        """
        initializes the game window, clock, and state.
        """
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('rl snake')
        self.clock = pygame.time.Clock()
        # reset the game to start
        self.reset()

    def reset(self):
        """
        resets the game to the initial state. called at the beginning of each new episode.
        """
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0 # keeps track of how long the game has been running

    def _place_food(self):
        """
        places food randomly on the grid, making sure it's not inside the snake.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food() # if food is on the snake, try again

    def step(self, action):
        """
        performs one step in the game based on the agent's action.
        this is the most important function for the rl agent.

        returns: (reward, game_over, score)
        """
        self.frame_iteration += 1

        # 1. move the snake based on the action
        self._move(action)
        self.snake.insert(0, self.head)

        # 2. check if the game is over
        reward = 0
        done = False
        # game ends if there's a collision or if the snake doesn't find food for too long
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10  # give a negative reward for dying
            return reward, done, self.score

        # 3. check if the snake ate the food
        if self.head == self.food:
            self.score += 1
            reward = 10  # give a positive reward for eating food
            self._place_food()
        else:
            self.snake.pop() # if no food is eaten, remove the tail

        # 4. update the ui and tick the clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 5. return reward, game over status, and score
        return reward, done, self.score

    def is_collision(self, pt=None):
        """
        checks if a given point collides with the boundaries or the snake itself.
        """
        if pt is None:
            pt = self.head
        # check if it hits a wall
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # check if it hits its own body
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """
        updates the game display with the new state.
        """
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        determines the new direction based on the relative action from the agent.
        action is one-hot encoded: [straight, right, left].
        """
        # [1,0,0] -> go straight
        # [0,1,0] -> turn right
        # [0,0,1] -> turn left

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.direction  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

# this block is for testing the game manually
if __name__ == '__main__':
    game = SnakeGame()

    # main game loop
    running = True
    while running:
        # default action is to go straight
        action = [1, 0, 0]

        # handle all events only once per loop
        for event in pygame.event.get():
            # handle window close
            if event.type == pygame.QUIT:
                running = False
            
            # handle keyboard input for manual play
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = [0, 0, 1]  # relative left turn
                elif event.key == pygame.K_RIGHT:
                    action = [0, 1, 0]  # relative right turn
        
        # perform a game step with the chosen action
        reward, done, score = game.step(action)

        # if game is over, reset it
        if done:
            game.reset()
    
    # quit pygame after the loop ends
    pygame.quit()