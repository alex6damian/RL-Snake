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
SPEED = 5

class SnakeGame:
    """
    the snake game environment for reinforcement learning agent
    supports training without ui for faster performance
    """
    def __init__(self, w=320, h=240, render=False):
        """
        initializes the game window, clock, and state
        
        args:
            w: width of the game window
            h: height of the game window
            render: if true, displays game; if false, runs without ui (faster)
        """
        self.w = w
        self.h = h
        self.render = render
        
        # init display only if render=true
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('rl snake')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None
        
        # reset the game to start
        self.reset()

    def reset(self):
        """
        resets the game to the initial state
        """
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      #Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
                      ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        """
        places food randomly on the grid
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def step(self, action):
        """
        performs one step in the game based on the agent's action
        with improved reward shaping
        
        returns: (reward, game_over, score)
        """
        self.frame_iteration += 1
        old_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        done = False
        
        # collision - large penalty
        if self.is_collision():
            done = True
            reward = -10
            return reward, done, self.score
        
        # timeout - prevents spinning in circles
        if self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return reward, done, self.score
        
        # ate food - large reward
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.frame_iteration = 0  # reset timeout after eating
        else:
            self.snake.pop()
            
            # reward for getting closer to food (subtle)
            new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            
            if new_distance < old_distance:
                reward = 0.1  # small reward for gradual approach
            else:
                reward = -0.1  # small penalty for moving away
        
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)
        
        return reward, done, self.score

    def is_collision(self, pt=None):
        """
        checks if a given point collides with boundaries or snake itself
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
        updates the game display with the new state
        runs only if render=true
        """
        if not self.render:
            return
            
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
        determines the new direction based on the relative action
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.direction
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

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


# manual testing (with ui)
if __name__ == '__main__':
    game = SnakeGame(render=True)  # with ui for manual testing

    running = True
    while running:
        action = [1, 0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = [0, 0, 1]
                elif event.key == pygame.K_RIGHT:
                    action = [0, 1, 0]
        
        reward, done, score = game.step(action)

        if done:
            game.reset()
    
    pygame.quit()