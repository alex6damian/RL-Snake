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
# NEW COLORS for enhanced graphics
GREEN = (124, 252, 0)  # Light grass green
DARK_RED = (150, 0, 0)  # Darker red for apple shading
BROWN = (101, 67, 33)  # Brown for apple stem
LEAF_GREEN = (34, 139, 34)  # Green for apple leaf
SNAKE_GREEN = (34, 139, 34)  # Main snake body color
SNAKE_DARK_GREEN = (0, 100, 0)  # Darker green for outline
SNAKE_LIGHT_GREEN = (144, 238, 144)  # Lighter green for belly/highlights

# game constants
BLOCK_SIZE = 20
SPEED = 11

class SnakeGame:
    """
    the snake game environment for reinforcement learning agent
    supports training without ui for faster performance
    """
    def __init__(self, w=800, h=800, render=False):
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
        num_cells_x = self.w // BLOCK_SIZE
        num_cells_y = self.h // BLOCK_SIZE  

        x = random.randint(0, num_cells_x - 1) * BLOCK_SIZE
        y = random.randint(0, num_cells_y - 1) * BLOCK_SIZE

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

    def _draw_apple(self, x, y):
        """
        Draws an apple at the given position.
        """
        center_x = x + BLOCK_SIZE // 2
        center_y = y + BLOCK_SIZE // 2
        
        # Draw the main apple body (circle)
        pygame.draw.circle(self.display, RED, (center_x, center_y), BLOCK_SIZE // 2 - 2)
        
        # Add a highlight for 3D effect (small lighter circle on top-left)
        pygame.draw.circle(self.display, (255, 100, 100), (center_x - 3, center_y - 3), 3)
        
        # Add darker shading on bottom-right
        pygame.draw.circle(self.display, DARK_RED, (center_x + 2, center_y + 3), 4)
        
        # Draw stem (small brown rectangle at the top)
        stem_rect = pygame.Rect(center_x - 1, y + 2, 2, 4)
        pygame.draw.rect(self.display, BROWN, stem_rect)
        
        # Draw a small leaf
        leaf_points = [
            (center_x + 2, y + 3),
            (center_x + 6, y + 2),
            (center_x + 4, y + 5)
        ]
        pygame.draw.polygon(self.display, LEAF_GREEN, leaf_points)

    def _draw_snake(self):
        """
        Draws the snake as a continuous linear shape with smooth connections.
        """
        if len(self.snake) == 0:
            return
        
        # Draw body segments as connected rectangles
        for i, pt in enumerate(self.snake):
            # Draw the main body rectangle with rounded edges
            rect = pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
            pygame.draw.rect(self.display, SNAKE_GREEN, rect, border_radius=5)
            
            # Add darker outline
            pygame.draw.rect(self.display, SNAKE_DARK_GREEN, rect, width=2, border_radius=5)
            
            # Add lighter center stripe for belly
            if i > 0:  # Not the head
                center_rect = pygame.Rect(pt.x + 6, pt.y + 6, BLOCK_SIZE - 12, BLOCK_SIZE - 12)
                pygame.draw.rect(self.display, SNAKE_LIGHT_GREEN, center_rect, border_radius=3)
            
            # Connect segments smoothly
            if i < len(self.snake) - 1:
                next_pt = self.snake[i + 1]
                # Draw connecting piece between segments
                if pt.x == next_pt.x:  # Vertical connection
                    if pt.y < next_pt.y:
                        connect_rect = pygame.Rect(pt.x + 2, pt.y + BLOCK_SIZE - 4, BLOCK_SIZE - 4, 6)
                    else:
                        connect_rect = pygame.Rect(pt.x + 2, pt.y - 2, BLOCK_SIZE - 4, 6)
                else:  # Horizontal connection
                    if pt.x < next_pt.x:
                        connect_rect = pygame.Rect(pt.x + BLOCK_SIZE - 4, pt.y + 2, 6, BLOCK_SIZE - 4)
                    else:
                        connect_rect = pygame.Rect(pt.x - 2, pt.y + 2, 6, BLOCK_SIZE - 4)
                pygame.draw.rect(self.display, SNAKE_GREEN, connect_rect)
        
        # Draw the head with eyes
        head = self.snake[0]
        head_center_x = head.x + BLOCK_SIZE // 2
        head_center_y = head.y + BLOCK_SIZE // 2
        
        # Draw eyes based on direction
        if self.direction == Direction.RIGHT:
            eye1_pos = (head_center_x + 3, head_center_y - 4)
            eye2_pos = (head_center_x + 3, head_center_y + 4)
        elif self.direction == Direction.LEFT:
            eye1_pos = (head_center_x - 3, head_center_y - 4)
            eye2_pos = (head_center_x - 3, head_center_y + 4)
        elif self.direction == Direction.UP:
            eye1_pos = (head_center_x - 4, head_center_y - 3)
            eye2_pos = (head_center_x + 4, head_center_y - 3)
        else:  # DOWN
            eye1_pos = (head_center_x - 4, head_center_y + 3)
            eye2_pos = (head_center_x + 4, head_center_y + 3)
        
        # Draw white of eyes
        pygame.draw.circle(self.display, WHITE, eye1_pos, 3)
        pygame.draw.circle(self.display, WHITE, eye2_pos, 3)
        
        # Draw pupils
        pygame.draw.circle(self.display, BLACK, eye1_pos, 1)
        pygame.draw.circle(self.display, BLACK, eye2_pos, 1)

    def _update_ui(self):
        """
        updates the game display with the new state
        runs only if render=true
        """
        if not self.render:
            return
        
        # Draw grass background
        self.display.fill(GREEN)
        
        # Draw snake with new graphics
        self._draw_snake()

        # Draw apple with new graphics
        self._draw_apple(self.food.x, self.food.y)
        
        # Draw score
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