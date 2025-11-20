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
GRAY = (100, 100, 100)  # Color for walls
GREEN = (124, 252, 0)  # Light grass green
DARK_RED = (150, 0, 0)  # Darker red for apple shading
BROWN = (101, 67, 33)  # Brown for apple stem
LEAF_GREEN = (34, 139, 34)  # Green for apple leaf
SNAKE_GREEN = (34, 139, 34)  # Main snake body color
SNAKE_DARK_GREEN = (0, 100, 0)  # Darker green for outline
SNAKE_LIGHT_GREEN = (144, 238, 144)  # Lighter green for belly/highlights

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
        # Initialize inner walls
        self._init_inner_walls()
        # reset the game to start
        self.reset()

    def _init_inner_walls(self):
        """
        Creates inner wall obstacles on the game board.
        """
        self.inner_walls = []
        
        # Horizontal wall in the middle-top area
        for x in range(7 * BLOCK_SIZE, 12 * BLOCK_SIZE, BLOCK_SIZE):
            self.inner_walls.append(Point(x, 6 * BLOCK_SIZE))
        
        # Vertical wall on the left side
        for y in range(6 * BLOCK_SIZE, 13 * BLOCK_SIZE, BLOCK_SIZE):
            self.inner_walls.append(Point(8 * BLOCK_SIZE, y))
        
        # Horizontal wall in the middle-bottom area
        for x in range(16 * BLOCK_SIZE, 20 * BLOCK_SIZE, BLOCK_SIZE):
            self.inner_walls.append(Point(x, 18 * BLOCK_SIZE))
        
        # Vertical wall on the right side
        for y in range(6 * BLOCK_SIZE, 10 * BLOCK_SIZE, BLOCK_SIZE):
            self.inner_walls.append(Point(24 * BLOCK_SIZE, y))

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
        places food randomly on the grid, making sure it's not inside the snake or on walls.
        """
        # Only place food in the inner area (not on walls)
        x = random.randint(1, (self.w - BLOCK_SIZE) // BLOCK_SIZE - 2) * BLOCK_SIZE + BLOCK_SIZE
        y = random.randint(1, (self.h - BLOCK_SIZE) // BLOCK_SIZE - 2) * BLOCK_SIZE + BLOCK_SIZE
        self.food = Point(x, y)
        # Make sure food is not on snake or inner walls
        if self.food in self.snake or self.food in self.inner_walls:
            self._place_food() # if food is on the snake or wall, try again

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
        checks if a given point collides with the boundaries, inner walls, or the snake itself.
        """
        if pt is None:
            pt = self.head
        # check if it hits outer wall
        if pt.x > self.w - 2 * BLOCK_SIZE or pt.x < BLOCK_SIZE or pt.y > self.h - 2 * BLOCK_SIZE or pt.y < BLOCK_SIZE:
            return True
        # check if it hits inner walls
        if pt in self.inner_walls:
            return True
        # check if it hits its own body
        if pt in self.snake[1:]:
            return True
        return False

    def _draw_grass_background(self):
        """
        draws a solid light green grass background.
        """
        self.display.fill(GREEN)

    def _draw_walls(self):
        """
        draws visible walls around the board perimeter.
        """
        # Top wall
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.rect(self.display, GRAY, pygame.Rect(x, 0, BLOCK_SIZE, BLOCK_SIZE))
        
        # Bottom wall
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.rect(self.display, GRAY, pygame.Rect(x, self.h - BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        # Left wall
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.rect(self.display, GRAY, pygame.Rect(0, y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Right wall
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.rect(self.display, GRAY, pygame.Rect(self.w - BLOCK_SIZE, y, BLOCK_SIZE, BLOCK_SIZE))

    def _draw_inner_walls(self):
        """
        draws the inner wall obstacles.
        """
        for wall in self.inner_walls:
            pygame.draw.rect(self.display, GRAY, pygame.Rect(wall.x, wall.y, BLOCK_SIZE, BLOCK_SIZE))

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
        updates the game display with the new state.
        """
        # Draw grass background first
        self._draw_grass_background()
        
        # Draw walls on top of grass
        self._draw_walls()
        
        # Draw inner walls
        self._draw_inner_walls()
        
        # Draw snake as a smooth linear shape
        self._draw_snake()

        # Draw apple
        self._draw_apple(self.food.x, self.food.y)
        
        # Draw score
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

        # if game is over, reset it (this includes wall collisions)
        if done:
            game.reset()
    
    # quit pygame after the loop ends
    pygame.quit()