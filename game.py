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
GREEN = (124, 252, 0)  # Light grass green
DARK_RED = (150, 0, 0)  # Darker red for apple shading
BROWN = (101, 67, 33)  # Brown for apple stem
LEAF_GREEN = (34, 139, 34)  # Green for apple leaf
SNAKE_GREEN = (34, 139, 34)  # Main snake body color
SNAKE_DARK_GREEN = (0, 100, 0)  # Darker green for outline
SNAKE_LIGHT_GREEN = (144, 238, 144)  # Lighter green for belly/highlights
GRAY = (128, 128, 128) # CULOARE OBSTACOLE

# game constants
BLOCK_SIZE = 20
SPEED = 20 # Viteza pentru modul manual (poți mări la 40)

class SnakeGame:
    """
    Snake game environment with FIXED OBSTACLES.
    """
    def __init__(self, w=800, h=800, render=False):
        self.w = w
        self.h = h
        self.render = render
        
        # init display only if render=true
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('RL Snake - Fixed Map')
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
                      ]
        self.score = 0
        self.food = None
        
        # --- GENERARE OBSTACOLE FIXE ---
        self.obstacles = []
        self._place_fixed_obstacles()
        # -------------------------------
        
        self._place_food()
        self.frame_iteration = 0

    def _place_fixed_obstacles(self):
        """
        Plasează obstacole într-un pattern FIX (Standard Benchmark Map).
        Acest lucru asigură o comparație corectă între agenți și permite replay-ului să funcționeze.
        """
        self.obstacles = []
        
        # Calculăm centrul grilei
        cx, cy = (self.w // BLOCK_SIZE) // 2, (self.h // BLOCK_SIZE) // 2
        
        # 1. Piloni în colțuri (offset de 5 blocuri)
        pilar_offset = 5
        pillars = [
            (pilar_offset, pilar_offset),
            (pilar_offset, (self.h // BLOCK_SIZE) - pilar_offset),
            ((self.w // BLOCK_SIZE) - pilar_offset, pilar_offset),
            ((self.w // BLOCK_SIZE) - pilar_offset, (self.h // BLOCK_SIZE) - pilar_offset)
        ]
        
        for px, py in pillars:
            pt = Point(px * BLOCK_SIZE, py * BLOCK_SIZE)
            self.obstacles.append(pt)

        # 2. Zid orizontal central (cu gaură în mijloc)
        gap = 3
        for x in range(cx - 8, cx + 9):
            if abs(x - cx) < gap: continue # Facem gaura
            pt = Point(x * BLOCK_SIZE, cy * BLOCK_SIZE)
            if pt not in self.obstacles:
                self.obstacles.append(pt)
                
        # 3. Zid vertical central (cu gaură în mijloc)
        for y in range(cy - 6, cy + 7):
            if abs(y - cy) < gap: continue
            pt = Point(cx * BLOCK_SIZE, y * BLOCK_SIZE)
            if pt not in self.obstacles:
                self.obstacles.append(pt)
                
        # Siguranță: Eliminăm orice obstacol care s-ar suprapune cu startul șarpelui
        self.obstacles = [pt for pt in self.obstacles if pt not in self.snake]

    def _place_food(self):
        """
        places food randomly on the grid, avoiding snake AND obstacles
        """
        num_cells_x = self.w // BLOCK_SIZE
        num_cells_y = self.h // BLOCK_SIZE  

        x = random.randint(0, num_cells_x - 1) * BLOCK_SIZE
        y = random.randint(0, num_cells_y - 1) * BLOCK_SIZE

        self.food = Point(x, y)
        
        # Verificăm să nu fie pe șarpe SAU pe obstacole
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()

    def step(self, action):
        self.frame_iteration += 1
        old_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        done = False
        
        # COLIZIUNE (Pereți, Sine, OBSTACOLE)
        if self.is_collision():
            done = True
            reward = -10
            return reward, done, self.score
        
        # TIMEOUT
        if self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return reward, done, self.score
        
        # MÂNCARE
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.frame_iteration = 0
        else:
            self.snake.pop()
            
            # Reward shaping (distanță)
            new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            if new_distance < old_distance:
                reward = 0.1
            else:
                reward = -0.1
        
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)
        
        return reward, done, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Lovire pereți
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Lovire propriul corp
        if pt in self.snake[1:]:
            return True
        # Lovire OBSTACOLE
        if pt in self.obstacles:
            return True
            
        return False

    def _draw_apple(self, x, y):
        center_x = x + BLOCK_SIZE // 2
        center_y = y + BLOCK_SIZE // 2
        pygame.draw.circle(self.display, RED, (center_x, center_y), BLOCK_SIZE // 2 - 2)
        pygame.draw.circle(self.display, (255, 100, 100), (center_x - 3, center_y - 3), 3)
        pygame.draw.circle(self.display, DARK_RED, (center_x + 2, center_y + 3), 4)
        stem_rect = pygame.Rect(center_x - 1, y + 2, 2, 4)
        pygame.draw.rect(self.display, BROWN, stem_rect)
        leaf_points = [(center_x + 2, y + 3), (center_x + 6, y + 2), (center_x + 4, y + 5)]
        pygame.draw.polygon(self.display, LEAF_GREEN, leaf_points)

    def _draw_snake(self):
        if len(self.snake) == 0: return
        for i, pt in enumerate(self.snake):
            rect = pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
            pygame.draw.rect(self.display, SNAKE_GREEN, rect, border_radius=5)
            pygame.draw.rect(self.display, SNAKE_DARK_GREEN, rect, width=2, border_radius=5)
            if i > 0:
                center_rect = pygame.Rect(pt.x + 6, pt.y + 6, BLOCK_SIZE - 12, BLOCK_SIZE - 12)
                pygame.draw.rect(self.display, SNAKE_LIGHT_GREEN, center_rect, border_radius=3)
            if i < len(self.snake) - 1:
                next_pt = self.snake[i + 1]
                if pt.x == next_pt.x:
                    if pt.y < next_pt.y: connect_rect = pygame.Rect(pt.x + 2, pt.y + BLOCK_SIZE - 4, BLOCK_SIZE - 4, 6)
                    else: connect_rect = pygame.Rect(pt.x + 2, pt.y - 2, BLOCK_SIZE - 4, 6)
                else:
                    if pt.x < next_pt.x: connect_rect = pygame.Rect(pt.x + BLOCK_SIZE - 4, pt.y + 2, 6, BLOCK_SIZE - 4)
                    else: connect_rect = pygame.Rect(pt.x - 2, pt.y + 2, 6, BLOCK_SIZE - 4)
                pygame.draw.rect(self.display, SNAKE_GREEN, connect_rect)
        
        head = self.snake[0]
        head_center_x = head.x + BLOCK_SIZE // 2
        head_center_y = head.y + BLOCK_SIZE // 2
        if self.direction == Direction.RIGHT: eye1, eye2 = (head_center_x+3, head_center_y-4), (head_center_x+3, head_center_y+4)
        elif self.direction == Direction.LEFT: eye1, eye2 = (head_center_x-3, head_center_y-4), (head_center_x-3, head_center_y+4)
        elif self.direction == Direction.UP: eye1, eye2 = (head_center_x-4, head_center_y-3), (head_center_x+4, head_center_y-3)
        else: eye1, eye2 = (head_center_x-4, head_center_y+3), (head_center_x+4, head_center_y+3)
        pygame.draw.circle(self.display, WHITE, eye1, 3); pygame.draw.circle(self.display, WHITE, eye2, 3)
        pygame.draw.circle(self.display, BLACK, eye1, 1); pygame.draw.circle(self.display, BLACK, eye2, 1)

    def _update_ui(self):
        if not self.render: return
        
        # 1. Fundal
        self.display.fill(GREEN)
        
        # 2. Obstacole (FIXE)
        for pt in self.obstacles:
            pygame.draw.rect(self.display, GRAY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), 1)
        
        # 3. Șarpe
        self._draw_snake()

        # 4. Măr
        self._draw_apple(self.food.x, self.food.y)
        
        # 5. Scor
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]): next_idx = (idx + 1) % 4; new_dir = clock_wise[next_idx]
        else: next_idx = (idx - 1) % 4; new_dir = clock_wise[next_idx]

        self.direction = new_dir
        x = self.head.x; y = self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)

# Manual testing
if __name__ == '__main__':
    game = SnakeGame(render=True)
    while True:
        # Loop simplificat pentru vizualizare hartă
        # Ca să miști șarpele, trebuie adăugată logică de citire taste, 
        # dar pentru a vedea harta e suficient să rulezi și se va opri când lovește zidul.
        game_over, score, _ = game.step([1, 0, 0]) # Mereu înainte
        if game_over:
            game.reset()