import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import numpy as np
import time
from game import SnakeGame, Point, Direction, BLOCK_SIZE, font

# Import culori
try:
    from game import (GREEN, DARK_RED, BROWN, LEAF_GREEN, 
                      SNAKE_GREEN, SNAKE_DARK_GREEN, SNAKE_LIGHT_GREEN, 
                      WHITE, BLACK, RED, GRAY)
except ImportError:
    GREEN = (124, 252, 0)
    RED = (200, 0, 0)
    DARK_RED = (150, 0, 0)
    BROWN = (101, 67, 33)
    LEAF_GREEN = (34, 139, 34)
    SNAKE_GREEN = (34, 139, 34)
    SNAKE_DARK_GREEN = (0, 100, 0)
    SNAKE_LIGHT_GREEN = (144, 238, 144)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)

def get_fixed_obstacles(w, h):
    """Generează aceleași obstacole ca în game.py (Harta Fixă)"""
    obstacles = []
    cx, cy = (w // BLOCK_SIZE) // 2, (h // BLOCK_SIZE) // 2
    
    pilar_offset = 5
    pillars = [
        (pilar_offset, pilar_offset),
        (pilar_offset, (h // BLOCK_SIZE) - pilar_offset),
        ((w // BLOCK_SIZE) - pilar_offset, pilar_offset),
        ((w // BLOCK_SIZE) - pilar_offset, (h // BLOCK_SIZE) - pilar_offset)
    ]
    for px, py in pillars:
        obstacles.append(Point(px * BLOCK_SIZE, py * BLOCK_SIZE))

    gap = 3
    for x in range(cx - 8, cx + 9):
        if abs(x - cx) < gap: continue
        pt = Point(x * BLOCK_SIZE, cy * BLOCK_SIZE)
        if pt not in obstacles: obstacles.append(pt)
            
    for y in range(cy - 6, cy + 7):
        if abs(y - cy) < gap: continue
        pt = Point(cx * BLOCK_SIZE, y * BLOCK_SIZE)
        if pt not in obstacles: obstacles.append(pt)
    
    return obstacles

def draw_obstacles(display, obstacles):
    for pt in obstacles:
        pygame.draw.rect(display, GRAY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(display, WHITE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), 1)

def draw_apple(display, x, y):
    center_x = x + BLOCK_SIZE // 2
    center_y = y + BLOCK_SIZE // 2
    pygame.draw.circle(display, RED, (center_x, center_y), BLOCK_SIZE // 2 - 2)
    pygame.draw.circle(display, (255, 100, 100), (center_x - 3, center_y - 3), 3)
    pygame.draw.circle(display, DARK_RED, (center_x + 2, center_y + 3), 4)
    stem_rect = pygame.Rect(center_x - 1, y + 2, 2, 4)
    pygame.draw.rect(display, BROWN, stem_rect)
    leaf_points = [(center_x + 2, y + 3), (center_x + 6, y + 2), (center_x + 4, y + 5)]
    pygame.draw.polygon(display, LEAF_GREEN, leaf_points)

def draw_snake(display, snake, direction):
    if len(snake) == 0: return
    for i, pt in enumerate(snake):
        rect = pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
        pygame.draw.rect(display, SNAKE_GREEN, rect, border_radius=5)
        pygame.draw.rect(display, SNAKE_DARK_GREEN, rect, width=2, border_radius=5)
        if i > 0:
            center_rect = pygame.Rect(pt.x + 6, pt.y + 6, BLOCK_SIZE - 12, BLOCK_SIZE - 12)
            pygame.draw.rect(display, SNAKE_LIGHT_GREEN, center_rect, border_radius=3)
        if i < len(snake) - 1:
            next_pt = snake[i + 1]
            if pt.x == next_pt.x:
                if pt.y < next_pt.y: connect_rect = pygame.Rect(pt.x + 2, pt.y + BLOCK_SIZE - 4, BLOCK_SIZE - 4, 6)
                else: connect_rect = pygame.Rect(pt.x + 2, pt.y - 2, BLOCK_SIZE - 4, 6)
            else:
                if pt.x < next_pt.x: connect_rect = pygame.Rect(pt.x + BLOCK_SIZE - 4, pt.y + 2, 6, BLOCK_SIZE - 4)
                else: connect_rect = pygame.Rect(pt.x - 2, pt.y + 2, 6, BLOCK_SIZE - 4)
            pygame.draw.rect(display, SNAKE_GREEN, connect_rect)
    
    head = snake[0]
    head_center_x = head.x + BLOCK_SIZE // 2
    head_center_y = head.y + BLOCK_SIZE // 2
    if direction == Direction.RIGHT: eye1, eye2 = (head_center_x+3, head_center_y-4), (head_center_x+3, head_center_y+4)
    elif direction == Direction.LEFT: eye1, eye2 = (head_center_x-3, head_center_y-4), (head_center_x-3, head_center_y+4)
    elif direction == Direction.UP: eye1, eye2 = (head_center_x-4, head_center_y-3), (head_center_x+4, head_center_y-3)
    else: eye1, eye2 = (head_center_x-4, head_center_y+3), (head_center_x+4, head_center_y+3)
    pygame.draw.circle(display, WHITE, eye1, 3); pygame.draw.circle(display, WHITE, eye2, 3)
    pygame.draw.circle(display, BLACK, eye1, 1); pygame.draw.circle(display, BLACK, eye2, 1)

def replay_best_game(replay_file):
    data = np.load(replay_file, allow_pickle=True).item()
    frames = data['frames']
    score = data['score']
    w = data['w']
    h = data['h']
    
    print(f"Replaying score: {score}")
    
    pygame.init()
    display = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f'Score: {score}')
    clock = pygame.time.Clock()
    
    obstacles = get_fixed_obstacles(w, h)

    for i, frame in enumerate(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
        
        display.fill(GREEN)
        
        draw_obstacles(display, obstacles)
        
        draw_snake(display, frame['snake'], frame['direction'])
        draw_apple(display, frame['food'].x, frame['food'].y)
        pygame.display.flip()
        clock.tick(30)
    
    time.sleep(1)
    pygame.quit()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    replay_files = [f for f in os.listdir(results_dir) if f.startswith('dueling_dqn_best_replay_')]
    
    if replay_files:
        latest_replay = sorted(replay_files)[-1]
        replay_path = os.path.join(results_dir, latest_replay)
        print(f"Loading: {latest_replay}")
        replay_best_game(replay_path)
    else:
        print("No replay files found.")