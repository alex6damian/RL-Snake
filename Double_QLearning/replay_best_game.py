import sys
import os

# --- FIX IMPORTURI ---
# Adăugăm folderul părinte în path pentru a găsi game.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import numpy as np
import time
from game import SnakeGame, Point, Direction, BLOCK_SIZE

# --- CONFIGURARE CULORI ---
try:
    from game import (GREEN, DARK_RED, BROWN, LEAF_GREEN, 
                      SNAKE_GREEN, SNAKE_DARK_GREEN, SNAKE_LIGHT_GREEN, 
                      WHITE, BLACK, RED, GRAY)
except ImportError:
    # Fallback dacă nu sunt definite în game.py
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
    """
    Reconstruiește harta obstacolelor. 
    Aceasta trebuie să fie IDENTICĂ cu logica din game.py.
    """
    obstacles = []
    cols = w // BLOCK_SIZE
    rows = h // BLOCK_SIZE
    cx, cy = cols // 2, rows // 2
    
    # 1. Piloni în colțuri
    pilar_offset = 5
    pillars = [
        (pilar_offset, pilar_offset),
        (pilar_offset, rows - pilar_offset),
        (cols - pilar_offset, pilar_offset),
        (cols - pilar_offset, rows - pilar_offset)
    ]
    for px, py in pillars:
        obstacles.append(Point(px * BLOCK_SIZE, py * BLOCK_SIZE))

    # 2. Ziduri centrale (Crucea)
    gap = 3
    # Orizontal
    for x in range(cx - 8, cx + 9):
        if abs(x - cx) < gap: continue
        pt = Point(x * BLOCK_SIZE, cy * BLOCK_SIZE)
        if pt not in obstacles: obstacles.append(pt)
            
    # Vertical
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
        # Gestionăm cazul în care datele sunt salvate ca dict sau ca obiect Point
        px = pt.x if hasattr(pt, 'x') else pt['x']
        py = pt.y if hasattr(pt, 'y') else pt['y']
        
        rect = pygame.Rect(px + 2, py + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
        pygame.draw.rect(display, SNAKE_GREEN, rect, border_radius=5)
        pygame.draw.rect(display, SNAKE_DARK_GREEN, rect, width=2, border_radius=5)
        
        # Detalii burtă
        if i > 0:
            center_rect = pygame.Rect(px + 6, py + 6, BLOCK_SIZE - 12, BLOCK_SIZE - 12)
            pygame.draw.rect(display, SNAKE_LIGHT_GREEN, center_rect, border_radius=3)
        
        # Conexiuni între segmente
        if i < len(snake) - 1:
            next_pt = snake[i + 1]
            npx = next_pt.x if hasattr(next_pt, 'x') else next_pt['x']
            npy = next_pt.y if hasattr(next_pt, 'y') else next_pt['y']
            
            if px == npx:
                if py < npy: connect_rect = pygame.Rect(px + 2, py + BLOCK_SIZE - 4, BLOCK_SIZE - 4, 6)
                else: connect_rect = pygame.Rect(px + 2, py - 2, BLOCK_SIZE - 4, 6)
            else:
                if px < npx: connect_rect = pygame.Rect(px + BLOCK_SIZE - 4, py + 2, 6, BLOCK_SIZE - 4)
                else: connect_rect = pygame.Rect(px - 2, py + 2, 6, BLOCK_SIZE - 4)
            pygame.draw.rect(display, SNAKE_GREEN, connect_rect)
    
    # Desenare Cap
    head = snake[0]
    hx = head.x if hasattr(head, 'x') else head['x']
    hy = head.y if hasattr(head, 'y') else head['y']
    
    head_cx = hx + BLOCK_SIZE // 2
    head_cy = hy + BLOCK_SIZE // 2
    
    # Poziționare ochi
    if direction == Direction.RIGHT: eye1, eye2 = (head_cx+3, head_cy-4), (head_cx+3, head_cy+4)
    elif direction == Direction.LEFT: eye1, eye2 = (head_cx-3, head_cy-4), (head_cx-3, head_cy+4)
    elif direction == Direction.UP: eye1, eye2 = (head_cx-4, head_cy-3), (head_cx+4, head_cy-3)
    else: eye1, eye2 = (head_cx-4, head_cy+3), (head_cx+4, head_cy+3)
    
    pygame.draw.circle(display, WHITE, eye1, 3); pygame.draw.circle(display, WHITE, eye2, 3)
    pygame.draw.circle(display, BLACK, eye1, 1); pygame.draw.circle(display, BLACK, eye2, 1)

def replay_best_game(replay_file):
    # Încărcare date
    try:
        data = np.load(replay_file, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"File not found: {replay_file}")
        return

    frames = data['frames']
    score = data['score']
    w = data.get('w', 640) # Default la 640 dacă lipsește
    h = data.get('h', 480)
    
    print(f"\n--- REPLAY DOUBLE Q-LEARNING ---")
    print(f"Score: {score}")
    print(f"Grid: {w}x{h}")
    print(f"Frames: {len(frames)}")
    print("--------------------------------\n")
    
    pygame.init()
    display = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f'Double Q Replay - Score: {score}')
    clock = pygame.time.Clock()
    
    # Generăm obstacolele
    obstacles = get_fixed_obstacles(w, h)

    running = True
    for i, frame in enumerate(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
        
        if not running: break
        
        display.fill(GREEN)
        
        # 1. Desenăm Obstacole
        draw_obstacles(display, obstacles)
        
        # 2. Extragem coordonatele mâncării (Robust: dict sau object)
        food = frame['food']
        fx = food.x if hasattr(food, 'x') else food['x']
        fy = food.y if hasattr(food, 'y') else food['y']
        
        # 3. Desenăm Șarpe și Măr
        draw_snake(display, frame['snake'], frame['direction'])
        draw_apple(display, fx, fy)
        
        pygame.display.flip()
        clock.tick(20) # FPS - ajustează viteza aici
    
    print("Replay finished.")
    time.sleep(1)
    pygame.quit()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit()

    # Căutăm fișiere specifice pentru DOUBLE Q (double_q_best_replay_)
    prefix = 'double_q_best_replay_'
    replay_files = [f for f in os.listdir(results_dir) if f.startswith(prefix)]
    
    if replay_files:
        # Sortăm pentru a lua cel mai recent
        latest_replay = sorted(replay_files)[-1]
        replay_path = os.path.join(results_dir, latest_replay)
        print(f"Loading replay file: {latest_replay}")
        replay_best_game(replay_path)
    else:
        print(f"No replay files found starting with '{prefix}' in {results_dir}")