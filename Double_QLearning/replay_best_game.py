import sys
import os

# --- FIX PENTRU IMPORTURI (Găsirea game.py în folderul părinte) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
os.chdir(parent_dir) # Schimbăm contextul pentru a încărca iconițe/fonturi corect
# ------------------------------------------------------------------

import pygame
import numpy as np
import time
from game import SnakeGame, Point, Direction, BLOCK_SIZE

# Culori (le definim explicit ca să fim siguri)
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
GRAY = (128, 128, 128) # Culoare pentru obstacole

def draw_apple(display, x, y):
    """ Desenează mărul detaliat (3D effect) """
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
    """ Desenează șarpele detaliat (segmente conectate + ochi) """
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
                connect_rect = pygame.Rect(pt.x + 2, min(pt.y, next_pt.y) + BLOCK_SIZE - 4, BLOCK_SIZE - 4, 8)
            else:
                connect_rect = pygame.Rect(min(pt.x, next_pt.x) + BLOCK_SIZE - 4, pt.y + 2, 8, BLOCK_SIZE - 4)
            pygame.draw.rect(display, SNAKE_GREEN, connect_rect)

    # Capul și ochii
    head = snake[0]
    head_cx, head_cy = head.x + BLOCK_SIZE // 2, head.y + BLOCK_SIZE // 2
    
    if direction == Direction.RIGHT: eye1, eye2 = (head_cx+3, head_cy-4), (head_cx+3, head_cy+4)
    elif direction == Direction.LEFT: eye1, eye2 = (head_cx-3, head_cy-4), (head_cx-3, head_cy+4)
    elif direction == Direction.UP: eye1, eye2 = (head_cx-4, head_cy-3), (head_cx+4, head_cy-3)
    else: eye1, eye2 = (head_cx-4, head_cy+3), (head_cx+4, head_cy+3)
    
    pygame.draw.circle(display, WHITE, eye1, 3)
    pygame.draw.circle(display, WHITE, eye2, 3)
    pygame.draw.circle(display, BLACK, eye1, 1)
    pygame.draw.circle(display, BLACK, eye2, 1)

def replay_best_game(replay_file):
    if not os.path.exists(replay_file):
        print(f"Error: File {replay_file} not found.")
        return

    # Încărcare date
    try:
        data = np.load(replay_file, allow_pickle=True).item()
        # Convertim frame-urile dacă sunt salvate ca dict în loc de Point
        frames_raw = data['frames']
        frames = []
        for f in frames_raw:
            # Reconstruim obiectele Point dacă e nevoie
            snake_pts = [Point(p['x'], p['y']) if isinstance(p, dict) else p for p in f['snake']]
            food_pt = Point(f['food']['x'], f['food']['y']) if isinstance(f['food'], dict) else f['food']
            frames.append({
                'snake': snake_pts,
                'food': food_pt,
                'score': f['score'],
                'direction': f['direction']
            })
            
        score = data.get('score', 0)
        w = data.get('w', 640)
        h = data.get('h', 480)
    except Exception as e:
        print(f"Error reading replay file: {e}")
        return

    print(f"\n{'='*60}")
    print(f"  REPLAYING BEST GAME")
    print(f"{'='*60}")
    print(f"Score: {score}")
    print(f"Frames: {len(frames)}")
    print(f"Grid: {w}x{h}")
    print(f"{'='*60}\n")

    # Inițializare Pygame și Joc REAL (pentru obstacole)
    pygame.init()
    # Inițializăm jocul doar pentru a calcula obstacolele
    game = SnakeGame(w=w, h=h, render=True) 
    
    clock = pygame.time.Clock()
    fps = 20 # Viteza inițială

    running = True
    paused = False
    
    for i, frame in enumerate(frames):
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_UP: fps += 5
                if event.key == pygame.K_DOWN: fps = max(1, fps - 5)

        if not running: break
        
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False; paused = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: paused = False
            time.sleep(0.1)

        # --- DESENARE ---
        
        # 1. Fundal
        game.display.fill(GREEN)
        
        # 2. Obstacole (CRITIC: Luăm lista din instanța jocului)
        if hasattr(game, 'obstacles'):
            for obs in game.obstacles:
                pygame.draw.rect(game.display, GRAY, pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE))
                # Contur obstacol
                pygame.draw.rect(game.display, (100, 100, 100), pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE), 2)

        # 3. Mâncare (Grafică avansată)
        draw_apple(game.display, frame['food'].x, frame['food'].y)

        # 4. Șarpe (Grafică avansată)
        draw_snake(game.display, frame['snake'], frame['direction'])

        # 5. UI (Scor și Info)
        font = pygame.font.SysFont('arial', 18)
        text_score = font.render(f"Score: {frame['score']}", True, WHITE)
        
        # Desenăm background la text pentru vizibilitate
        game.display.blit(text_score, (10, 10))

        pygame.display.flip()
        clock.tick(fps)

    print("Replay finished.")
    time.sleep(1)
    pygame.quit()

if __name__ == '__main__':
    # Căutăm fișierul în folderul results curent
    current_folder = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(current_folder, 'results')
    
    if os.path.exists(results_folder):
        files = [f for f in os.listdir(results_folder) if f.endswith('.npy')]
        if files:
            # Luăm cel mai nou fișier
            latest_file = max([os.path.join(results_folder, f) for f in files], key=os.path.getctime)
            replay_best_game(latest_file)
        else:
            print(f"No .npy files found in {results_folder}")
    else:
        print(f"Results folder not found: {results_folder}")