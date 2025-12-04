import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import numpy as np
import time
from game import SnakeGame, Point, Direction, BLOCK_SIZE, font

# Import all the new colors from game.py
try:
    from game import (GREEN, DARK_RED, BROWN, LEAF_GREEN, 
                      SNAKE_GREEN, SNAKE_DARK_GREEN, SNAKE_LIGHT_GREEN, 
                      WHITE, BLACK, RED)  # ADAUGĂ RED AICI
except ImportError:
    # Fallback colors if not available
    GREEN = (124, 252, 0)
    RED = (200, 0, 0)  # ADAUGĂ RED AICI
    DARK_RED = (150, 0, 0)
    BROWN = (101, 67, 33)
    LEAF_GREEN = (34, 139, 34)
    SNAKE_GREEN = (34, 139, 34)
    SNAKE_DARK_GREEN = (0, 100, 0)
    SNAKE_LIGHT_GREEN = (144, 238, 144)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

def draw_apple(display, x, y):
    """
    Draws an apple at the given position.
    """
    center_x = x + BLOCK_SIZE // 2
    center_y = y + BLOCK_SIZE // 2
    
    # Draw the main apple body (circle)
    pygame.draw.circle(display, RED, (center_x, center_y), BLOCK_SIZE // 2 - 2)
    
    # Add a highlight for 3D effect (small lighter circle on top-left)
    pygame.draw.circle(display, (255, 100, 100), (center_x - 3, center_y - 3), 3)
    
    # Add darker shading on bottom-right
    pygame.draw.circle(display, DARK_RED, (center_x + 2, center_y + 3), 4)
    
    # Draw stem (small brown rectangle at the top)
    stem_rect = pygame.Rect(center_x - 1, y + 2, 2, 4)
    pygame.draw.rect(display, BROWN, stem_rect)
    
    # Draw a small leaf
    leaf_points = [
        (center_x + 2, y + 3),
        (center_x + 6, y + 2),
        (center_x + 4, y + 5)
    ]
    pygame.draw.polygon(display, LEAF_GREEN, leaf_points)

def draw_snake(display, snake, direction):
    """
    Draws the snake as a continuous linear shape with smooth connections.
    """
    if len(snake) == 0:
        return
    
    # Draw body segments as connected rectangles
    for i, pt in enumerate(snake):
        # Draw the main body rectangle with rounded edges
        rect = pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
        pygame.draw.rect(display, SNAKE_GREEN, rect, border_radius=5)
        
        # Add darker outline
        pygame.draw.rect(display, SNAKE_DARK_GREEN, rect, width=2, border_radius=5)
        
        # Add lighter center stripe for belly
        if i > 0:  # Not the head
            center_rect = pygame.Rect(pt.x + 6, pt.y + 6, BLOCK_SIZE - 12, BLOCK_SIZE - 12)
            pygame.draw.rect(display, SNAKE_LIGHT_GREEN, center_rect, border_radius=3)
        
        # Connect segments smoothly
        if i < len(snake) - 1:
            next_pt = snake[i + 1]
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
            pygame.draw.rect(display, SNAKE_GREEN, connect_rect)
    
    # Draw the head with eyes
    head = snake[0]
    head_center_x = head.x + BLOCK_SIZE // 2
    head_center_y = head.y + BLOCK_SIZE // 2
    
    # Draw eyes based on direction
    if direction == Direction.RIGHT:
        eye1_pos = (head_center_x + 3, head_center_y - 4)
        eye2_pos = (head_center_x + 3, head_center_y + 4)
    elif direction == Direction.LEFT:
        eye1_pos = (head_center_x - 3, head_center_y - 4)
        eye2_pos = (head_center_x - 3, head_center_y + 4)
    elif direction == Direction.UP:
        eye1_pos = (head_center_x - 4, head_center_y - 3)
        eye2_pos = (head_center_x + 4, head_center_y - 3)
    else:  # DOWN
        eye1_pos = (head_center_x - 4, head_center_y + 3)
        eye2_pos = (head_center_x + 4, head_center_y + 3)
    
    # Draw white of eyes
    pygame.draw.circle(display, WHITE, eye1_pos, 3)
    pygame.draw.circle(display, WHITE, eye2_pos, 3)
    
    # Draw pupils
    pygame.draw.circle(display, BLACK, eye1_pos, 1)
    pygame.draw.circle(display, BLACK, eye2_pos, 1)

def replay_best_game(replay_file):
    # load replay data
    data = np.load(replay_file, allow_pickle=True).item()
    frames = data['frames']
    score = data['score']
    w = data['w']
    h = data['h']
    
    print(f"\n{'='*60}")
    print(f"  REPLAYING BEST GAME")
    print(f"{'='*60}")
    print(f"Score: {score}")
    print(f"Frames: {len(frames)}")
    print(f"Dimensions: {w}x{h}")
    print(f"{'='*60}\n")
    
    # initialize pygame
    pygame.init()
    display = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f'Score: {score}')
    clock = pygame.time.Clock()
    
    # shows every frame
    for i, frame in enumerate(frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            # pause with SPACE
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = True
                    while paused:
                        for pause_event in pygame.event.get():
                            if pause_event.type == pygame.QUIT:
                                pygame.quit()
                                return
                            if pause_event.type == pygame.KEYDOWN:
                                if pause_event.key == pygame.K_SPACE:
                                    paused = False
        
        # draw grass background
        display.fill(GREEN)
        
        # draw snake with enhanced graphics
        draw_snake(display, frame['snake'], frame['direction'])
        
        # draw apple with enhanced graphics
        draw_apple(display, frame['food'].x, frame['food'].y)
        
        # Display score with better styling
        # text = font.render(f"Score: {frame['score']} | Frame: {i+1}/{len(frames)}", True, WHITE)
        # Add black background for better readability
        # text_rect = text.get_rect()
        # text_rect.topleft = (5, 5)
        # background_rect = text_rect.inflate(10, 5)
        # pygame.draw.rect(display, BLACK, background_rect)
        # display.blit(text, [5, 5])
        
        pygame.display.flip()
        
        # control speed - you can adjust this
        clock.tick(60)  # 10 FPS - ajustează pentru viteză
    
    print(f"\nReplay finished! Final score: {frames[-1]['score']}")
    print("Press any key to close...")
    
    # Wait for user to close
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                waiting = False
    
    pygame.quit()

if __name__ == '__main__':
    # caută cel mai recent fișier de replay
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    replay_files = [f for f in os.listdir(results_dir) if f.startswith('best_game_replay_')]
    
    if not replay_files:
        print("No replay files found!")
        print(f"Looked in: {results_dir}")
    else:
        # ia cel mai recent
        latest_replay = sorted(replay_files)[-1]
        replay_path = os.path.join(results_dir, latest_replay)
        print(f"Loading replay: {latest_replay}")
        print("\nControls:")
        print("  SPACE - Pause/Resume")
        print("  ESC/Close - Exit")
        print()
        replay_best_game(replay_path)