import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer

# Constante pentru antrenament
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class DQNAgent:
    """
    Agent Deep Q-Learning optimizat pentru antrenament rapid.  
    """
    def __init__(self, epsilon_start=80, epsilon_min=0):
        self.n_games = 0
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_start
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # Verifică dacă CUDA e disponibil
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        
        self.model = Linear_QNet(11, 256, 3).to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Recording
        self.best_game_score = 0
        self.best_game_frames = []
        self.current_game_frames = []
        self.is_recording = False
        self. best_game_w = 0
        self.best_game_h = 0
    
    def get_state(self, game):
        """Returnează starea - OPTIMIZAT."""
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction. DOWN

        state = np.array([
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game. is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l, dir_r, dir_u, dir_d,

            game. food.x < game.head.x,
            game.food.x > game. head.x,
            game. food.y < game.head. y,
            game.food. y > game.head.y
        ], dtype=np.float32)

        return state
    
    def start_recording(self):
        """Începe înregistrarea unui joc nou."""
        self.current_game_frames = []
        self.is_recording = True
    
    def record_frame(self, game):
        """
        Salvează frame-ul curent - FIX: înregistrează toate frame-urile pentru best game. 
        """
        if self. is_recording:
            frame = {
                'snake': game.snake. copy(),
                'food': game.food,
                'score': game.score,
                'direction': game.direction
            }
            self.current_game_frames.append(frame)
    
    def save_best_game(self, score, w, h):
        """Salvează jocul dacă este cel mai bun."""
        if score > self.best_game_score:
            self.best_game_score = score
            self. best_game_frames = self. current_game_frames.copy()
            self.best_game_w = w
            self.best_game_h = h
    
    def save_model(self):
        """Salvează modelul neural."""
        model_folder_path = './model'
        if not os. path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, 'dqn_model.pth')
        torch.save(self.model. state_dict(), file_name)
    
    def load_model(self, file_name='dqn_model.pth'):
        """Încarcă modelul neural."""
        model_path = os.path.join('./model', file_name)
        if os.path.exists(model_path):
            self.model. load_state_dict(torch. load(model_path, map_location=self.device))
            self.model.eval()
            return True
        return False
    
    def remember(self, state, action, reward, next_state, done):
        """Stochează experiența."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        """Experience replay - OPTIMIZAT."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        if len(mini_sample) == 0:
            return
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        """Antrenament pe ultimul pas."""
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        """Epsilon-greedy - OPTIMIZAT cu torch."""
        self.epsilon = max(self.epsilon_min, self.epsilon_start - self.n_games)
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
            with torch.no_grad():
                prediction = self.model(state_tensor)
            move = torch.argmax(prediction). item()
            final_move[move] = 1
        
        return final_move