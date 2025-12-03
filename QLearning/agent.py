import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point 

# training constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.95
        self.memory = deque(maxlen=MAX_MEMORY)
        
        # two q-tables for double q-learning
        self.q_table_1 = {}
        self.q_table_2 = {}
        
        # epsilon decay parameters
        self.epsilon_start = 80 # low exploration based on qtables
        self.epsilon_min = 5
        self.epsilon_decay = 0.995

        # tracking actions for analysis
        self.current_game_frames = []  # save frames of the current game
        self.best_game_frames = []
        self.best_game_score = 0
        self.best_game_w = 0
        self.best_game_h = 0
        
        self.load_model()
    
    def start_recording(self):
        # start a new recording session
        self.current_game_frames = []
    
    def record_frame(self, game):
        # record the whole state of the game at current frame
        frame = {
            'snake': [Point(pt.x, pt.y) for pt in game.snake],  # deep copy
            'food': Point(game.food.x, game.food.y),
            'direction': game.direction,
            'score': game.score
        }
        self.current_game_frames.append(frame)
    
    def save_best_game(self, score, game_w, game_h):
        # save the current game if it's the best one so far
        if score > self.best_game_score:
            self.best_game_score = score
            self.best_game_w = game_w  
            self.best_game_h = game_h 
            self.best_game_frames = self.current_game_frames.copy()
            print(f"\n🏆 NEW RECORD GAME SAVED! Score: {score}, Frames: {len(self.best_game_frames)}")
            return True
        return False
    
    def get_state(self, game):
        head = game.snake[0]
        
        # normalized distance to food
        dx = (game.food.x - head.x) / game.w
        dy = (game.food.y - head.y) / game.h
        
        # discretize distances into categories
        food_dir_x = 0 if abs(dx) < 0.1 else (1 if dx > 0 else -1)
        food_dir_y = 0 if abs(dy) < 0.1 else (1 if dy > 0 else -1)
        
        # check danger in all 4 directions (absolute, not relative)
        danger_up = game.is_collision(Point(head.x, head.y - 20))
        danger_down = game.is_collision(Point(head.x, head.y + 20))
        danger_left = game.is_collision(Point(head.x - 20, head.y))
        danger_right = game.is_collision(Point(head.x + 20, head.y))
        
        state = (
            danger_up, danger_down, danger_left, danger_right,
            food_dir_x + 1,  # convert -1,0,1 -> 0,1,2 for tuple
            food_dir_y + 1,
            game.direction.value  # current direction
        )
        
        return state
    
    def get_q_values(self, state, table=1):
        q_table = self.q_table_1 if table == 1 else self.q_table_2
        if state not in q_table:
            q_table[state] = np.zeros(3)  # neutral initialization
        return q_table[state]

    def remember(self, state, action, reward, next_state, done):
        # calculate td error as priority
        q_current = self.get_q_values(state, table=1)[np.argmax(action)]
        q_next = np.max(self.get_q_values(next_state, table=1))
        td_error = abs(reward + self.gamma * q_next - q_current)
        
        # store with priority
        priority = td_error + 0.01  # add epsilon to avoid zero priority
        self.memory.append((state, action, reward, next_state, done, priority))

    def train_short_memory(self, state, action, reward, next_state, done):
        """immediate training after each step"""
        self.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """batch training with prioritized experience replay"""
        if len(self.memory) > BATCH_SIZE:
            # extract priorities
            priorities = np.array([exp[5] for exp in self.memory])
            probabilities = priorities / priorities.sum()
            
            # sample based on priority
            indices = np.random.choice(len(self.memory), BATCH_SIZE, p=probabilities)
            mini_sample = [self.memory[i] for i in indices]
        else:
            mini_sample = self.memory
        
        for state, action, reward, next_state, done, _ in mini_sample:
            self.train_step(state, action, reward, next_state, done)

    def train_step(self, state, action, reward, next_state, done):
        """double q-learning update step"""
        action_index = np.argmax(action)
        alpha = max(0.1, 0.5 / (1 + self.n_games * 0.001))
        
        # randomly choose which q-table to update
        if random.random() < 0.5:
            q_values = self.get_q_values(state, table=1).copy()
            q_next_1 = self.get_q_values(next_state, table=1)
            q_next_2 = self.get_q_values(next_state, table=2)
            
            if done:
                new_q = reward
            else:
                # table 1 selects action, table 2 evaluates it
                best_action = np.argmax(q_next_1)
                new_q = reward + self.gamma * q_next_2[best_action]
            
            q_values[action_index] = (1 - alpha) * q_values[action_index] + alpha * new_q
            self.q_table_1[state] = q_values
        else:
            q_values = self.get_q_values(state, table=2).copy()
            q_next_1 = self.get_q_values(next_state, table=1)
            q_next_2 = self.get_q_values(next_state, table=2)
            
            if done:
                new_q = reward
            else:
                best_action = np.argmax(q_next_2)
                new_q = reward + self.gamma * q_next_1[best_action]
            
            q_values[action_index] = (1 - alpha) * q_values[action_index] + alpha * new_q
            self.q_table_2[state] = q_values

    def get_action(self, state):
        """epsilon-greedy action selection with exponential decay"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start * (self.epsilon_decay ** self.n_games)
        )
        
        final_move = [0, 0, 0]
        
        if random.random() < self.epsilon / 100:
            # exploration - random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # exploitation - combine both q-tables for decision
            q1 = self.get_q_values(state, table=1)
            q2 = self.get_q_values(state, table=2)
            q_combined = (q1 + q2) / 2
            
            move = np.argmax(q_combined)
            final_move[move] = 1
        
        return final_move
    
    def save_model(self, filename='q_tables.npy'):
        """save both q-tables to disk"""
        try:
            if not os.path.exists('model'):
                os.makedirs('model')
            
            filepath = os.path.join('model', filename)
            
            # save both tables
            data = {
                'q_table_1': self.q_table_1,
                'q_table_2': self.q_table_2
            }
            
            np.save(filepath, data)
            print(f"model saved: {len(self.q_table_1)} states learned")

        except Exception as e:
            print(f"error saving model: {e}")

    def load_model(self, filename='q_tables.npy'):
        """load both q-tables from disk"""
        filepath = os.path.join('model', filename)
        if os.path.exists(filepath):
            try:
                data = np.load(filepath, allow_pickle=True).item()
                self.q_table_1 = data.get('q_table_1', {})
                self.q_table_2 = data.get('q_table_2', {})
                print(f"model loaded: {len(self.q_table_1)} states")
            except Exception as e:
                print(f"error loading model: {e}")
                self.q_table_1 = {}
                self.q_table_2 = {}
        else:
            print("new model - starting training from scratch")
            self.q_table_1 = {}
            self.q_table_2 = {}