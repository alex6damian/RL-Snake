import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from game import Direction, Point
from model import ActorCriticNet, PPOTrainer

# Training constants
MAX_MEMORY = 100_000
UPDATE_TIMESTEP = 2000  # Update policy every n timesteps
K_EPOCHS = 4  # Update policy for K epochs
EPS_CLIP = 0.2  # Clip parameter for PPO
LR = 0.0003

class PPOAgent:
    # PPO agent with actor-critic architecture
    def __init__(self, lr=LR, gamma=0.99, eps_clip=EPS_CLIP, k_epochs=K_EPOCHS):
        self.n_games = 0
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # PPO uses on-policy learning, so we have a smaller memory
        self.memory = []
        self.update_timestep = UPDATE_TIMESTEP
        self.timestep = 0

        # Check device (GPU vs CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")

        # Actor-Critic network
        # Input: 11 (state), Hidden: 256, Output: 3 (actions)
        self.model = ActorCriticNet(11, 256, 3).to(self.device)
        self.trainer = PPOTrainer(self.model, lr=lr, gamma=gamma,
                                 eps_clip=eps_clip, K_epochs=k_epochs)

        # Recording stats
        self.best_game_score = 0
        self.best_game_frames = []
        self.current_game_frames = []
        self.is_recording = False
        self.best_game_w = 0
        self.best_game_h = 0

    def get_state(self, game):
        # return state (same as DQN for consistency)
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = np.array([
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ], dtype=np.float32)

        return state

    def start_recording(self):
        # start recording
        self.current_game_frames = []
        self.is_recording = True

    def record_frame(self, game):
        # record current frame
        if self.is_recording:
            frame = {
                'snake': game.snake.copy(),
                'food': game.food,
                'score': game.score,
                'direction': game.direction
            }
            self.current_game_frames.append(frame)

    def save_best_game(self, score, w, h):
        # saving best game
        if score > self.best_game_score:
            self.best_game_score = score
            self.best_game_frames = self.current_game_frames.copy()
            self.best_game_w = w
            self.best_game_h = h
        self.current_game_frames = []

    def save_model(self):
        # saving model
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, 'ppo_model.pth')
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name='ppo_model.pth'):
        # loading model
        model_path = os.path.join('./model', file_name)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            return True
        return False

    def remember(self, state, action, action_logprob, reward, state_value, done):
        # saving experience
        self.memory.append((state, action, action_logprob.item(), reward, state_value.item(), done))

    def clear_memory(self):
        # clear memory after update (PPO is on-policy)
        self.memory = []

    def get_action(self, state, train=True):
        # epsilon-greedy with stochastic policy
        state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)

        if train:
            # sample action from policy
            with torch.no_grad():
                action, action_logprob, state_value = self.model.get_action(state_tensor)
        else:
            # greedy action selection (for evaluation)
            with torch.no_grad():
                action_probs, state_value = self.model(state_tensor)
                action = torch.argmax(action_probs).item()
                action_logprob = None

        # convert action to one-hot encoding
        final_move = [0, 0, 0]
        final_move[action] = 1

        if train:
            return final_move, action_logprob, state_value
        else:
            return final_move

    def update_policy(self):
        # update policy using PPO
        if len(self.memory) == 0:
            return 0

        loss = self.trainer.train_step(self.memory)
        self.clear_memory()
        return loss
