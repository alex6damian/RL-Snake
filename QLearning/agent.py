import sys
import os
# Adaugă directorul părinte (root) la calea de căutare a modulelor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Acum importurile vor funcționa corect
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point # <- Acest import va funcționa acum

# Constante pentru antrenament
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # Learning Rate

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # parametru pentru explorare/exploatare
        self.gamma = 0.9  # discount factor
        self.memory = deque(maxlen=MAX_MEMORY)  # stochează experiențele (state, action, reward, next_state, done)
        # TODO: Aici va veni modelul (Q-table sau Rețea Neuronală)
        # Pentru Q-learning clasic, am folosi un dicționar
        self.q_table = {}

    def get_state(self, game):
        """
        Determină starea curentă a jocului.
        Aceasta este o funcție esențială.
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Starea este un tuplu de 11 valori booleene
        state = (
            # Pericol în față
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Pericol la dreapta
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Pericol la stânga
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Direcția de mișcare
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Poziția mâncării
            game.food.x < game.head.x,  # Mâncare la stânga
            game.food.x > game.head.x,  # Mâncare la dreapta
            game.food.y < game.head.y,  # Mâncare sus
            game.food.y > game.head.y   # Mâncare jos
        )

        return tuple(int(i) for i in state)

    def get_q_values(self, state):
        """
        Obține valorile Q pentru o stare. Dacă starea e nouă, o inițializează.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)  # [drept, dreapta, stânga]
        return self.q_table[state]

    def remember(self, state, action, reward, next_state, done):
        """
        Stochează o experiență în memorie.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Antrenament pe baza ultimei mutări (învață la fiecare pas).
        """
        self.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        """
        Antrenament pe baza unui eșantion aleatoriu din memorie (învață din experiențe trecute).
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self.train_step(state, action, reward, next_state, done)

    def train_step(self, state, action, reward, next_state, done):
        """
        Funcția care aplică formula Bellman pentru a actualiza Q-table.
        """
        q_values = self.get_q_values(state).copy()
        q_values_next = self.get_q_values(next_state)

        # Găsește indexul acțiunii. Acțiunea e [1,0,0], [0,1,0] sau [0,0,1]
        action_index = np.argmax(action)

        if done:
            new_q = reward
        else:
            # Formula Q-learning: Q_new = R + gamma * max(Q_next)
            new_q = reward + self.gamma * np.max(q_values_next)

        # Actualizează valoarea Q pentru starea și acțiunea respective
        q_values[action_index] = new_q
        self.q_table[state] = q_values

    def get_action(self, state):
        """
        Decide ce acțiune să ia: explorare (aleatoriu) sau exploatare (cea mai bună).
        """
        # Epsilon scade pe măsură ce agentul învață mai multe jocuri
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        # Explorare vs. Exploatare
        if random.randint(0, 200) < self.epsilon:
            # Alege o mișcare aleatorie
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Alege cea mai bună mișcare pe baza Q-table
            q_values = self.get_q_values(state)
            move = np.argmax(q_values)
            final_move[move] = 1

        return final_move
