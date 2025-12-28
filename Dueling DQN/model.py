import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class DuelingQNet(nn.Module):
    """
    Dueling DQN Architecture.
    Splits the network into two streams: Value (V) and Advantage (A).
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingQNet, self).__init__()
        
        # 1. Stratul Comun (Shared Feature Layer)
        # Procesează input-ul inițial (starea jocului)
        self.linear1 = nn.Linear(input_size, hidden_size)
        
        # 2. Value Stream (Fluxul de Valoare)
        # Estimează valoarea generală a stării: V(s)
        self.fc_value = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1) # Ieșire scalară (1 singură valoare)
        
        # 3. Advantage Stream (Fluxul de Avantaj)
        # Estimează importanța fiecărei acțiuni: A(s, a)
        self.fc_adv = nn.Linear(hidden_size, hidden_size)
        self.advantage = nn.Linear(hidden_size, output_size) # Ieșire vectorială (3 acțiuni)

    def forward(self, x):
        # Trecem prin stratul comun
        x = F.relu(self.linear1(x))
        
        # Calculăm Value (V)
        val = F.relu(self.fc_value(x))
        V = self.value(val)
        
        # Calculăm Advantage (A)
        adv = F.relu(self.fc_adv(x))
        A = self.advantage(adv)
        
        # Combinăm V și A folosind formula Dueling DQN:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Scăderea mediei ajută la stabilitatea antrenării
        Q = V + (A - torch.mean(A, dim=1, keepdim=True))
        
        return Q

    def save(self, file_name='model.pth'):
        # Salvăm în folderul curent (Dueling_DQN/model)
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """Optimized trainer (Identic cu cel din DQN simplu)."""
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = next(model.parameters()).device # Preia device-ul modelului (CPU/GPU)

    def train_step(self, state, action, reward, next_state, done):
        """Optimized training for batch processing."""
        # Convert to numpy arrays first (faster conversion)
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action)
        reward = np.array(reward, dtype=np.float32)

        # Handle single dimension (if batch size = 1)
        if len(state.shape) == 1:
            state = np.expand_dims(state, 0)
            next_state = np.expand_dims(next_state, 0)
            action = np.expand_dims(action, 0)
            reward = np.expand_dims(reward, 0)
            done = (done, )

        # Convert to tensors once
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        # 1: Predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        
        # 2: Q_new = r + y * max(next_predicted Q value)
        with torch.no_grad():
            next_q_values = self.model(next_state)
            max_next_q = torch.max(next_q_values, dim=1)[0]

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * max_next_q[idx]

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 3: Loss and Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()