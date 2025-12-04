import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet(nn.Module):
    """Optimized network for DQN."""
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path. exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """Optimized trainer."""
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = next(model.parameters()).device
    
    def train_step(self, state, action, reward, next_state, done):
        """Optimized training for batch processing."""
        # convert to numpy arrays first (faster)
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np. array(action)
        reward = np.array(reward, dtype=np.float32)
        
        # check if batch or single sample
        if len(state. shape) == 1:
            state = np.expand_dims(state, 0)
            next_state = np.expand_dims(next_state, 0)
            action = np.expand_dims(action, 0)
            reward = np.expand_dims(reward, 0)
            done = (done, )
        
        # convert to tensors once
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch. tensor(next_state, dtype=torch.float, device=self. device)
        action = torch. tensor(action, dtype=torch. long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        
        # predicted Q values
        pred = self.model(state)
        target = pred.clone()
        
        # calculate Q values for batch (vectorized)
        with torch.no_grad():
            next_q_values = self.model(next_state)
            max_next_q = torch.max(next_q_values, dim=1)[0]
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * max_next_q[idx]
            
            target[idx][torch.argmax(action[idx]). item()] = Q_new
        
        # backpropagation
        self. optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()