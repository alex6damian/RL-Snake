import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import numpy as np

class ActorCriticNet(nn.Module):
    """
    Actor-Critic Network for PPO.
    Actor: Outputs action probabilities (policy)
    Critic: Outputs state value estimate V(s)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCriticNet, self).__init__()

        # Shared feature extraction layers
        self.shared1 = nn.Linear(input_size, hidden_size)
        self.shared2 = nn.Linear(hidden_size, hidden_size)

        # Actor head (policy network)
        self.actor = nn.Linear(hidden_size, output_size)

        # Critic head (value network)
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize weights to prevent NaN
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # Shared layers
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))

        # Actor output: action probabilities with numerical stability
        actor_logits = self.actor(x)
        action_probs = F.softmax(actor_logits, dim=-1)

        # Clamp probabilities to prevent numerical instability
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        # Critic output: state value
        state_value = self.critic(x)

        return action_probs, state_value

    def get_action(self, state):
        """Sample action from policy."""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.item(), action_logprob, state_value

    def evaluate(self, state, action):
        """Evaluate action taken (used during training)."""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, state_value, dist_entropy

    def save(self, file_name='ppo_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class PPOTrainer:
    """PPO Trainer with clipped objective."""
    def __init__(self, model, lr, gamma, eps_clip, K_epochs):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip  # Clip parameter for PPO
        self.K_epochs = K_epochs  # Number of epochs to update policy

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.device = next(model.parameters()).device

        self.MseLoss = nn.MSELoss()

    def train_step(self, memory):
        """
        PPO update step with multiple epochs.
        memory: list of (state, action, action_logprob, reward, state_value, done)
        """
        if len(memory) == 0:
            return 0.0

        # Convert memory to tensors
        old_states = torch.tensor(np.array([exp[0] for exp in memory]), dtype=torch.float, device=self.device)
        old_actions = torch.tensor(np.array([exp[1] for exp in memory]), dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor(np.array([exp[2] for exp in memory]), dtype=torch.float, device=self.device)

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, done in [(exp[3], exp[5]) for exp in memory][::-1]:
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards (with safety check)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.model.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            # Make sure dimensions match
            if state_values.dim() == 0:
                state_values = state_values.unsqueeze(0)

            # Calculate advantage
            advantages = rewards - state_values.detach()

            # Calculate ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss: PPO clipped objective + value loss + entropy bonus
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # Backpropagation
            self.optimizer.zero_grad()
            loss.mean().backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            self.optimizer.step()

        return loss.mean().item()
