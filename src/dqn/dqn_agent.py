import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from omegaconf import DictConfig

from model import DQN, NoisyDQN

class DQNAgent:
    def __init__(self, cfg: DictConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(action_size=cfg.action_dim , state_dim=cfg.state_dim, hidden_dim1=cfg.hidden_dim_1, hidden_dim2=cfg.hidden_dim_2).to(self.device)
        self.target_network = DQN(action_size=cfg.action_dim , state_dim=cfg.state_dim, hidden_dim1=cfg.hidden_dim_1, hidden_dim2=cfg.hidden_dim_2).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=cfg.learning_rate)
        self.criterion = nn.MSELoss()
        self.updates = cfg.updates  # Number of updates per step
        self.tau = cfg.tau  # Rate of soft update
        self.gamma = cfg.gamma  # Discount factor
        self.epsilon = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay

        
    def select_action(self, state, get_q_values=False, evaluate_agent=False):
        
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            # import pdb; pdb.set_trace()
            q_values = self.q_network(state_tensor)
            if get_q_values:
                return q_values.cpu().numpy()
            # import pdb; pdb.set_trace()
            if evaluate_agent:
                return torch.argmax(q_values, dim=1).item()
            # import pdb; pdb.set_trace()
            return torch.argmax(q_values, dim=1).tolist()
                
    def update(self, batch):
        mean_loss = []
        for _ in range(self.updates):
            states, actions, rewards, next_states, dones = batch
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Q-values of current states
            q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            # Q-values of next states using target network
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            
            # Target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = self.criterion(q_values, target_q_values)
            
            mean_loss.append(loss.detach().item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update()
        self.decay_epsilon()
        return np.mean(mean_loss)

    def soft_update(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)




class DoubleDQNAgent(DQNAgent):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
    def update(self, batch):
        mean_loss = []
        for _ in range(self.updates):
            states, actions, rewards, next_states, dones = batch
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            # Q-values of current states
            q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # Get best actions for next states from Q-network
            best_actions = self.q_network(next_states).max(1)[1].detach()

            # Get Q-values of next states using target network and the actions from Q-network
            next_q_values = self.target_network(next_states).gather(1, best_actions.unsqueeze(-1)).squeeze(-1).detach()

            # Target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = self.criterion(q_values, target_q_values)

            mean_loss.append(loss.detach().item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update()

        self.decay_epsilon()
        return np.mean(mean_loss)
    



class NoisyDQNAgent(DQNAgent):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.q_network = NoisyDQN(action_size=cfg.action_dim, state_dim=cfg.state_dim, hidden_dim1=cfg.hidden_dim_1, hidden_dim2=cfg.hidden_dim_2).to(self.device)
        self.target_network = NoisyDQN(action_size=cfg.action_dim, state_dim=cfg.state_dim, hidden_dim1=cfg.hidden_dim_1, hidden_dim2=cfg.hidden_dim_2).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=cfg.learning_rate)

    def set_noise_level(self, noise_level):
        for layer in [self.q_network.fc1, self.q_network.fc2, self.q_network.fc3,
                      self.target_network.fc1, self.target_network.fc2, self.target_network.fc3]:
            layer.set_noise_level(noise_level)