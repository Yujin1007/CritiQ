import torch
import torch.nn as nn
import torch.optim as optim
import os
class Discriminator(nn.Module):
    def __init__(self, state_dim=19, action_dim=4, seq_len=10, hidden_dim=256, lr=1e-3, device="cpu"):
        super(Discriminator, self).__init__()

        # 🔹 Define Discriminator (Binary Classifier for Expert vs. BC Output)
        self.model = nn.Sequential(
            nn.Linear(state_dim*seq_len + action_dim*seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs probability of (s, a) being from expert
        )
        
        # 🔹 Optimizer and Loss Function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.device = device
        self.to(device)

    def forward(self, state_action):
        """ Forward pass to classify (s, a) pairs. """
        # x = torch.cat((state, action), dim=-1)  # Concatenate state & action
        return self.model(state_action.to(self.device))

    def train_discriminator(self, expert_states, expert_actions, bc_states, bc_actions, batch_size=32):
        """ Train Discriminator to classify expert vs. BC-generated data. """
        self.train()
        self.optimizer.zero_grad()

        # Labels: 1 for expert data, 0 for BC policy output
        expert_labels = torch.ones((batch_size, 1))
        bc_labels = torch.zeros((batch_size, 1))

        # Forward pass through Discriminator
        expert_preds = self(expert_states, expert_actions)  # Expert data (should be 1)
        bc_preds = self(bc_states, bc_actions)  # BC policy data (should be 0)

        # Compute Binary Cross-Entropy Loss
        loss_expert = self.criterion(expert_preds, expert_labels)
        loss_bc = self.criterion(bc_preds, bc_labels)
        loss = loss_expert + loss_bc  # Total loss

        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_reward(self, state, action):
        """ Compute discriminator-based reward as -log(D(s, a)) """
        with torch.no_grad():
            reward = -torch.log(self(state, action) + 1e-8)  # Avoid log(0)
        return reward
    def load_model(self, path: str):
        # Load a pre-trained student policy
        
        self.load_state_dict(torch.load(path, weights_only=True))
        
    def predict(self, state_action_seq, deterministic=False):
        self.eval()
        with torch.no_grad():
            prob = self.forward(state_action_seq)
            return prob

class LSTMDiscriminator(nn.Module):
    def __init__(self, state_dim=19, action_dim=5, hidden_dim=64, lstm_layers=2, sequence_length=10, lr=1e-4, device="cpu"):
        super(LSTMDiscriminator, self).__init__()

        self.sequence_length = sequence_length
        self.device = device  # Store device information

        # 🔹 LSTM Feature Extractor
        self.lstm = nn.LSTM(input_size=state_dim + action_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True)

        # 🔹 Final Classifier (Expert vs. Agent)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of (s, a) sequence being from expert
        )

        # 🔹 Optimizer & Loss Function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        # Move model to specified device
        self.to(self.device)

    def forward(self, state_action_seq):
        """
        Forward pass through LSTM Discriminator.
        Input: (batch_size, sequence_length, state_dim + action_dim)
        """
        if not isinstance(state_action_seq, torch.Tensor):
            # Convert to Tensor
            state_action_seq = torch.tensor(state_action_seq, dtype=torch.float32)
        state_action_seq = state_action_seq.to(self.device)  # Move input to device
        
        lstm_out, _ = self.lstm(state_action_seq)  # Output: (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # Take last hidden state (batch_size, hidden_dim)
        return self.classifier(last_hidden)

    def train_discriminator(self, expert_seq, bc_seq, batch_size=32):
        """ Train Discriminator on sequences of (s, a). """
        self.train()
        self.optimizer.zero_grad()

        # Move input data to device
        expert_seq = expert_seq.to(self.device)
        bc_seq = bc_seq.to(self.device)

        # Labels: 1 for expert data, 0 for BC-generated data
        expert_labels = torch.ones((batch_size, 1), device=self.device)
        bc_labels = torch.zeros((batch_size, 1), device=self.device)

        # Forward pass through Discriminator
        expert_preds = self(expert_seq)  # Expert (should be 1)
        bc_preds = self(bc_seq)  # BC (should be 0)

        # Compute Binary Cross-Entropy Loss
        loss_expert = self.criterion(expert_preds, expert_labels)
        loss_bc = self.criterion(bc_preds, bc_labels)
        loss = loss_expert + loss_bc  # Total loss

        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def load_model(self, path: str):
        # Load a pre-trained student policy
        
        self.load_state_dict(torch.load(path, weights_only=True))
        
    def predict(self, state_action_seq, deterministic=False):
        self.eval()
        with torch.no_grad():
            prob = self.forward(state_action_seq)
            return prob