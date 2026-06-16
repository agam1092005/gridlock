import torch
import torch.nn as nn
import os
import logging

logger = logging.getLogger(__name__)

class BiGRUModel(nn.Module):
    def __init__(self, embedding_dim=768, structured_dim=50, hidden_dim=128, seq_len=5):
        super(BiGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        input_dim = embedding_dim + structured_dim
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output heads
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Scale to 0-1, then multiply by 100 later
        )
        
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus() # Ensure positive duration
        )

    def forward(self, x_seq):
        # x_seq shape: (batch, seq_len, input_dim)
        gru_out, _ = self.gru(x_seq)
        
        # Attention over seq_len
        attn_weights = self.attention(gru_out) # (batch, seq_len, 1)
        context = torch.sum(attn_weights * gru_out, dim=1) # (batch, hidden_dim * 2)
        
        severity_out = self.severity_head(context) * 100.0
        duration_out = self.duration_head(context)
        
        return severity_out, duration_out, attn_weights

class SequenceEncoder:
    def __init__(self, seq_len=5):
        self.seq_len = seq_len
        
    def build_sequence(self, current_incident, historical_incidents):
        """
        Build a sequence of incident embeddings for the BiGRU model.
        
        Args:
            current_incident: Dict with 'embedding' (768-dim) and 'structured_features' (~50-dim)
            historical_incidents: List of past incidents, most recent first, max seq_len items
            
        Returns:
            Tensor of shape (seq_len, embedding_dim + structured_dim) = (5, 818)
        """
        import torch
        
        # Ensure we have exactly seq_len items (pad with zeros if needed)
        sequence = []
        
        # Add historical incidents (oldest to newest order for temporal progression)
        if historical_incidents:
            # Reverse to get oldest first
            historical_incidents = list(reversed(historical_incidents))
            
            for incident in historical_incidents[:self.seq_len - 1]:
                # Concatenate embedding and structured features
                embedding = incident.get('embedding', torch.zeros(768))
                structured = incident.get('structured_features', torch.zeros(50))
                
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                if not isinstance(structured, torch.Tensor):
                    structured = torch.tensor(structured, dtype=torch.float32)
                
                combined = torch.cat([embedding, structured], dim=-1)
                sequence.append(combined)
        
        # Add current incident at the end
        current_embedding = current_incident.get('embedding', torch.zeros(768))
        current_structured = current_incident.get('structured_features', torch.zeros(50))
        
        if not isinstance(current_embedding, torch.Tensor):
            current_embedding = torch.tensor(current_embedding, dtype=torch.float32)
        if not isinstance(current_structured, torch.Tensor):
            current_structured = torch.tensor(current_structured, dtype=torch.float32)
        
        current_combined = torch.cat([current_embedding, current_structured], dim=-1)
        sequence.append(current_combined)
        
        # Pad with zeros if we don't have enough historical incidents
        while len(sequence) < self.seq_len:
            sequence.insert(0, torch.zeros(768 + 50))
        
        # Stack into tensor: (seq_len, 818)
        sequence_tensor = torch.stack(sequence[-self.seq_len:], dim=0)
        
        return sequence_tensor

def train_bigru(model, dataloader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_sev = nn.MSELoss()
    criterion_dur = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_seq, y_sev, y_dur in dataloader:
            optimizer.zero_grad()
            
            pred_sev, pred_dur, _ = model(x_seq)
            
            loss_sev = criterion_sev(pred_sev.squeeze(), y_sev)
            loss_dur = criterion_dur(pred_dur.squeeze(), y_dur)
            loss = loss_sev + loss_dur
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

def save_model(model, directory):
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(directory, 'bigru_model.pth'))

def load_model(model, directory):
    model.load_state_dict(torch.load(os.path.join(directory, 'bigru_model.pth')))
    model.eval()
