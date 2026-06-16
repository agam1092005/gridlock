import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import logging

logger = logging.getLogger(__name__)

class STGCNModel(nn.Module):
    def __init__(self, num_nodes, in_features=3, hidden_dim=64, seq_len=24, pred_len=6):
        super(STGCNModel, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Temporal Encoder (GRU processing feature dimension per node over time)
        self.temporal_encoder = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Spatial Convolutions
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Decoder (MLP)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len),
            nn.Sigmoid() # Scale 0 to 1 for occupancy percentage
        )

    def forward(self, x, edge_index):
        """
        x: [batch, seq_len, num_nodes, in_features]
        edge_index: [2, num_edges]
        returns: [batch, pred_len, num_nodes]
        """
        batch_size = x.size(0)
        
        # Reshape for GRU: treat batch * num_nodes as batch dimension
        # x_trans: [batch, num_nodes, seq_len, in_features]
        x_trans = x.transpose(1, 2).contiguous()
        x_gru_in = x_trans.view(-1, self.seq_len, x.size(3))
        
        # temporal_out: [batch * num_nodes, seq_len, hidden_dim]
        temporal_out, _ = self.temporal_encoder(x_gru_in)
        
        # Take the last hidden state
        # h_t: [batch * num_nodes, hidden_dim]
        h_t = temporal_out[:, -1, :]
        
        # Reshape back for GCN: [batch, num_nodes, hidden_dim]
        h_t = h_t.view(batch_size, self.num_nodes, self.hidden_dim)
        
        # Apply GCN per batch item (or flatten batch if edge_index is same)
        # Assuming static graph edge_index for all items in batch
        out_spatial = []
        for b in range(batch_size):
            h_b = h_t[b] # [num_nodes, hidden_dim]
            h_b = torch.relu(self.conv1(h_b, edge_index))
            h_b = torch.relu(self.conv2(h_b, edge_index))
            h_b = torch.relu(self.conv3(h_b, edge_index))
            out_spatial.append(h_b)
            
        # out: [batch, num_nodes, hidden_dim]
        out_spatial = torch.stack(out_spatial)
        
        # Predict: [batch, num_nodes, pred_len]
        pred = self.decoder(out_spatial)
        
        # Return [batch, pred_len, num_nodes]
        return pred.transpose(1, 2).contiguous() * 100.0 # scale to 0-100%
