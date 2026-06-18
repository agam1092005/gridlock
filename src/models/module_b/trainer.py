import torch
import torch.nn as nn
import os
import logging
from .stgcn_model import STGCNModel
from .dataset import prepare_dataloaders
from .graph_loader import GraphLoader

logger = logging.getLogger(__name__)


class STGCNTrainer:
    def __init__(self, model, edge_index, device="cpu"):
        self.model = model.to(device)
        self.edge_index = edge_index.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def inject_incident_impact(self, x, incident_node, severity_score):
        """
        Dynamically injects an incident impact spike into the feature tensor
        at the specific node to inform the STGCN.
        x: [batch, seq_len, num_nodes, in_features]
        """
        # Apply spike to the last timestep of the sequence for the affected node
        # Feature 0 is occupancy. Scale impact by severity (0-100)
        spike_val = min(1.0, severity_score / 100.0 + 0.2)
        x[:, -1, incident_node, 0] = spike_val
        return x

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            # Simulate random incident injections for training robustness
            if torch.rand(1).item() > 0.5:
                # Pick a random node and inject impact
                inc_node = torch.randint(0, self.model.num_nodes, (1,)).item()
                sev = torch.randint(40, 100, (1,)).item()
                x = self.inject_incident_impact(x, inc_node, sev)

            self.optimizer.zero_grad()
            pred = self.model(x, self.edge_index)

            loss = self.criterion(pred, y * 100.0)  # scale y back to 0-100 for loss if needed
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x, self.edge_index)
                loss = self.criterion(pred, y * 100.0)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = GraphLoader()
    graph = loader.load_osm_graph()

    train_loader, val_loader, _ = prepare_dataloaders(num_nodes=graph.num_nodes)

    model = STGCNModel(num_nodes=graph.num_nodes)
    trainer = STGCNTrainer(model, graph.edge_index)

    for epoch in range(2):
        t_loss = trainer.train_epoch(train_loader)
        v_loss = trainer.validate(val_loader)
        logger.info(f"Epoch {epoch}: Train Loss={t_loss:.4f}, Val Loss={v_loss:.4f}")

    trainer.save("models/artifacts/module_b/stgcn_model.pth")
