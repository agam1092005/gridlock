import os
import logging
import torch
from src.models.module_b.trainer import STGCNTrainer
from src.models.module_b.stgcn_model import STGCNModel
from src.models.module_b.graph_loader import GraphLoader
from src.models.module_b.dataset import prepare_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing Module B graph loader...")
    loader = GraphLoader()
    # It will use the real coordinates bounds over Bengaluru
    graph = loader.load_osm_graph()

    logger.info(f"Loaded graph with {graph.num_nodes} nodes and {graph.edge_index.size(1)} edges.")

    # Use CPU or GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloaders return dummy data right now, which is fine for the skeleton
    logger.info("Preparing dataloaders...")
    train_loader, val_loader, _ = prepare_dataloaders(num_nodes=graph.num_nodes)

    logger.info("Initializing STGCN model...")
    model = STGCNModel(num_nodes=graph.num_nodes)
    trainer = STGCNTrainer(model, graph.edge_index, device=device)

    epochs = 2
    for epoch in range(epochs):
        logger.info(f"Starting Epoch {epoch+1}/{epochs}...")
        t_loss = trainer.train_epoch(train_loader)
        v_loss = trainer.validate(val_loader)
        logger.info(f"Epoch {epoch+1} Completed: Train Loss={t_loss:.4f}, Val Loss={v_loss:.4f}")

    model_path = "models/artifacts/module_b/stgcn_model.pth"
    trainer.save(model_path)
    logger.info(f"Saved trained STGCN model to {model_path}")


if __name__ == "__main__":
    main()
