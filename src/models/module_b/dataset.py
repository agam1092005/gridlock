import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging

logger = logging.getLogger(__name__)


class STGCNDataset(Dataset):
    def __init__(self, data_tensor, seq_len=24, pred_len=6):
        """
        data_tensor: [num_timesteps, num_nodes, num_features]
        seq_len: historical lookback window
        pred_len: prediction horizon
        """
        self.data = data_tensor
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_timesteps = data_tensor.shape[0]

    def __len__(self):
        return self.num_timesteps - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]  # [seq_len, num_nodes, num_features]
        y = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_len
        ]  # [pred_len, num_nodes, num_features]

        # Typically, we just want to predict the first feature (occupancy)
        y = y[:, :, 0]  # [pred_len, num_nodes]
        return x, y


def prepare_dataloaders(num_nodes=100, num_timesteps=1000, batch_size=32):
    """
    Generates synthetic spatial-temporal dataset and returns loaders.
    """
    logger.info("Generating synthetic spatial-temporal dataset.")
    # Features: [occupancy (target), time_of_day, day_of_week]
    data_tensor = torch.rand((num_timesteps, num_nodes, 3), dtype=torch.float)

    # Chronological Split: 60/20/20
    train_split = int(0.6 * num_timesteps)
    val_split = int(0.8 * num_timesteps)

    train_data = data_tensor[:train_split]
    val_data = data_tensor[train_split:val_split]
    test_data = data_tensor[val_split:]

    train_dataset = STGCNDataset(train_data)
    val_dataset = STGCNDataset(val_data)
    test_dataset = STGCNDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
