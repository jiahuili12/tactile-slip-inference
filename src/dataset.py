"""Prepared [window, time, 15 channels] data."""
import numpy as np
import torch
from torch.utils.data import Dataset


class SlipDataset(Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as arrays:
            self.x = torch.from_numpy(arrays["x"].copy())
            self.y = torch.from_numpy(arrays["y"].astype(np.float32))
            self.object_ids = arrays["object_id"].copy()
        if self.x.ndim != 3 or self.x.shape[2] != 15 or len(self.x) != len(self.y):
            raise ValueError("Expected aligned [N,T,15] sensor windows and labels")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
