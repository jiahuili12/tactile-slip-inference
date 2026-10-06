"""Pollen's published architecture, expressed separately from our small baselines.

Protocol reference (Apache-2.0): pollen-robotics/anyskin-slip-detection,
commit 24a3728c35985def720ff292a8a45ef7e8662a64, src/lstm.py.
Layer names/shapes are compatible with the upstream state dictionary.
"""
import torch
from torch import nn


class PollenLSTM(nn.Module):
    """15 -> LSTM(128) -> Linear(128,128) -> Linear(128,1); no activation."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(15, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        # Omitting h0/c0 uses the same zero initial states as upstream.
        sequence, _ = self.lstm(x)
        return self.out(self.fc(sequence[:, -1, :]))
