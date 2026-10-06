"""Match hidden width and head; architecture and visible history are factors."""
from torch import nn


class SlipModel(nn.Module):
    def __init__(self, architecture, hidden_size=128):
        super().__init__()
        if architecture not in {"lstm", "gru"}:
            raise ValueError("architecture must be lstm or gru")
        layer = nn.LSTM if architecture == "lstm" else nn.GRU
        self.rnn = layer(15, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # No hidden-state carry between windows and no future input.
        history, _ = self.rnn(x)
        return self.out(self.fc(history[:, -1])).squeeze(-1)
