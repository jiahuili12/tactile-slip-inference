"""Small causal recurrent baselines with the same input/output interface."""
import torch
from torch import nn


class SlipRNN(nn.Module):
    def __init__(self, model="lstm", hidden_size=32, num_layers=1):
        super().__init__()
        if model not in {"lstm", "gru"}:
            raise ValueError("model must be lstm or gru")
        rnn_type = nn.LSTM if model == "lstm" else nn.GRU
        self.rnn = rnn_type(15, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, return_features=False):
        output, _ = self.rnn(x)
        features = output[:, -1]
        logits = self.head(features).squeeze(-1)
        return (logits, features) if return_features else logits


def build_model(config):
    return SlipRNN(config["model"], config["hidden_size"], config["num_layers"])


def load_model(checkpoint):
    # Only use checkpoints produced by this project.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["config"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved
