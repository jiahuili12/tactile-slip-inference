"""Optional parity test against the reviewed, hash-pinned upstream source.

No network calls. See pollen_reproduction/README.md to populate the ignored cache.
The fixture is synthetic and does not measure slip-detection performance.
"""
import hashlib
import importlib.util
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.src.utils import ROOT
from pollen_reproduction.src.pollen_data import prepare_arrays
from pollen_reproduction.src.reproduce_pollen import train_reproduction

HASHES = {
    "lstm.py": "792ac2f7e99284f0fd708b9b2c67227aa19987549a82f16835855fbe6620d579",
    "train.py": "38614d2a4003176353a67d9fc04380f6ec85c7843fb48d09646bbf1b07a28727",
}


def test_training_matches_reviewed_upstream_code(tmp_path, monkeypatch):
    reference = ROOT / ".cache/pollen_reference"
    modules = {}
    for filename, expected_hash in HASHES.items():
        path = reference / filename
        if not path.is_file():
            pytest.skip("Optional pinned Pollen source snapshot is not cached")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    for name in ["lstm", "train"]:
        spec = importlib.util.spec_from_file_location(name, reference / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module

    rng = np.random.default_rng(123)
    x, y, train, selection, _ = prepare_arrays(rng.normal(size=(160, 15)), np.tile([0, 1], 80))
    torch.set_num_threads(1)
    torch.manual_seed(19)
    original = modules["lstm"].LSTMNet(15, 128, 128, 1, 1)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train])),
                              batch_size=32, shuffle=True)
    selection_loader = DataLoader(TensorDataset(torch.from_numpy(x[selection]), torch.from_numpy(y[selection])),
                                  batch_size=32, shuffle=False)
    modules["train"].train(train_loader, selection_loader, original,
                            torch.nn.BCEWithLogitsLoss(), torch.optim.Adam(original.parameters(), lr=.001),
                            "cpu", epochs=2, checkpoint_path=str(tmp_path / "upstream.pt"))
    local = tmp_path / "local"
    local.mkdir()
    train_reproduction(local, x, y, train, selection, seed=19, epochs_per_phase=2)
    original_state = torch.load(tmp_path / "upstream.pt", weights_only=True)
    local_state = torch.load(local / "phase2_best.pt", weights_only=True)["state_dict"]
    assert original_state.keys() == local_state.keys()
    for name in original_state:
        torch.testing.assert_close(original_state[name], local_state[name], rtol=0, atol=0)
