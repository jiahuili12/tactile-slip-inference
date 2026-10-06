"""Shared paths, configuration and reproducible experiment utilities."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    # Original checkpoints contain this old relative path. Keep their bytes
    # and provenance untouched while making evaluation work after relocation.
    if path == Path("data/processed"):
        path = Path("lstm_gru/data/processed")
    return path if path.is_absolute() else ROOT / path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    return yaml.safe_load(project_path(path).read_text(encoding="utf-8"))


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def run_dir(config, seed):
    return project_path(config["output_dir"]) / f"{config['model']}_seed{seed}"
