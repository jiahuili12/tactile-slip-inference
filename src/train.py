"""Train either recurrent baseline; select checkpoints by validation NLL."""
import argparse
import hashlib
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import load_config, project_path, run_dir, save_json, seed_everything
from .dataset import SlipDataset
from .models import build_model


@torch.inference_mode()
def predict(model, dataset, batch_size=256):
    outputs = []
    model.eval()
    for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        outputs.append(model(x).numpy())
    return np.concatenate(outputs)


def train(config, seed):
    seed_everything(seed, config.get("threads", 2))
    output = run_dir(config, seed)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists():
        raise FileExistsError(f"Run already exists: {output}; choose a new output_dir or seed")
    config = dict(config, seed=seed)
    save_json(output / "config.json", config)
    processed = project_path(config["processed_dir"])
    training = SlipDataset(processed / "train.npz")
    validation = SlipDataset(processed / "validation.npz")
    generator = torch.Generator().manual_seed(seed)
    batches = DataLoader(training, batch_size=config["batch_size"], shuffle=True, generator=generator)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()  # Unweighted NLL keeps probabilities interpretable.
    best, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    fingerprint = hashlib.sha256((processed / "report.json").read_bytes()).hexdigest()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for x, y in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
        logits = predict(model, validation)
        val_loss = float(np.mean(np.logaddexp(0, logits.astype(float)) - validation.y.numpy() * logits))
        history.append({"epoch": epoch, "train_nll": total / len(training), "validation_nll": val_loss})
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(f"{config['model']} seed={seed} epoch={epoch}: train={total / len(training):.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            torch.save({"config": config, "state_dict": model.state_dict(), "epoch": epoch,
                        "validation_nll": best, "data_fingerprint": fingerprint}, output / "best.pt")
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    save_json(output / "training_summary.json", {
        "epochs_run": len(history), "best_validation_nll": best,
        "training_seconds": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": "cpu", "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "threads": torch.get_num_threads(), "data_fingerprint": fingerprint,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, args.seed if args.seed is not None else config["seed"])


if __name__ == "__main__":
    main()
