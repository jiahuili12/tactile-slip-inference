"""Load every pinned CSV with the upstream loader; reproduce its row split.

WARNING: all-data scaling and random-row splitting are intentional here.
They reproduce Pollen's public script, not a recommended deployment evaluation.
"""
import hashlib
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.src.utils import ROOT, save_json
from common.src.download_data import DATASET, REVISION

CHANNELS = [f"mag{i}_{axis}" for i in range(1, 6) for axis in "xyz"]


def prepare_arrays(features, labels):
    """Preserve upstream float64 scaling, float32 tensors, and split order."""
    scaler = StandardScaler().fit(features)
    x = scaler.transform(features).astype(np.float32)[:, None, :]
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    train_ids, test_ids = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42
    )
    return x, y, train_ids, test_ids, scaler


def load_data():
    """Use the actual HF Hub loader and cross-check against local file hashes.

    The first call requires internet access; subsequent calls use the HF cache.
    No credentials, dataset scripts, pretrained weights or robot access needed.
    """
    from datasets import load_dataset

    cache = ROOT / ".cache/huggingface"
    dataset = load_dataset(
        DATASET, revision=REVISION, data_dir="data", cache_dir=str(cache), token=False
    )["train"]
    # Upstream converts Python column lists to np.array: values already have
    # the dataset card's float32 quantisation, then become float64 in NumPy.
    features = np.array([list(dataset[name]) for name in CHANNELS]).T
    labels = np.asarray(list(dataset["slip"]), dtype=np.float32)

    # Determine the actual HF file order by comparing every row with the
    # previously downloaded, SHA-256-verified local CSVs (including no_slip).
    import pandas as pd

    manifest = json.loads((ROOT / "common/data/source_manifest.json").read_text("utf-8"))
    if manifest["revision"] != REVISION:
        raise ValueError("Local data manifest has a different revision; rerun common.src.download_data")
    remaining = {}
    for entry in manifest["files"]:
        path = ROOT / "common/data/raw" / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Local data checksum mismatch: {path.name}")
        frame = pd.read_csv(path)
        remaining[path.name] = (
            frame[CHANNELS].to_numpy().astype(np.float32).astype(np.float64),
            frame["slip"].to_numpy(dtype=np.float32),
        )
    objects = np.empty(len(labels), dtype="U32")
    file_rows, offset = [], 0
    while remaining:
        matched = None
        for name, (values, target) in remaining.items():
            end = offset + len(target)
            if (np.array_equal(features[offset:end], values)
                    and np.array_equal(labels[offset:end], target)):
                matched = name
                break
        if matched is None:
            raise ValueError("HF data/order does not match the pinned local CSVs")
        _, target = remaining.pop(matched)
        end = offset + len(target)
        objects[offset:end] = matched.removesuffix(".csv")
        file_rows.append({"file": matched, "rows": len(target), "start_row": offset})
        offset = end
    if offset != len(labels):
        raise ValueError("Unmatched HF rows")

    x, y, train_ids, test_ids, scaler = prepare_arrays(features, labels)
    provenance = {
        "dataset": DATASET, "revision": REVISION,
        "hf_fingerprint": dataset._fingerprint,
        "loader": "datasets.load_dataset from the pinned HF Hub revision",
        "rows": len(y), "channels": CHANNELS, "sequence_length": 1,
        "train_rows": len(train_ids), "selection_rows": len(test_ids),
        "train_slip_fraction": float(y[train_ids].mean()),
        "selection_slip_fraction": float(y[test_ids].mean()),
        "file_order_verified_against_local_csvs": True,
        "files_in_hf_order": file_rows,
        "source_files": manifest["files"],
        "array_sha256": hashlib.sha256(features.tobytes() + labels.tobytes()).hexdigest(),
        "train_index_sha256": hashlib.sha256(train_ids.astype("<i8").tobytes()).hexdigest(),
        "selection_index_sha256": hashlib.sha256(test_ids.astype("<i8").tobytes()).hexdigest(),
        "scaler_fit_on": "ALL rows, intentionally following upstream (leakage)",
        "split": "train_test_split(test_size=0.2, random_state=42), no stratification",
        "selection_is_independent_test": False,
    }
    return x, y, train_ids, test_ids, objects, scaler, provenance


def save_data_audit(run, train_ids, test_ids, scaler, provenance):
    np.savez(run / "split_indices.npz", train=train_ids, selection=test_ids)
    save_json(run / "scaler.json", {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    save_json(run / "data_audit.json", provenance)
