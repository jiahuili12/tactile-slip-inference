"""One shared set of endpoints; object-disjoint data and train-only scaling."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.src.utils import project_path, save_json

CHANNELS = [f"mag{i}_{axis}" for i in range(1, 6) for axis in "xyz"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_splits(splits):
    if set(splits) != {"train", "calibration", "test"}:
        raise ValueError("Exactly train/calibration/test splits are required")
    seen = set()
    for files in splits.values():
        if not files:
            raise ValueError("Empty split")
        for name in files:
            if name != Path(name).name or name in seen or name == "no_slip.csv":
                raise ValueError(f"Invalid, mixed-object or duplicate recording: {name}")
            seen.add(name)


def resample_segments(times, values, labels, hz=100, max_gap=.1):
    cuts = np.flatnonzero((np.diff(times) <= 0) | (np.diff(times) > max_gap)) + 1
    result = []
    for ids in np.split(np.arange(len(times)), cuts):
        if len(ids) < 2:
            continue
        t = times[ids]
        grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * hz)) + 1) / hz
        selected = np.searchsorted(t, grid, side="right") - 1
        result.append((values[ids[selected]], labels[ids[selected]], grid))
    return result


def read_recording(path, hz, max_gap):
    frame = pd.read_csv(path)
    values = frame[CHANNELS].to_numpy(dtype=np.float32)
    target = frame.slip.astype(str).str.lower().str.strip().map({"0": 0, "1": 1, "false": 0, "true": 1})
    if not np.isfinite(values).all() or target.isna().any():
        raise ValueError(f"Invalid values/labels in {path}")
    nanos = pd.to_datetime(frame.log_time, errors="raise").astype("int64").to_numpy()
    times = (nanos - nanos[0]).astype(np.float64) / 1e9
    return resample_segments(times, values, target.to_numpy(dtype=np.int64), hz, max_gap)


def make_windows(segments, max_window, stride, object_id):
    xs, ys, ids = [], [], []
    for segment, (values, labels, _) in enumerate(segments):
        for end in range(max_window - 1, len(values), stride):
            xs.append(values[end - max_window + 1:end + 1])
            ys.append(labels[end])
            ids.append(f"{object_id}:{segment}:{end}")
    if not xs:
        raise ValueError(f"No complete windows for {object_id}")
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(ids)


def select_history(windows, length):
    if length < 1 or length > windows.shape[1]:
        raise ValueError("Invalid history length")
    return np.ascontiguousarray(windows[:, -length:, :])


def prepare(config):
    splits = config["splits"]
    validate_splits(splits)
    raw, output = project_path(config["raw_dir"]), project_path(config["prepared_dir"])
    source = json.loads(project_path(config["source_manifest"]).read_text("utf-8"))
    expected = {entry["file"]: entry["sha256"] for entry in source["files"]}
    hashes = {name: digest(raw / name) for names in splits.values() for name in names}
    if any(hashes[name] != expected.get(name) for name in hashes):
        raise ValueError("Raw data do not match the pinned download manifest")
    specification = {"splits": splits, "raw_hashes": hashes, "hz": config["hz"],
                     "max_window": max(config["sequence_lengths"]), "stride": config["stride"],
                     "max_gap_seconds": config["max_gap_seconds"], "dataset_revision": source["revision"]}
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    if output.exists():
        report = json.loads((output / "report.json").read_text("utf-8"))
        if report["preprocessing_signature"] != signature:
            raise ValueError("Prepared data use another protocol; choose a new prepared_dir")
        for name, checksum in report["prepared_sha256"].items():
            if digest(output / name) != checksum:
                raise ValueError(f"Prepared file changed: {name}")
        return report

    recordings = {name: read_recording(raw / name, config["hz"], config["max_gap_seconds"])
                  for name in hashes}
    # Each unique resampled training row contributes once, not once per window.
    rows = np.concatenate([segment[0] for name in splits["train"] for segment in recordings[name]])
    mean = rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(rows.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    output.mkdir(parents=True)
    np.savez(output / "scaler.npz", mean=mean, std=std)
    report = {**specification, "preprocessing_signature": signature, "channels": CHANNELS,
              "scaler_fit_on": "unique resampled TRAIN rows only", "scaler_fit_rows": len(rows),
              "label": "current slip at the common endpoint", "excluded": ["no_slip.csv"], "counts": {}}
    for split, files in splits.items():
        xs, ys, objects, endpoint_ids = [], [], [], []
        for name in files:
            obj = Path(name).stem
            x, y, ids = make_windows(recordings[name], specification["max_window"], config["stride"], obj)
            xs.append(((x - mean) / std).astype(np.float32))
            ys.append(y)
            objects.extend([obj] * len(y))
            endpoint_ids.append(ids)
        x, y, ids = np.concatenate(xs), np.concatenate(ys), np.concatenate(endpoint_ids)
        if np.unique(y).size != 2:
            raise ValueError(f"Split {split} is missing a class")
        np.savez_compressed(output / f"{split}.npz", x=x, y=y,
                            object_id=np.asarray(objects), endpoint_id=ids)
        report["counts"][split] = {"objects": len(files), "windows": len(y), "slip_fraction": float(y.mean())}
    report["prepared_sha256"] = {name: digest(output / name) for name in
                                 ["train.npz", "calibration.npz", "test.npz", "scaler.npz"]}
    save_json(output / "report.json", report)
    return report
