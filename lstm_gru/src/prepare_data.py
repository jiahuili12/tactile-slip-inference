"""Object-disjoint splits, causal resampling, training-only normalisation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.src.utils import ROOT, project_path, save_json

CHANNELS = [f"mag{i}_{axis}" for i in range(1, 6) for axis in "xyz"]
SPLITS = ("train", "validation", "calibration", "test")


def validate_splits(splits):
    if set(splits) != set(SPLITS):
        raise ValueError(f"Expected splits: {SPLITS}")
    seen = set()
    for name in SPLITS:
        if not splits[name]:
            raise ValueError(f"Empty split: {name}")
        for filename in splits[name]:
            if filename in seen:
                raise ValueError(f"File occurs in multiple splits: {filename}")
            if filename != Path(filename).name:
                raise ValueError("Split manifest must contain plain filenames")
            seen.add(filename)


def causal_resample(times, values, labels, hz=100, max_gap=0.1):
    """Never read a measurement from after the requested grid time.

    A gap or time reset starts a new segment. No window may span segments.
    """
    if len(times) < 2 or hz <= 0 or max_gap <= 0:
        raise ValueError("Need >=2 observations and positive hz/max_gap")
    cuts = np.flatnonzero((np.diff(times) <= 0) | (np.diff(times) > max_gap)) + 1
    segments = []
    for ids in np.split(np.arange(len(times)), cuts):
        if len(ids) < 2:
            continue
        t = times[ids]
        grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * hz)) + 1) / hz
        indexes = np.searchsorted(t, grid, side="right") - 1
        segments.append((values[ids[indexes]], labels[ids[indexes]], grid))
    return segments


def read_recording(path, hz, max_gap):
    frame = pd.read_csv(path)
    required = CHANNELS + ["log_time", "slip"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing columns in {path.name}")
    values = frame[CHANNELS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite sensor values in {path.name}")
    raw_labels = frame["slip"].astype(str).str.lower().str.strip()
    mapped = raw_labels.map({"0": 0, "1": 1, "false": 0, "true": 1})
    if mapped.isna().any():
        raise ValueError(f"Invalid slip labels in {path.name}")
    stamps = pd.to_datetime(frame["log_time"], errors="raise")
    nanos = stamps.astype("int64").to_numpy()
    times = (nanos - nanos[0]).astype(np.float64) / 1e9
    intervals = np.diff(times)
    positive = intervals[intervals > 0]
    stats = {"raw_rows": len(frame), "raw_positive_fraction": float(mapped.mean()),
             "median_raw_interval_ms": float(np.median(positive) * 1000) if len(positive) else None,
             "gaps_or_resets": int(((intervals <= 0) | (intervals > max_gap)).sum())}
    return causal_resample(times, values, mapped.to_numpy(dtype=np.int64), hz, max_gap), stats


def window_segments(segments, window, stride):
    xs, ys, ts, segment_ids = [], [], [], []
    for segment_id, (values, labels, times) in enumerate(segments):
        for end in range(window - 1, len(values), stride):
            xs.append(values[end - window + 1:end + 1])
            ys.append(labels[end])  # Detect current slip; never claim future prediction.
            ts.append(times[end])
            segment_ids.append(segment_id)
    if not xs:
        raise ValueError("No windows: recording too short or gaps too frequent")
    return np.stack(xs), np.asarray(ys), np.asarray(ts), np.asarray(segment_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="common/data/raw")
    parser.add_argument("--output", default="lstm_gru/data/processed")
    parser.add_argument("--splits", default="lstm_gru/data/splits.json")
    parser.add_argument("--hz", type=int, default=100)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-gap", type=float, default=0.1)
    args = parser.parse_args()
    if args.window < 2 or args.stride < 1:
        raise ValueError("window>=2 and stride>=1 required")
    split_path = project_path(args.splits)
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    validate_splits(splits)
    raw = project_path(args.raw)
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    recordings, source_stats = {}, {}
    for filename in [f for name in SPLITS for f in splits[name]]:
        recordings[filename], source_stats[filename] = read_recording(raw / filename, args.hz, args.max_gap)
    # Each training observation appears once in the scaler fit, not once per overlapping window.
    training_rows = np.concatenate([s[0] for f in splits["train"] for s in recordings[f]])
    mean = training_rows.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = training_rows.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    np.savez(output / "scaler.npz", mean=mean, std=std)
    report = {"hz": args.hz, "window": args.window, "stride": args.stride,
              "max_gap_seconds": args.max_gap, "label": "slip_at_window_end",
              "channels": CHANNELS, "splits": {}, "source_stats": source_stats,
              "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
              "scaler_fit_rows": len(training_rows),
              "input_sha256": {f: hashlib.sha256((raw / f).read_bytes()).hexdigest() for f in recordings},
              "excluded_files": sorted(p.name for p in raw.glob('*.csv') if p.name not in recordings)}
    manifest_path = ROOT / "common/data/source_manifest.json"
    if manifest_path.exists():
        report["source_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for name in SPLITS:
        x_all, y_all, ids_all, times_all, seg_all = [], [], [], [], []
        for filename in splits[name]:
            x, y, times, segment_ids = window_segments(recordings[filename], args.window, args.stride)
            x_all.append(((x - mean) / std).astype(np.float32))
            y_all.append(y)
            ids_all.extend([filename.removesuffix('.csv')] * len(y))
            times_all.append(times)
            seg_all.append(segment_ids)
        x, y = np.concatenate(x_all), np.concatenate(y_all)
        np.savez_compressed(output / f"{name}.npz", x=x, y=y,
                            object_id=np.asarray(ids_all), time=np.concatenate(times_all),
                            segment_id=np.concatenate(seg_all))
        report["splits"][name] = {"files": splits[name], "windows": len(y),
                                  "positive_windows": int(y.sum()), "positive_fraction": float(y.mean())}
        if np.unique(y).size != 2:
            raise ValueError(f"Split {name} lacks a class; revise manifest before training")
        print(f"{name}: {len(y):,} windows, slip fraction={y.mean():.3f}", flush=True)
    save_json(output / "report.json", report)
    save_json(ROOT / "lstm_gru/results/data_report.json", report)


if __name__ == "__main__":
    main()
