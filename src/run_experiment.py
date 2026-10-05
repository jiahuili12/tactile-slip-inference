"""Run matched LSTM/GRU experiments, calibrate and aggregate three seeds."""
import argparse

from .calibrate import calibrate
from .common import load_config, project_path, run_dir
from .evaluate import evaluate
from .plot_results import aggregate
from .report_results import report
from .train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, help="Override both configs equally")
    parser.add_argument("--output-dir", default="runs/final_v1")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--resume", action="store_true", help="Reuse completed checkpoints")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be unique")
    runs = []
    for model in ("lstm", "gru"):
        config = load_config(f"configs/{model}.yaml")
        config["output_dir"] = args.output_dir
        if args.epochs is not None:
            config["epochs"] = args.epochs
        for seed in args.seeds:
            output = run_dir(config, seed)
            if not (args.resume and (output / "best.pt").exists() and (output / "training_summary.json").exists()):
                train(config, seed)
            calibrate(output)
            evaluate(output)
            runs.append(output)
    aggregate(runs, args.results_dir)
    report(args.results_dir, config["processed_dir"])


if __name__ == "__main__":
    main()
