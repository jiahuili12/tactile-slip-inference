# tactile-slip-inference

Compare compact LSTM and GRU slip detectors on public AnySkin time series, then assess scalar temperature calibration on unseen objects. This is an offline baseline for later tactile-control experiments.

## Start in VS Code / Windows PowerShell

Open this repository folder and select `.venv/Scripts/python.exe` using **Python: Select Interpreter**.
An existing local `.venv` can be used directly. To create a new environment with your installed Python:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run commands from the repository root. Activation is optional: calling the environment's Python explicitly avoids PowerShell execution-policy issues.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.download_data
.\.venv\Scripts\python.exe -X utf8 -m src.prepare_data
.\.venv\Scripts\python.exe -X utf8 -m pytest -q --basetemp .cache/pytest
.\.venv\Scripts\python.exe -X utf8 -m src.run_experiment
```

The default experiment runs both models for up to 20 epochs, with validation early stopping and seeds 42, 43 and 44. It uses CPU and two PyTorch threads. No cloud compute, GitHub upload or robot connection is involved.

For a separate quick end-to-end check:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.run_experiment --seeds 99 --epochs 2 --output-dir runs/quick --results-dir results/quick
```

Existing completed run directories are protected against silent retraining. `--resume` reuses completed model checkpoints and repeats calibration/evaluation/aggregation. For a new experiment use a new output directory. Do not change preprocessing underneath existing checkpoints.

## Individual steps

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs/lstm.yaml --seed 42
.\.venv\Scripts\python.exe -m src.calibrate --run runs/lstm_seed42
.\.venv\Scripts\python.exe -m src.evaluate --run runs/lstm_seed42
```

Use `configs/gru.yaml` for GRU. The separate-step config default saves to `runs/`; the comparison pipeline saves to `runs/final_v1/`.

## Experimental protocol

- Public data: [Pollen Robotics AnySkin slip detection](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection), revision pinned in the downloader.
- Four disjoint object groups; see `data/splits.json`. The test comprises three unseen object recordings.
- Fifteen magnetic channels, causal 100 Hz resampling, 50-sample windows, stride 5, current-slip label at window end. No bidirectional network or future label input.
- Training-only channel standardisation; unweighted BCE; Adam; identical hidden size (32), one recurrent layer, batch size (128), learning rate and epoch budget. LSTM and GRU have different parameter counts, which are reported; this is a matched-hidden-size, not matched-parameter-count comparison.
- Validation NLL chooses the checkpoint. A positive temperature in [0.05,20] is fitted only on calibration NLL. Optional F1 thresholds are chosen only on validation.
- Main classification metrics use threshold 0.5. ECE is top-label confidence ECE with 15 equal-width bins; NLL and Brier score also assess probability quality. PR-AUC is reported as average precision (`pr_auc_ap`).
- Scalar temperature leaves ranking and decisions at probability 0.5 unchanged. It can improve or worsen held-out calibration: calibration-set improvement is not a guarantee under object shift.
- CPU latency uses a complete window, batch size 1, 20 warmups and 200 timings. It excludes data acquisition, preprocessing, calibration and actuation; it is not a robot control-frequency measurement.

## Files and outputs

```text
configs/               matched architecture/training configurations
data/splits.json        fixed object split
data/source_manifest.json  download revision and per-file SHA-256 hashes
src/download_data.py   download original data and card
src/prepare_data.py    causal resampling, scaler and windows
src/dataset.py         PyTorch dataset
src/models.py          LSTM and GRU definitions
src/train.py           shared training/early stopping
src/calibrate.py       held-out temperature fit
src/metrics.py         metrics and risk–coverage
src/evaluate.py        held-out predictions and latency
src/plot_results.py    aggregation and diagnostic plots
src/run_experiment.py  six-run comparison
src/report_results.py readable report and constant-prior baseline
tests/                 leakage, causality, calibration and integration checks
results/               real metrics, data report and figures
```

`runs/` stores checkpoint weights, training histories, temperature fits and per-window test probabilities. It is ignored by Git. `results/metrics.csv` records each run; `summary.csv` reports mean and sample standard deviation across seeds (not statistical confidence intervals); `per_object.csv` exposes performance differences between test objects. Raw data, processed tensors, environment and caches are ignored by Git.

## Limits and interpretation

The public data were collected by Pollen Robotics on a Reachy 2 gripper. Offline performance does not establish performance on SO-100/SO-101, reduced object drop rates, future slip prediction, calibrated force estimation or physical safety. Those require separate robot data and controlled trials.

Window errors are temporally correlated. This one fixed split and three test objects are an initial baseline, not a statistically definitive architecture ranking. The provider's mixed-object `no_slip.csv` is excluded because object membership cannot be assigned to disjoint splits. Exact trial/reset identifiers are not supplied; preprocessing only detects timestamp gaps/resets.

See [data documentation](data/README.md), [sources and method citations](THIRD_PARTY.md), and the generated `results/REPORT.md` for actual results. AnySkin and Pollen's reported numbers use different protocols and are not directly comparable to this baseline. Temperature scaling is implemented; DAC and hardware control are future extensions.

Project code uses the MIT license. Third-party data/dependencies retain their respective licenses.
