# Offline AnySkin baseline report

Generated: 2026-10-03T11:56:45.432853+00:00 (UTC).

All values below are computed from the public data and actual local runs. No physical robot was used.

## Data and experiment

Source: [Pollen Robotics AnySkin dataset](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection), pinned revision `d9f5e315006482635a2effa34488715871d55c13`.

Four object-disjoint sets; causal 100 Hz resampling, 50-sample input, stride 5, current-slip labels. Training-only standardisation. Matched hidden size, CPU, up to 20 epochs, validation-NLL early stopping.

| Split | Objects | Windows | Slip fraction |
|---|---:|---:|---:|
| train | 8 | 4525 | 0.158 |
| validation | 2 | 1359 | 0.259 |
| calibration | 3 | 2041 | 0.146 |
| test | 3 | 1813 | 0.210 |

Seeds: 42, 43, 44. Mean ± sample standard deviation across seeds; these are not confidence intervals.

## Held-out test results

| Model | Calibration | F1 at 0.5 | F1 at validation-selected threshold | AP | ECE | NLL |
|---|---|---:|---:|---:|---:|---:|
| gru | raw | 0.145 ± 0.023 | 0.405 ± 0.024 | 0.332 ± 0.008 | 0.153 ± 0.029 | 0.540 ± 0.028 |
| gru | temperature | 0.145 ± 0.023 | 0.405 ± 0.024 | 0.332 ± 0.008 | 0.113 ± 0.014 | 0.535 ± 0.011 |
| lstm | raw | 0.000 ± 0.000 | 0.368 ± 0.071 | 0.330 ± 0.014 | 0.117 ± 0.036 | 0.527 ± 0.008 |
| lstm | temperature | 0.000 ± 0.000 | 0.368 ± 0.071 | 0.330 ± 0.014 | 0.091 ± 0.002 | 0.523 ± 0.012 |

## Interpretation

- A constant training-prior model achieves accuracy 0.790, F1 0.000, AP 0.210, NLL 0.524. Accuracy alone is not useful evidence of successful slip detection here.
- LSTM's default-threshold predictions are dominated by the no-slip class. Both models need substantially better discrimination before robot deployment. A validation-selected threshold recovers some recall, with a precision trade-off; it is not selected on the test data.
- Mean ECE is reduced by temperature scaling in this run. This changes confidence, not the learned features or the classification decision at 0.5.
- Test NLL worsened after temperature scaling in 3 of 6 individual runs despite calibration-set NLL improving. Calibration under held-out object shift is not guaranteed.
- This experiment does not establish an overall LSTM/GRU winner. GRU has fewer parameters but the models trade off recall, ranking and calibration, and only three test objects are available.

## Runtime

Batch-1 CPU model-only latency, 20 warmups, 200 timings, two PyTorch threads. Includes a complete window forward pass; excludes sensing, preprocessing, calibration and control.

| Model | Parameters | Median latency (ms), mean across seeds |
|---|---:|---:|
| gru | 4737 | 0.481 |
| lstm | 6305 | 0.100 |

## Reproduce and inspect

Run `python -m src.download_data`, `python -m src.prepare_data`, `python -m src.run_experiment` in a new environment/project, or use `--resume` for completed local runs.
See `metrics.csv`, `summary.csv`, `per_object.csv`, `data_report.json`, and `figures/`. Per-run logits, histories, fitted temperatures and checkpoints are in `runs/final_v1/` by default (ignored by Git).

## Limits and next experiments

Overlapping windows are correlated, trial/reset identifiers are absent, and the fixed split contains only three test objects. These results are not drop-rate, force, safety or real-robot measurements.
Potential next studies include causal signal differences or baseline correction, additional object-held-out folds, richer recurrent models, and evaluation on locally collected SO-100/AnySkin data. These are hypotheses to test, not explanations already established by this experiment. Further development after seeing these test results should use a fresh holdout for final claims.

See `THIRD_PARTY.md` for source and method references. No upstream code or weights were vendored. Temperature scaling is implemented; DAC is not.
