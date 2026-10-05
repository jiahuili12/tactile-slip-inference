# Public data and local layout

Source: [Pollen Robotics AnySkin slip dataset](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection).
Pinned revision: `d9f5e315006482635a2effa34488715871d55c13`.
The dataset card declares Apache-2.0. The provider collected these recordings on a Reachy 2 gripper, not an SO-100 or SO-101.

`python -m src.download_data` downloads 17 original CSVs and the upstream dataset card into ignored `raw/`; it writes exact source URLs and SHA-256 hashes to `source_manifest.json`.
No authentication or Hugging Face token is required for this public dataset.

Each CSV has `log_time`, 15 channels (`mag1_x` through `mag5_z`), and a binary `slip` label.
The upstream annotations are manually labelled slip events; these are not force ground truth.

`splits.json` assigns complete object recordings to train (8 objects), validation (2), calibration (3), and test (3).
`no_slip.csv` is downloaded but excluded: it mixes unidentified objects, so assigning it safely to an object-disjoint split is not possible with the available metadata.
The split is fixed in advance and not selected to optimise test performance. It measures generalisation to these three held-out object recordings, not the entire population of objects.

Preprocessing breaks sequences at time resets or gaps longer than 100 ms, causally resamples to 100 Hz using the last available observation, and constructs 50-sample windows at stride 5. Resampling does not mean the original hardware sampled uniformly at 100 Hz; see `results/data_report.json` for measured intervals.
The label is the last observation's current slip label. The method is detection, not advance slip prediction. No future information is used.
Mean/std are fitted on unique resampled training rows only. Validation chooses early stopping and optional F1 thresholds; calibration fits temperature; test is used only for final evaluation.

Generated arrays and scaler are in ignored `processed/`. Adjacent windows overlap and their errors are correlated. Window counts are not independent physical trial counts. The published recordings do not provide explicit trial/reset identifiers; the timestamp gap rule cannot guarantee separation of all grasp episodes inside a recording.
