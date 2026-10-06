# Sources and acknowledgements

## Data actually used

- Pollen Robotics, **AnySkin Slip Detection Dataset**: https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection
- Revision: `d9f5e315006482635a2effa34488715871d55c13`; provider-declared license: Apache-2.0.
- Original collection: Reachy 2 gripper, 16 objects and an additional mixed-object static-contact recording.
- Data are downloaded locally, not committed to this repository. Exact file URLs, hashes and download time are recorded in [data/source_manifest.json](data/source_manifest.json). The upstream dataset card is saved in `common/data/raw/UPSTREAM_README.md` without modification.

## Pollen training-protocol reproduction (second experiment)

- Pollen Robotics **anyskin-slip-detection**, Apache-2.0: https://github.com/pollen-robotics/anyskin-slip-detection
- Pinned source commit: `24a3728c35985def720ff292a8a45ef7e8662a64` (2025-03-12).
- Architecture reference: https://github.com/pollen-robotics/anyskin-slip-detection/blob/24a3728c35985def720ff292a8a45ef7e8662a64/src/lstm.py
- Training reference: https://github.com/pollen-robotics/anyskin-slip-detection/blob/24a3728c35985def720ff292a8a45ef7e8662a64/src/train.py
- Publisher's reported 0.9826 accuracy: https://huggingface.co/pollen-robotics/anyskin-slip-detection/blob/main/README.md
- Local implementation: `pollen_reproduction/src/pollen_model.py`, `pollen_reproduction/src/pollen_data.py`, `pollen_reproduction/src/reproduce_pollen.py`, `pollen_reproduction/src/pollen_report.py`. These implement the published architecture/protocol with CPU support, pinned data, reproducible initialization, auditing and additional reporting. They are not a new proposed architecture.
- Upstream source files and pretrained weights are not vendored. The public code's two training loops, all-data scaler, single-step input and selection-set checkpointing are explicitly retained. See [protocol and adaptations](../pollen_reproduction/README.md).

## Matched-endpoint temporal ablation (third experiment)

- [Experiment README](../temporal_ablation/README.md), [fixed configuration](configs/temporal_ablation.yaml).
- Uses the same pinned Pollen data above, excluding the mixed-object `no_slip.csv` from object-disjoint evaluation.
- Uses the Pollen LSTM hidden width and two-linear-layer output head described above, with an analogous GRU variant. References are the same pinned `lstm.py` and `train.py`; no upstream pretrained weights are used.
- Changes the evaluation design: shared endpoints for 1/50-step inputs, train-only standardisation, fixed final epoch 200, separate calibration objects, held-out test objects. The test objects were evaluated in the first experiment, so this is an exploratory comparison, not a newly untouched blind test.
- Implements scalar temperature scaling following Guo et al. below; it does not implement DAC.
- Leakage guidance: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). Test observations must not fit a scaler or select a checkpoint when claiming an independent final test.

## Other open-source references (not vendored)

- Pollen Robotics **anyskin-slip-detection** (Apache-2.0): https://github.com/pollen-robotics/anyskin-slip-detection
- Pollen Robotics **tactile_gripper** (check its upstream license before copying): https://github.com/pollen-robotics/tactile_gripper
- AnySkin authors' sensor interface: https://github.com/raunaqbhirangi/anyskin

These repositories provide context and alternative reference implementations. Their source files and pretrained weights are not copied into this project. The **earlier** compact LSTM in `lstm_gru/src/models.py` is a separate matched baseline, not the Pollen architecture; the reproduction uses `pollen_reproduction/src/pollen_model.py` instead.

## Methods and papers

- Bhirangi et al., **AnySkin: Plug-and-play Skin Sensing for Robotic Touch** (2024): https://arxiv.org/abs/2409.08276 ; project: https://any-skin.github.io/
- Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017: https://proceedings.mlr.press/v70/guo17a.html . Positive scalar temperature scaling is implemented by minimising binary NLL on a separate calibration set.
- Hochreiter and Schmidhuber, **Long Short-Term Memory** (1997): https://doi.org/10.1162/neco.1997.9.8.1735
- Cho et al., **Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation** (2014): https://aclanthology.org/D14-1179/

## Hardware references for later work

- SO-100/SO-101 hardware: https://github.com/TheRobotStudio/SO-ARM100
- LeRobot: https://github.com/huggingface/lerobot

No robot-control code or physical-robot experiment is part of this repository. Density-Aware Calibration (DAC) is not implemented here. Both object-held-out experiments implement scalar temperature scaling; the Pollen reproduction does not calibrate its predictions.

## Python dependencies

PyTorch, NumPy, pandas, SciPy, scikit-learn, Matplotlib, PyYAML, Hugging Face datasets/Hub and pytest are installed as dependencies, not copied as source. Their upstream licenses apply to those packages. See [requirements.txt](environment/requirements.txt), [requirements-reproduction.txt](environment/requirements-reproduction.txt) and the tested local [requirements-lock.txt](environment/requirements-lock.txt).
