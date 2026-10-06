# 共享原始 AnySkin 数据

来源：[Pollen Robotics AnySkin slip dataset](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection)。
固定 revision：`d9f5e315006482635a2effa34488715871d55c13`。数据卡声明 Apache-2.0。

- `raw/`：17 个原始 CSV，以及原样保存的 `UPSTREAM_README.md`。
- [source_manifest.json](source_manifest.json)：每个文件的来源 URL、大小和 SHA-256。
- 数据由 Reachy 2 夹爪上的 AnySkin 采集，不是本项目在 SO-100/SO-101 上采集的数据。
- 每行包含时间、15 个磁场通道和人工标注的当前滑移标签；不是力的真值或未来滑移标签。

两套实验使用同一份原始数据，处理方式不同：

- [LSTM/GRU](../../lstm_gru/README.md)：按物体隔离，排除混合物体的 `no_slip.csv`；窗口数组和标准化参数在 `lstm_gru/data/processed/`。
- [Pollen 复现](../../pollen_reproduction/README.md)：使用全部 17 个 CSV；按原流程随机划分数据行，不使用多时间步窗口。

从项目根目录下载：`python -m common.src.download_data`。已下载的数据无需因目录调整重新下载。
