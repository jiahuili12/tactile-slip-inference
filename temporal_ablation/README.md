# 第三次实验：历史长度 × 循环网络结构

**研究什么？** 使用同一批 AnySkin 公开数据、同一物体划分和相同预测时刻，比较 **LSTM/GRU × 1/50 时间步**。只改变循环网络类型及可见历史，观察历史信息对**未见物体的当前滑移识别**是否有帮助，再单独检查温度缩放的概率校准效果。

这里不是继续复刻 Pollen 的随机行测试，而是一次有固定协议的新对照。**不含 DAC、不控制真实机器人。** 代码和小型结果可上传 GitHub，原始数据、处理后数组及训练权重保留在本地。

## 结果

**已完成四组各 200 轮训练，并从最终权重复核全部预测。** 测试是 3 个未参与训练/校准的物体，共 1,813 个配对窗口。

| 模型 | 时间步 | Accuracy | 滑移 F1 | AP |
|---|---:|---:|---:|---:|
| LSTM | 1 | 77.94% | 0.0909 | 0.1756 |
| LSTM | 50 | 74.52% | 0.1283 | 0.1923 |
| GRU | 1 | 75.95% | 0.1711 | 0.2191 |
| GRU | 50 | 74.24% | 0.1463 | 0.3165 |

**简单结论：**

- **历史更长不保证分类更好。** GRU 的 AP（反映滑移评分的排序质量）从 0.2191 提高到 0.3165，但固定 0.5 阈值下的 F1 反而下降。LSTM 的 F1/AP 都略有改善。不能概括为“50 步全面优于 1 步”。
- **校准不等于识别能力提升。** 50 步 GRU 的测试 ECE 从 0.2311 降至 0.0655，NLL 从 1.9800 降至 0.5253；但 accuracy/F1 不变。两个单步模型的测试 ECE 反而上升，尽管 NLL 下降。
- **当前跨物体泛化仍然较弱。** 始终预测不滑移也能得到 78.99% accuracy；四模型的滑移召回率只有 5.25%–11.81%。甚至仅输出训练集滑移比例的常数概率基线，测试 NLL 为 0.5169、ECE 为 0.0291。这说明低 ECE 本身不代表模型能有效检测滑移，更不能证明可安全控制机器人。
- **这是探索性结果，不是架构定论。** 只有单种子、单划分，测试物体在第一轮实验中已被评估；不能据此普遍断言 GRU 或 LSTM 更好。我们没有根据这些成绩重新挑轮次、阈值或重训。

完整指标、逐物体结果、图表和限制见 [results/REPORT.md](results/REPORT.md)。不能把这里与 Pollen 准确率的差解释为其结果的“虚高幅度”。

## 实验设计

| 项目 | 四组共用的设置 |
|---|---|
| 数据 | Pollen Robotics 公开 AnySkin 磁信号，15 个通道 |
| 训练物体 | apple, ball, book, bottle, cable, carambole, cork, cream, cup, elastic |
| 校准物体 | foam, foam_filter, gluestick |
| 测试物体 | grappe, socks, tissues |
| 输入 | 100 Hz 因果重采样；50 步约 0.5 秒；步幅 5 |
| 对齐 | 先构造 50 步窗口；单步只取窗口最后一行，四组预测相同端点和标签 |
| 标准化 | 仅训练物体的独立重采样行拟合 mean/std |
| 网络 | 单层 LSTM 或 GRU，hidden size 128，Linear(128,128) → Linear(128,1)，无中间激活 |
| 训练 | seed 42，Adam，学习率 0.001，batch 32，BCE，固定 200 轮 |
| 权重选择 | 只用第 200 轮；不早停、不在训练中看测试/校准成绩 |
| 校准 | 独立校准集最小化 NLL，拟合正标量温度，范围 0.05–20 |
| 决策 | 概率严格大于 0.5 判滑移；不根据测试集调阈值 |

原实验的 2 个验证物体并入训练，成为 10/3/3 个训练/校准/测试物体。这里**不需要验证集**，是因为训练时长、模型和阈值事先固定，不做选择；这不代表所有研究都可以省略验证集。校准集仍然必要，因为拟合温度也会使用真实标签，不能拿最终测试标签拟合。

预处理会检查并保存三部分数据；训练函数只加载训练窗口。四组训练完成后才让模型预测校准集并拟合各自的温度，最后预测测试集、计算指标。保存源数据、处理后数据和权重的 SHA-256 供复核。

## 目录

```text
temporal_ablation/
├── src/                  数据、LSTM/GRU、固定轮数训练、校准、评估和报告
├── tests/                合成数据上的协议/完整流程测试
├── data/fixed_v1/        共享端点窗口、训练集 scaler（本地，Git 忽略）
├── runs/fixed200_seed42/ 四个最终权重、训练日志、逐窗口预测（本地，Git 忽略）
├── results/              可提交的小型指标、审计记录、图表、报告
└── README.md
```

通用配置放在 [common/configs/temporal_ablation.yaml](../common/configs/temporal_ablation.yaml)；原始数据共享 [common/data/](../common/data/README.md)，不会复制或改写前两次实验。

## 运行

在**整个项目根目录**的 PowerShell 终端运行。使用现有 `.venv`；不需要新增依赖。

```powershell
# 先验证实现；这里只使用小型合成数据
.\.venv\Scripts\python.exe -X utf8 -m pytest temporal_ablation/tests -q --basetemp .cache/pytest-temporal

# 首次正式训练，两个 CPU 工作进程，不访问网络或真实机器人
.\.venv\Scripts\python.exe -X utf8 -u -m temporal_ablation

# 如已有默认结果，使用新目录，避免覆盖现有结果
.\.venv\Scripts\python.exe -X utf8 -u -m temporal_ablation --run-dir temporal_ablation/runs/repeat_seed42 --results-dir temporal_ablation/results_repeat_seed42

# 正式实验完成后，只复核数据/权重/预测/指标，不重新训练
.\.venv\Scripts\python.exe -X utf8 -m temporal_ablation.src.verify
```

脚本拒绝覆盖已存在的 run/results 目录。正式报告保存全部 200 轮训练日志而非“最好的一轮”。测试指标由本地 `runs/*/test_predictions.csv` 可复算。

## 和原来有什么不同？

- **第一次 LSTM/GRU：**已有 50 步历史和物体隔离，但用验证集早停、hidden size 32，没有同端点单步对照。
- **Pollen 复刻：**输入只有一个时间步，随机划分数据行，全数据拟合 scaler，按名为 test 的集合选权重。我们的复刻 98.4346% 是模型选择集成绩。
- **本次：**四组配对端点、训练集标准化、未见物体测试、固定最终轮数。保留 Pollen 的 hidden size、输出头、优化器、batch size 和 200 总轮数，但没有保留它的模型选择流程。

Pollen 的约 98% **存在乐观的模型选择偏差风险，不能作为独立最终测试成绩**；不是说它的计算有误，也无法从现有结果判断具体高估多少。

## 限制

- 本次模型没见过测试物体，但这些物体的成绩此前已在项目中查看。因此是**探索性、预先固定的对照**，不是新的盲测。
- 只有单种子、单划分、3 个测试物体；窗口重叠，不应把所有窗口当作独立实验次数。
- 同 hidden size 不等于同参数量；LSTM 90,881 参数，GRU 72,321 参数。
- 温度缩放只改变概率；在固定 0.5 阈值下不会改变 accuracy/F1，也不会改变 AP/ROC 排序。跨物体分布变化下，校准改善不保证迁移到测试集。
- 数据来自 Reachy 2/AnySkin，不能据此声称验证了 SO-101 实物闭环、提前预警、掉落率或实时控制效果。

## 来源

- **数据：**[Pollen Robotics AnySkin Slip Detection Dataset](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection)，固定 revision `d9f5e315006482635a2effa34488715871d55c13`，上游标注 Apache-2.0；下载与哈希见 [source_manifest.json](../common/data/source_manifest.json)。混合物体 `no_slip.csv` 不纳入物体隔离实验。
- **架构与训练设置参考：**[Pollen 模型](https://github.com/pollen-robotics/anyskin-slip-detection/blob/24a3728c35985def720ff292a8a45ef7e8662a64/src/lstm.py) 与 [训练脚本](https://github.com/pollen-robotics/anyskin-slip-detection/blob/24a3728c35985def720ff292a8a45ef7e8662a64/src/train.py)，Apache-2.0。这里重新实现相同 LSTM 宽度和线性头，并做对应 GRU 对照；不使用上游预训练权重。
- **温度缩放：**[Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)。这不是 Density-Aware Calibration。
- 完整方法与依赖来源见 [THIRD_PARTY.md](../common/THIRD_PARTY.md)，项目许可见 [LICENSE](../common/LICENSE)。
