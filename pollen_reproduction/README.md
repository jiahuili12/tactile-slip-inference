# Pollen 公开训练流程复现说明

## 目的与来源

目的：从随机初始化重新训练，检查是否能够接近 Pollen 模型卡的 **98.26% accuracy**，不是调用预训练权重，也不是先假定 GRU 比 LSTM 好。

- [训练代码，固定 commit](https://github.com/pollen-robotics/anyskin-slip-detection/blob/24a3728c35985def720ff292a8a45ef7e8662a64/src/train.py)
- [模型定义，同一 commit](https://github.com/pollen-robotics/anyskin-slip-detection/blob/24a3728c35985def720ff292a8a45ef7e8662a64/src/lstm.py)
- [发布者模型卡](https://huggingface.co/pollen-robotics/anyskin-slip-detection/blob/main/README.md)
- [数据固定版本](https://huggingface.co/datasets/pollen-robotics/anyskin_slip_detection/tree/d9f5e315006482635a2effa34488715871d55c13)

模型卡没有给出产生 98.26% 的全部运行配置。因此本项目是**公开训练协议复现**，不宣称逐位复现发布者的某次运行。

## 对照表

| 项目 | Pollen 公开脚本 | 本次实现 |
|---|---|---|
| 数据 | Hub 的 `data/` 全部 CSV | 固定 revision，17 个 CSV |
| 加载 | Hugging Face `load_dataset` | 同一加载入口，核验本地 CSV 数值和顺序 |
| 特征 | 5 个磁传感器 × 3 轴 | 相同的 15 个通道及顺序 |
| 标准化 | 划分前拟合全体数据 | 忠实保留，并标注泄漏 |
| 划分 | 随机按行，80/20，random_state=42 | 相同，不做物体隔离或分层 |
| 输入 | `[batch, 1, 15]` | 相同，只有一个时间步 |
| 网络 | LSTM 128 + 两个线性层 | 相同尺寸、顺序和零初态 |
| 训练 | batch 32、Adam 0.001、BCE | 相同，无类别权重或梯度裁剪 |
| epochs | 连续两个 100-epoch 循环 | 200 epochs，保留轮间最佳分数清零 |
| 最终模型 | 第二轮中准确率最高的权重 | `phase2_best.pt`，额外保留第一轮权重作审计 |
| 判断滑移 | sigmoid 输出严格大于 0.5 | 相同 |
| 设备 | 硬编码 CUDA | 本机 CPU；运行环境写入结果 |
| 随机性 | 只固定数据划分 seed | 额外固定 PyTorch/NumPy seed=42 |
| 增强 | 算出增强数组，但没有用于训练 | 省略这个无效计算，不使用增强 |
| 输出 | 最后一批 loss 和准确率 | 加上平均 loss、完整历史、指标与来源 |

CPU/GPU、PyTorch 版本和初始化差异可能改变结果；不为接近目标分数反复挑选随机种子。默认只运行预先指定的 seed 42。

## 为什么保留原流程中的问题

这次先回答“公开流程实际可以得到什么分数”。若同时改为训练集标准化、物体隔离和独立验证，就不再是同一个流程。保留这些做法只是为了审计复现，不代表推荐沿用。

1. **按行随机划分**：同一物体、同一段记录的相邻时刻可能同时出现在训练和评估中，不能推论新物体泛化。
2. **全数据标准化**：模型选择集的特征分布参与了预处理参数估计。
3. **反复选模型**：原脚本每个 epoch 都看所谓 test accuracy 并据此保存权重；本文明确命名为 selection accuracy，不称独立 test accuracy。
4. **只有一个时间步**：这个配置没有用连续历史去学习时间依赖，不能据此声称已经实现时序滑移预警。
5. **场景有限**：数据是 Reachy 2 上人工拉动物体并人工标注的记录；不是 SO-101 掉落率、控制安全性或未来滑移标签。

## 怎样读结果

- [REPORT.md](../results/pollen_reproduction/REPORT.md)：模型卡分数、本次分数、差多少**个百分点**、precision/recall/F1 和混淆矩阵。
- `history.csv`：完整 200 个 epoch；第二阶段继续训练，不是独立种子实验。
- `per_recording.csv`：源文件在 selection set 中的表现，不能误称 unseen-object performance。
- `data_audit.json`：数据版本、文件 SHA-256、实际加载顺序、划分索引哈希。
- `run_config.json`：包版本、设备、随机种子和超参数。
- `training_summary.json`：两个阶段各自选中的 epoch 和耗时。
- `runs/.../selection_predictions.npz`：行索引、真实标签和 logits，可重新核算指标。

旧的 [物体隔离实验](OBJECT_HOLDOUT_BASELINE.md) 仍保留，二者不能只拿一个 accuracy 数字比较算法优劣。下一阶段才是在统一、严格协议下比较模型与校准方法；本次不实现 DAC。

## 可选：与原代码逐权重核验

`tests/test_pollen_reference.py` 在合成数据上以相同初始化、批次和两个训练阶段运行双方代码，检查最终选中的每个权重是否完全相等。这是代码一致性检查，不是额外实验成绩。

默认测试不联网。原文件只保存在 Git 忽略的缓存中，测试执行前核验 SHA-256；没有缓存时此项会 skip。需要重新建立缓存时，在项目根目录运行：

```powershell
New-Item -ItemType Directory -Force .cache/pollen_reference
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/pollen-robotics/anyskin-slip-detection/24a3728c35985def720ff292a8a45ef7e8662a64/src/train.py" -OutFile .cache/pollen_reference/train.py
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/pollen-robotics/anyskin-slip-detection/24a3728c35985def720ff292a8a45ef7e8662a64/src/lstm.py" -OutFile .cache/pollen_reference/lstm.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_pollen_reference.py --basetemp .cache/pytest_reference
```

下载的原文件保留 Pollen 的 Apache-2.0 许可，不纳入本项目源码发布；本项目自己的复现实现有明确的来源说明。
