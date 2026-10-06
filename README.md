# tactile-slip-inference

项目按三类组织：**通用基础、第一次 LSTM/GRU 实验、Pollen 公开流程复现**。本次只是目录重组，没有重新训练模型或修改已有指标。

## 从哪里开始

| 目录 | 内容 | 入口 |
|---|---|---|
| `common/` | 共享原始数据、环境依赖、配置、许可证、来源和公共工具 | [通用基础说明](common/README.md) |
| `lstm_gru/` | 第一次 LSTM/GRU 训练、温度缩放、专用数据、全部模型和结果 | [第一次实验说明](lstm_gru/README.md) |
| `pollen_reproduction/` | Pollen 原流程复现的代码、测试、模型和全部结果 | [Pollen 复现说明](pollen_reproduction/README.md) |

```text
tactile-slip-inference/
├── common/
│   ├── data/raw/              共享的 17 个公开 CSV 和上游数据卡
│   ├── data/source_manifest.json
│   ├── environment/           依赖清单、已验证的版本锁定文件
│   ├── configs/lstm_gru/      LSTM/GRU 配置，按实验分类
│   ├── src/                  公共路径工具、数据下载器
│   ├── tests/                项目结构与路径兼容性测试
│   ├── LICENSE
│   └── THIRD_PARTY.md
├── lstm_gru/
│   ├── src/                  LSTM、GRU、训练、校准、评估和报告
│   ├── data/                 物体划分、处理后的窗口数据与 scaler
│   ├── runs/                 baseline_v1 和 final_v1 的原始模型与预测
│   ├── results/              原有指标、图表、报告
│   └── tests/
├── pollen_reproduction/
│   ├── src/                  Pollen 模型、数据处理、训练和报告
│   ├── runs/seed42/           原来的完整 200-epoch 训练产物
│   ├── results/              98.4346% 那次运行的指标、图表与报告
│   └── tests/
└── README.md
```

按你的确认，运行基础设施 `.venv/`、`.git/`、`.vscode/`、`.cache/` 和 `.pytest_cache/` 保留在根目录；`.gitignore`、`pytest.ini` 也保留在根目录供工具识别。VS Code 的 Python 解释器仍是 `.venv/Scripts/python.exe`，不需要重新选择或安装环境。

## 查看已经完成的结果

- [第一次 LSTM/GRU 实验报告](lstm_gru/results/REPORT.md)：物体隔离测试，包含温度缩放。
- [Pollen 复现报告](pollen_reproduction/results/REPORT.md)：完整 200 epochs，模型选择集 accuracy **98.4346%**、滑移 F1 **0.9436**。

两者评估协议不同，不能直接用 accuracy 比较优劣。Pollen 的 selection set 参与选模型，不是独立测试。两套实验都不包含 DAC、机器人控制或掉落率验证。

## 运行方式

所有命令都在**整个项目根目录**运行，不要先进入某个实验子目录。

```powershell
# 检查整个项目，不重新训练
.\.venv\Scripts\python.exe -X utf8 -m pytest -q --basetemp .cache/pytest

# 查看两个实验的参数
.\.venv\Scripts\python.exe -m lstm_gru --help
.\.venv\Scripts\python.exe -m pollen_reproduction --help
```

具体数据下载、训练和新建重复实验的方法分别见三个目录的 README。已经完成的模型默认不会被静默覆盖；本次保留了旧权重内的历史配置，并兼容其中的旧数据路径。

原始数据、环境、缓存、处理后的数组和训练权重继续被 Git 忽略。代码、配置、小型结果和报告可以提交；**这次没有提交或上传 GitHub**。

## 来源与许可

[数据与方法来源](common/THIRD_PARTY.md) · [项目许可证](common/LICENSE)
