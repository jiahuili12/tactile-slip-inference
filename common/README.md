# 1. 通用基础

这里放三次实验共用的基础内容，不放某个实验的模型或性能结果。

| 位置 | 用途 |
|---|---|
| [data/README.md](data/README.md) | 共享原始数据及其来源 |
| `data/raw/` | Pollen 原始 CSV 和原样保存的上游数据卡，Git 忽略 |
| [data/source_manifest.json](data/source_manifest.json) | 固定数据版本、下载地址和 SHA-256 |
| `environment/` | 基础依赖、复现依赖、精确版本清单 |
| `configs/lstm_gru/` | 第一次实验的配置；配置集中存放，但按实验分目录 |
| `configs/temporal_ablation.yaml` | 第三次实验的固定协议：四组对照、物体划分和温度缩放 |
| `src/utils.py` | 项目根目录定位、配置读取和公共工具 |
| `src/download_data.py` | 下载公共数据，供三次实验使用 |
| `tests/`、`testing.py` | 共享路径测试和 pytest 辅助设置 |
| [LICENSE](LICENSE) | 项目许可证 |
| [THIRD_PARTY.md](THIRD_PARTY.md) | 第三方数据、代码与论文来源 |

**已安装的实际环境仍在项目根目录的 `.venv/`，这里保存的是可重建环境的依赖说明。**

从整个项目根目录运行：

```powershell
# 已有环境不需要重装；新环境可先执行 py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r common/environment/requirements-reproduction.txt

# 或按本地已验证版本安装
.\.venv\Scripts\python.exe -m pip install -r common/environment/requirements-lock.txt

# 只有原始数据缺失时才需要下载
.\.venv\Scripts\python.exe -X utf8 -m common.src.download_data
```

只做 LSTM/GRU 基线时也可使用 `common/environment/requirements.txt`；复现依赖文件在此基础上增加 Hugging Face datasets。

各次训练实际使用的配置快照属于实验记录，分别保存在实验的 `runs/` 和 `results/` 中，不混入这里的可编辑配置。旧模型中保存的 `data/processed` 路径由公共工具兼容到 `lstm_gru/data/processed`，未重写旧权重。

[返回项目首页](../README.md)
