# 基于 CNN–LSTM 的台风路径预测

[English](README.md) · **简体中文** · [运行与复现说明](docs/RUN_GUIDE.md)

结合台风历史轨迹和 ERA5 环境场，使用 PyTorch 实现 CNN–LSTM 路径预测。本仓库包含数据预处理、模型训练、已保存模型以及推理可视化代码，展示从环境特征构建到预测轨迹绘制的完整流程。

完整项目统一维护在 **`main` 分支**，原有实现与研究文件均予以保留。

## 示例结果

下图为仓库已有的历史输出，用于展示轨迹可视化形式。本次文档整理未重新训练模型或复现这些结果，也未据此报告预测精度。

| Carmen | Kujira |
| :---: | :---: |
| ![Carmen 台风历史轨迹可视化](inference/Carmen.png) | ![Kujira 台风历史轨迹可视化](inference/Kujira.png) |

红色点表示原始轨迹，黄色点表示预测轨迹。

## 方法概览

模型读取连续 **8 个时间步**的经纬度增量、地面气压（SP）和海表温度（SST），预测后续 **2 个时间步**的经纬度增量，再累加为位置。经度与纬度分别使用独立模型。具体预测时长取决于输入数据的时间间隔。

```text
8 个时间步的经纬度增量 + 展平后的 SP / SST 网格
                         ↓
     三组 Conv1d → BatchNorm → LeakyReLU → MaxPool
                         ↓
                    三层 LSTM
                         ↓
                 后续 2 步的坐标增量
                         ↓
                  累加得到预测位置
```

环境网格先展平并与经纬度增量拼接；代码中的 `Conv1d` 沿时间维度处理序列。模型定义可见 [训练脚本](train/train.py) 和 [推理脚本](inference/inference.py)。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`CMABSTdata/`](CMABSTdata/) | CMA 台风最佳路径历史数据 |
| [`EAR5/`](EAR5/) | ERA5 请求记录及示例 GRIB 文件，沿用原目录名称 |
| [`data_preprocess/`](data_preprocess/) | 数据处理脚本、Notebook、轨迹 CSV 和 SP / SST 网格 JSON |
| [`train/`](train/) | 训练脚本与推理示例所用轨迹数据 |
| [`train/model/`](train/model/) | 已保存的纬度与经度模型文件 |
| [`inference/`](inference/) | 推理、绘图脚本、Notebook 和历史输出 |
| [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md) | 环境准备、运行目录及已知复现问题 |
| [`requirements.txt`](requirements.txt) | 历史 Linux 环境包清单 |

## 获取与运行

从 `main` 分支获取项目：

```bash
git clone --branch main https://github.com/Will1202/Typhoon-Trajectory-Prediction-Based-On-CNN-LSTM-model-main.git
cd Typhoon-Trajectory-Prediction-Based-On-CNN-LSTM-model-main
```

历史环境使用 **Python 3.10** 和 **PyTorch 2.1**。`requirements.txt` 使用 Conda 导出格式，包含 Linux 构建信息及 pip 来源包条目，不能直接用于 `pip install -r requirements.txt`。环境准备与依赖说明见[运行指南](docs/RUN_GUIDE.md)。

准备好环境和数据后，在 `inference` 目录运行已有示例：

```bash
cd inference
python inference.py
```

脚本默认使用 `Carmen` 及相应轨迹、环境网格和已保存模型。现有代码通过相对路径读取文件，运行目录需要与上述命令一致。执行时会覆盖 `inference/` 下的 `predicted_typhoon_path.csv`、`original_typhoon_path.csv` 和 `Carmen.png`；需要保留历史输出时，请先备份。

训练入口为 [`train/train.py`](train/train.py)，数据处理入口为 [`data_preprocess/data_preprocess.ipynb`](data_preprocess/data_preprocess.ipynb)。运行前请查阅指南中的路径、数据匹配、归一化及模型兼容性说明。历史脚本保留了原有实现与复现限制，当前文档不代表端到端运行已验证通过。

## 数据来源

- **台风轨迹**：[CMA Tropical Cyclone Best Track Dataset](https://tcdata.typhoon.org.cn/en/zjljsjj.html)。
- **环境场**：[ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)，通过 Copernicus Climate Data Store 获取。

数据处理代码将轨迹记录与 SP / SST 网格按台风名称和时间对应。自行获取 ERA5 数据时，需要配置个人 CDS API 凭据，并遵循数据提供方的访问、使用与引用要求。

## 许可证

代码遵循 [MIT License](LICENSE)，保留原版权声明：**Copyright (c) 2024 Yiwei Liang**。第三方数据的使用与引用遵循各数据提供方的条款。
