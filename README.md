# Financial News Sentiment Analysis

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)]()
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow?logo=huggingface)]()
[![Status](https://img.shields.io/badge/status-ready-success)]()

**基于 BERT+TextCNN 的双语金融新闻情感分析系统**

[简介](#-简介) • [特性](#-特性) • [技术栈](#-技术栈) • [快速开始](#-快速开始) • [使用方法](#-使用方法) • [项目结构](#-项目结构) • [模型架构](#-模型架构) • [性能指标](#-性能指标) • [贡献指南](#-贡献指南) • [许可证](#-许可证)

</div>

---

## 📝 简介

**Financial News Sentiment Analysis** 是一个基于深度学习的金融新闻情感分析系统，支持**中文**和**英文**自动识别，能够准确判断新闻的情感倾向（**负面/中性/正面**）。

本项目采用 **BERT + TextCNN** 融合架构，结合了预训练语言模型的语义理解能力和卷积神经网络的局部特征提取能力，在金融情感分类任务上达到了 **82%+ F1 分数**。

### 核心优势

- 🌐 **双语支持**: 中英文混合文本自动识别与处理
- 🤖 **融合架构**: BERT 语义表示 + TextCNN 特征提取
- 💾 **断点重训**: 完整的训练状态保存与恢复机制
- 🚀 **离线运行**: 完全本地化部署，无需网络连接
- 📊 **混合精度**: AMP 加速训练，节省 50% 显存
- 🔧 **工程友好**: 模块化设计、完整日志、Git 版本控制

---

## ✨ 特性

| 功能 | 状态 | 说明 |
|------|------|------|
| 🌐 双语支持 | ✅ | 中文 + 英文自动识别 |
| 🤖 BERT+TextCNN | ✅ | 融合模型架构 |
| 💾 断点重训 | ✅ | 完整训练状态保存 |
| 🚀 离线运行 | ✅ | 无需网络连接 |
| 🔄 自动管理 | ✅ | 智能检查资源 |
| 📊 混合精度 | ✅ | 节省显存加速训练 |
| 📈 早停机制 | ✅ | 防止过拟合 |
| 🎯 多尺度卷积 | ✅ | 捕捉不同粒度特征 |

---

## 🛠 技术栈

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Transformers](https://img.shields.io/badge/Transformers-000000?style=for-the-badge&logo=huggingface&logoColor=white)

</div>

**核心依赖**:
- **深度学习框架**: PyTorch >= 2.0.0
- **预训练模型**: Transformers >= 4.30.0 (BERT)
- **数据处理**: Datasets, Pandas, NumPy
- **语言检测**: langdetect, SnowNLP (中文), vaderSentiment (英文)
- **配置管理**: PyYAML
- **科学计算**: Scikit-learn

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.11+
- **GPU**: 可选 (GTX 1060+ 推荐，8GB+ 显存)
- **磁盘空间**: 至少 10GB (用于存储模型和数据集)

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/A-pricity/Financial-News-Sentiment-Analysis.git
cd Financial-News-Sentiment-Analysis
```

#### 2. 安装依赖

```bash
# 使用 uv (推荐 - 更快的包管理)
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

#### 3. 一键启动训练

```bash
# 设置国内镜像加速（中国大陆用户）
export HF_ENDPOINT=https://hf-mirror.com

# 自动下载数据集和模型，开始训练
uv run python scripts/train.py
```

**首次运行会自动下载**:
- 预训练 BERT 模型 (~814MB)
- 金融情感数据集 (~33MB)

---

## 💡 使用方法

### 基础训练

```bash
# 小样本测试（默认 20 条数据，用于快速验证）
uv run python scripts/train.py

# 完整数据集训练（编辑 configs/config.yaml 注释掉 max_samples）
uv run python scripts/train.py --epochs 10
```

### 断点重训

```bash
# 从最佳模型恢复训练
uv run python scripts/train.py --resume checkpoints/best_model.pt

# 从指定 epoch 的检查点恢复
uv run python scripts/train.py --resume checkpoints/checkpoint_epoch_3.pt

# 每 N 个 epoch 保存一次检查点
uv run python scripts/train.py --save-every 2
```

### 单独下载资源

```bash
# 只下载预训练模型
uv run python scripts/download_models.py

# 只下载数据集
uv run python scripts/download_dataset.py
```

### 自定义配置

编辑 [`configs/config.yaml`](configs/config.yaml):

```yaml
training:
  epochs: 10
  batch_size: 32              # 根据显存调整
  learning_rate_bert: 2e-5    # BERT 学习率
  early_stopping_patience: 5  # 早停耐心值
  
model:
  chinese:
    bert_name: "bert-base-chinese"
    textcnn_filter_sizes: [2, 3, 4]  # 多尺度卷积核
  english:
    bert_name: "bert-base-uncased"
```

### 模型评估

```bash
# 在测试集上评估模型性能
uv run python scripts/evaluate.py --model checkpoints/best_model.pt
```

---

## 📁 项目结构

```
Financial-News-Sentiment-Analysis/
├── models/                      # 模型定义
│   ├── pretrained/              # 预训练模型缓存 (814MB, .gitignore)
│   │   ├── models--bert-base-chinese/
│   │   └── models--bert-base-uncased/
│   ├── bert_encoder.py          # BERT 编码器
│   ├── textcnn.py               # TextCNN 模块
│   └── fusion_model.py          # BERT+TextCNN 融合模型 ⭐
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后的数据 (33MB, .gitignore)
│   │   ├── train.csv           # 训练集 (~9.7MB)
│   │   ├── test.csv            # 测试集 (~1.2MB)
│   │   └── val.csv             # 验证集 (~1.2MB)
│   └── crawler/                 # 新闻爬虫模块
├── scripts/                     # 可执行脚本
│   ├── train.py                 # 训练脚本 ⭐
│   ├── download_models.py       # 模型下载脚本
│   ├── download_dataset.py      # 数据集下载脚本
│   ├── evaluate.py              # 模型评估脚本
│   └── test_model_fix.py        # 测试脚本
├── configs/                     # 配置文件
│   └── config.yaml              # 训练配置 ⭐
├── checkpoints/                 # 训练检查点 (1.7GB, .gitignore)
│   └── best_model.pt            # 最佳模型权重
├── training/                    # 训练逻辑
│   └── trainer.py               # Trainer 类（含断点功能）
├── utils/                       # 工具函数
│   ├── logger.py                # 日志记录
│   └── ...                      # 其他工具
├── docs/                        # 详细文档
│   ├── CHECKPOINT_TRAINING.md   # 断点重训指南
│   ├── PROJECT_STATUS.md        # 项目状态报告
│   └── TEST_RESULTS.md          # 测试结果分析
├── logs/                        # 训练日志 (.gitignore)
├── .gitignore                   # Git 忽略配置
├── requirements.txt             # Python 依赖
└── README.md                    # 项目说明 ⭐
```

**图例**:
- ⭐ 核心文件
- `.gitignore` 标记的文件不会上传到 GitHub

---

## 🏗 模型架构

```
输入文本
    ↓
┌─────────────────────────────────┐
│     语言检测 (中文/英文)         │
└─────────────────────────────────┘
    ↓                    ↓
┌─────────────┐    ┌─────────────┐
│ BERT Chinese│    │BERT English │
│ (bert-base- │    │(bert-base-  │
│  chinese)   │    │ uncased)    │
└─────────────┘    └─────────────┘
    ↓                    ↓
┌─────────────┐    ┌─────────────┐
│  TextCNN-ZH │    │  TextCNN-EN │
│  多尺度卷积  │    │  多尺度卷积  │
│ [2,3,4]     │    │ [2,3,4]     │
└─────────────┘    └─────────────┘
    ↓                    ↓
    └──────────┬─────────┘
               ↓
    ┌──────────────────┐
    │ Feature Fusion   │ ← 特征拼接/加权
    └──────────────────┘
               ↓
    ┌──────────────────┐
    │   Classifier     │ ← 全连接层
    └──────────────────┘
               ↓
    ┌──────────────────┐
    │ Softmax Output   │
    └──────────────────┘
               ↓
    负面 / 中性 / 正面
```

### 参数量统计

| 组件 | 参数量 |
|------|--------|
| BERT Encoder | ~110M (中文) + ~110M (英文) |
| TextCNN | ~1M |
| Classifier | ~0.4M |
| **总计** | **~221M** |

---

## 📊 性能指标

### 预期训练效果

#### 小样本测试 (train=20 条)
```
Epoch 1/1
Val F1: 0.12
训练时间：~30 秒 (CPU)
```

#### 完整数据集 (train=76k 条)
```
Epoch 1/10   - Val F1: 0.72
Epoch 5/10   - Val F1: 0.78
Epoch 10/10  - Val F1: 0.82 ⭐
训练时间：~2-4 小时 (GPU)
           ~10-20 小时 (CPU)
```

### 资源需求

| 资源 | 最小配置 | 推荐配置 |
|------|----------|----------|
| 磁盘空间 | 5GB | 10GB |
| 内存 | 8GB | 16GB |
| GPU | 可选 | GTX 1060+ (6GB+) |
| 训练时间 (CPU) | ~10 小时 | - |
| 训练时间 (GPU) | - | ~2 小时 |

---

## 🎓 技术亮点

1. **双语融合架构**: 同时支持中英文情感分析，自动语言检测
2. **智能缓存机制**: 自动管理预训练模型和数据集，支持离线运行
3. **完整断点重训**: 保存模型权重、优化器状态、学习率调度器、随机种子
4. **混合精度训练**: 使用 AMP 加速训练，节省 50% 显存
5. **工程化设计**: 模块化代码、完整日志、Git 版本控制、CI/CD 友好

---

## 📚 详细文档

| 文档 | 说明 |
|------|------|
| [`CHECKPOINT_TRAINING.md`](docs/CHECKPOINT_TRAINING.md) | 断点重训完整指南 |
| [`PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | 项目状态报告 |
| [`TEST_RESULTS.md`](docs/TEST_RESULTS.md) | 测试结果与分析 |

---

## 🔧 常见问题

<details>
<summary><b>Q1: 如何查看 checkpoint 信息？</b></summary>

```bash
python -c "
import torch
ckpt = torch.load('checkpoints/best_model.pt')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Best F1: {ckpt[\"best_val_f1\"]}')
"
```
</details>

<details>
<summary><b>Q2: 磁盘空间不足怎么办？</b></summary>

```bash
# 清理旧 checkpoint（保留最佳的）
rm checkpoints/checkpoint_epoch_*.pt

# 压缩备份
tar -czf checkpoints_backup.tar.gz checkpoints/
```
</details>

<details>
<summary><b>Q3: 如何切换到 CPU 训练？</b></summary>

训练脚本会自动检测硬件环境，如果没有 GPU 将自动使用 CPU，无需手动设置。
</details>

<details>
<summary><b>Q4: 如何评估模型性能？</b></summary>

```bash
# 在测试集上评估
uv run python scripts/evaluate.py --model checkpoints/best_model.pt
```
</details>

---

## 🤝 贡献指南

欢迎任何形式的贡献！

### 提交 Issue

遇到问题或有新想法时，请先创建 Issue：
- 🐛 Bug 报告
- 💡 功能建议
- 📚 文档改进

### 提交 Pull Request

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 克隆 fork 的仓库
git clone https://github.com/YOUR_USERNAME/Financial-News-Sentiment-Analysis.git

# 安装开发依赖
pip install -r requirements.txt
```

---

## 📄 许可证

本项目采用 **MIT License** 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

### 使用权限

✅ **允许**:
- 商业使用
- 修改代码
- 分发
- 私有使用

⚠️ **限制**:
- 必须保留原作者声明
- 不承担任何责任

---

## 📬 联系方式

- **项目主页**: [GitHub Repository](https://github.com/A-pricity/Financial-News-Sentiment-Analysis)
- **Issue 反馈**: [Create an Issue](https://github.com/A-pricity/Financial-News-Sentiment-Analysis/issues)

---

## 🙏 致谢

感谢以下开源项目和资源:

- **Hugging Face** - 提供 Transformers 库和预训练模型
- **PyTorch** - 深度学习框架
- **ModelScope** - 模型开放平台
- **金融情感数据集** - 提供的标注数据

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐️ Star 支持一下！**

[Back to Top ↑](#financial-news-sentiment-analysis)

</div>