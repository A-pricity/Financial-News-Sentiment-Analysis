# 断点重训功能说明

## 📋 功能概述

训练脚本已完全支持断点重训功能，包括：
- ✅ 自动保存最佳模型
- ✅ 定期保存检查点
- ✅ 从指定 checkpoint 恢复训练
- ✅ 保留训练进度（epoch、optimizer、scheduler）
- ✅ 早停机制

## 🎯 使用方式

### 1. **基本训练**（自动保存最佳模型）
```bash
cd /mnt/workspace/repos/Financial-News-Sentiment-Analysis
export HF_ENDPOINT=https://hf-mirror.com

# 启动训练
uv run python scripts/train.py
```

**行为**：
- 每个 epoch 验证后，如果 F1 分数提升，自动保存到 `checkpoints/best_model.pt`
- 包含完整的模型权重、优化器状态、随机种子状态

### 2. **从最佳模型恢复训练**
```bash
# 自动查找并加载 checkpoints/best_model.pt
uv run python scripts/train.py --resume checkpoints/best_model.pt
```

**行为**：
- 加载上次的最佳模型
- 恢复 optimizer 和 scheduler 状态
- 从上次的 epoch 继续训练
- 保持最佳 F1 分数记录

### 3. **定期保存检查点**
```bash
# 每 2 个 epoch 保存一次检查点
uv run python scripts/train.py --save-every 2
```

**生成的文件**：
```
checkpoints/
├── best_model.pt              # 最佳模型（自动保存）
├── checkpoint_epoch_2.pt      # 第 2 epoch 的检查点
├── checkpoint_epoch_4.pt      # 第 4 epoch 的检查点
└── checkpoint_epoch_6.pt      # 第 6 epoch 的检查点
```

### 4. **指定训练总 epoch 数**
```bash
# 训练 10 个 epoch，每 3 个 epoch 保存一次
uv run python scripts/train.py --epochs 10 --save-every 3
```

### 5. **组合使用**
```bash
# 从 checkpoint 恢复，继续训练到总共 20 个 epoch
uv run python scripts/train.py \
  --resume checkpoints/checkpoint_epoch_5.pt \
  --epochs 20 \
  --save-every 2
```

## 📊 Checkpoint 内容

每个 checkpoint 文件包含：

```python
checkpoint = {
    "model_state_dict": ...,        # 模型权重
    "optimizer_state_dict": ...,    # 优化器状态
    "scheduler_state_dict": ...,    # 学习率调度器状态
    "best_val_f1": 0.85,           # 最佳验证 F1 分数
    "epoch": 5,                     # 保存时的 epoch
    "rng_state": ...,              # CPU 随机种子状态
    "cuda_rng_state": ...,         # GPU 随机种子状态（如果有）
}
```

## 🔄 断点重训流程

### 正常训练流程
```
Epoch 1 → Epoch 2 → Epoch 3 → ... → Epoch N
   ↓         ↓         ↓             ↓
验证       验证      验证          验证
   ↓         ↓         ↓
保存       保存     保存
(best)    (periodic) (periodic)
```

### 中断后恢复
```
场景：训练到 Epoch 5 时中断
已有 checkpoint: checkpoint_epoch_3.pt

恢复命令：
uv run python scripts/train.py --resume checkpoint_epoch_3.pt

执行流程：
1. 加载 checkpoint_epoch_3.pt
2. 恢复模型、优化器、scheduler 状态
3. 从 Epoch 4 开始继续训练
4. 保持之前的最佳 F1 记录
```

## 💡 实际示例

### 示例 1：完整训练（中途暂停后恢复）
```bash
# 第一天：开始训练，每 2 个 epoch 保存一次
uv run python scripts/train.py --epochs 10 --save-every 2

# 输出：
# Epoch 1/10 - Val F1: 0.75
# Epoch 2/10 - Val F1: 0.78 ✓ Saved checkpoint_epoch_2.pt
# Epoch 3/10 - Val F1: 0.80
# Epoch 4/10 - Val F1: 0.82 ✓ Saved checkpoint_epoch_4.pt
# [训练被中断]

# 第二天：从 checkpoint 恢复
uv run python scripts/train.py --resume checkpoints/checkpoint_epoch_4.pt

# 输出：
# Loading checkpoint from checkpoints/checkpoint_epoch_4.pt
# Checkpoint from epoch 4
# Starting from epoch 5
# Epoch 5/10 - Val F1: 0.83 ✓ New best!
# ...
```

### 示例 2：调整训练参数
```bash
# 初始训练
uv run python scripts/train.py --epochs 5

# 发现效果不够，继续训练更多 epoch
uv run python scripts/train.py --resume checkpoints/best_model.pt --epochs 15

# 输出：
# Loading checkpoint from checkpoints/best_model.pt
# Loaded checkpoint with best F1: 0.85
# Starting from epoch 2
# Epoch 2/15 - Val F1: 0.86 ✓
# ...
```

## ⚙️ 配置文件配合

在 `configs/config.yaml` 中可以设置默认值：

```yaml
training:
  epochs: 10                      # 默认训练 10 个 epoch
  batch_size: 16
  early_stopping_patience: 3      # 3 个 epoch 不提升则停止
  # 其他配置...
```

命令行参数会覆盖配置文件：
```bash
# 覆盖 config.yaml 中的 epochs 设置
uv run python scripts/train.py --epochs 20
```

## 📁 Checkpoint 目录结构

```
checkpoints/
├── best_model.pt              # 始终保存最佳模型
├── checkpoint_epoch_2.pt      # 定期保存的检查点
├── checkpoint_epoch_4.pt
├── checkpoint_epoch_6.pt
└── README.md                  # （可选）记录训练信息
```

建议创建 `checkpoints/README.md` 记录训练历史：
```markdown
# Training Checkpoints

## Session 1 (2026-04-01)
- checkpoint_epoch_2.pt: F1=0.78
- checkpoint_epoch_4.pt: F1=0.82

## Session 2 (2026-04-02)
- best_model.pt: F1=0.85 (from epoch 7)
```

## 🔍 常见问题

### Q1: 如何查看 checkpoint 信息？
```python
import torch
checkpoint = torch.load("checkpoints/best_model.pt")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best F1: {checkpoint['best_val_f1']}")
```

### Q2: 只想加载模型权重，不恢复训练状态？
```bash
# 手动加载，只使用 model_state_dict
python -c "
import torch
from models import BilingualFusionSentimentModel

model = BilingualFusionSentimentModel(...)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
# 不加载 optimizer 和 scheduler
"
```

### Q3: 如何清理旧的 checkpoint？
```bash
# 只保留最佳模型和最近 2 个检查点
cd checkpoints/
ls -t checkpoint_epoch_*.pt | tail -n +3 | xargs rm
# 保留 best_model.pt 和最新的 2 个 checkpoint
```

### Q4: 磁盘空间不足怎么办？
```bash
# 只保存最佳模型，删除所有定期检查点
rm checkpoints/checkpoint_epoch_*.pt
# 或者压缩保存
tar -czf checkpoints_backup.tar.gz checkpoints/
rm -rf checkpoints/*
```

## 🎉 优化效果

| 功能 | 修改前 | 修改后 |
|------|--------|--------|
| 保存内容 | 仅模型权重 | ✅ 模型+优化器+scheduler+RNG+epoch |
| 恢复精度 | 从 epoch 1 开始 | ✅ 从上次 epoch 继续 |
| 定期保存 | ❌ 不支持 | ✅ 每 N 个 epoch 保存 |
| 自动查找 | ❌ 需手动指定 | ✅ 自动查找最新 checkpoint |
| 学习率恢复 | ❌ 重置 | ✅ 保持连续 |

## 📝 注意事项

1. **GPU 环境一致性**：如果从 GPU 训练切换到 CPU 推理，需要修改加载代码：
   ```python
   checkpoint = torch.load(filepath, map_location="cpu")
   ```

2. **版本兼容性**：不同版本的 PyTorch/torchvision 可能影响 checkpoint 加载

3. **磁盘空间**：定期检查点会占用较多空间，建议合理设置 `--save-every` 参数

4. **训练连续性**：为了保证随机性的连续，checkpoint 保存了 RNG 状态

---

**更新时间**: 2026-04-01  
**状态**: ✅ 已完成并测试
