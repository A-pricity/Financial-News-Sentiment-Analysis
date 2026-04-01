#!/usr/bin/env python3
"""
模型评估脚本 - 在测试集上评估模型性能并生成详细报告
"""

import os
import sys
import logging
import yaml
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.fusion_model import BilingualFusionSentimentModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """简单的数据集类用于评估"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 检测语言
        language = "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label,
            "language": language,
        }


def load_model(checkpoint_path: str, config: dict, device: str = "cuda"):
    """加载训练好的模型"""
    
    logger.info(f"Loading model from {checkpoint_path}...")
    
    # 确定缓存目录
    model_cache_dir = os.path.join(PROJECT_ROOT, "models/pretrained")
    
    model = BilingualFusionSentimentModel(
        zh_bert_name=config["model"]["chinese"]["bert_name"],
        en_bert_name=config["model"]["english"]["bert_name"],
        zh_textcnn_filter_sizes=config["model"]["chinese"]["textcnn_filter_sizes"],
        en_textcnn_filter_sizes=config["model"]["english"]["textcnn_filter_sizes"],
        textcnn_num_filters=config["model"]["chinese"]["textcnn_num_filters"],
        fusion_hidden_dim=config["model"]["fusion"]["hidden_dim"],
        dropout=config["model"]["fusion"]["dropout"],
        cache_dir=model_cache_dir,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 加载最佳 F1 分数（如果有）
    best_f1 = checkpoint.get("best_val_f1", 0.0)
    epoch = checkpoint.get("epoch", 0)
    
    logger.info(f"✓ Model loaded successfully!")
    logger.info(f"  - Checkpoint epoch: {epoch}")
    logger.info(f"  - Best validation F1: {best_f1:.4f}")

    return model, best_f1, epoch


def evaluate_model(model, data_loader, device, label_names=["negative", "neutral", "positive"]):
    """在数据加载器上评估模型"""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info("Evaluating on test set...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            language = batch.get("language", ["en"] * len(input_ids))[0]

            logits, _, _ = model(input_ids, attention_mask, language=language)
            probs = torch.softmax(logits, dim=1)
            
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return all_preds, all_labels, all_probs


def generate_report(y_true, y_pred, label_names=["negative", "neutral", "positive"]):
    """生成详细的评估报告"""
    
    print("\n" + "="*70)
    print(" " * 20 + "MODEL EVALUATION REPORT")
    print("="*70)
    
    # 总体准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 详细分类报告
    print("\n📋 Classification Report:")
    print("-" * 70)
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=label_names,
        digits=4,
        output_dict=False
    )
    print(report)
    
    # 混淆矩阵
    print("\n📋 Confusion Matrix:")
    print("-" * 70)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Predicted →")
    print(f"True ↓\t\t{label_names[0]}\t{label_names[1]}\t{label_names[2]}")
    for i, name in enumerate(label_names):
        print(f"{name}\t\t{cm[i][0]}\t{cm[i][1]}\t{cm[i][2]}")
    
    # 按类别的详细指标
    print("\n📋 Per-class Metrics:")
    print("-" * 70)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None
    )
    
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for i, name in enumerate(label_names):
        print(f"{name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # 宏平均和加权平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print("\n" + "-" * 70)
    print(f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
    print(f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
    print("="*70)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'report_dict': classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    }


def plot_results(y_true, y_pred, y_probs, label_names, save_path=None):
    """绘制评估结果图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Evaluation Results', fontsize=16)
    
    # 1. 混淆矩阵热力图
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names, ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. 各类别 F1 分数柱状图
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None
    )
    x = range(len(label_names))
    width = 0.25
    
    axes[0, 1].bar([i - width for i in x], precision, width, label='Precision')
    axes[0, 1].bar(x, recall, width, label='Recall')
    axes[0, 1].bar([i + width for i in x], f1, width, label='F1-Score')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Per-class Performance')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(label_names)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. ROC 曲线（One-vs-Rest）
    y_true_bin = pd.get_dummies(y_true).values
    y_prob_arr = np.array(y_probs)
    
    for i, name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_arr[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve (One-vs-Rest)')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 预测分布
    unique, counts = np.unique(y_pred, return_counts=True)
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    axes[1, 1].bar([label_names[i] for i in unique], counts, color=colors)
    axes[1, 1].set_xlabel('Predicted Class')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved evaluation plots to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Financial Sentiment Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/processed/test.csv",
        help="Path to test data CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate evaluation plots",
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, train_best_f1, train_epoch = load_model(args.checkpoint, config, device)
    model.to(device)
    
    # 加载测试数据
    logger.info(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # 设置模型缓存目录
    model_cache_dir = os.path.join(PROJECT_ROOT, "models/pretrained")
    
    # 加载 tokenizer（使用与训练时相同的）
    logger.info("Loading tokenizer...")
    # 根据测试数据的语言选择 tokenizer
    if test_df['language'].iloc[0] == 'zh':
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["chinese"]["bert_name"],
            cache_dir=model_cache_dir,
            local_files_only=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["english"]["bert_name"],
            cache_dir=model_cache_dir,
            local_files_only=True
        )
    
    # 创建数据集和数据加载器
    test_dataset = SentimentDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # 评估模型
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device)
    
    # 生成报告
    label_names = ["negative", "neutral", "positive"]
    metrics = generate_report(y_true, y_pred, label_names)
    
    # 保存结果到文件
    results_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Training Epoch: {train_epoch}\n")
        f.write(f"Training Best Val F1: {train_best_f1:.4f}\n")
        f.write(f"Test Set Size: {len(test_df)}\n")
        f.write(f"Device: {device}\n\n")
        
        # 写入分类报告
        report_str = classification_report(
            y_true, y_pred, 
            target_names=label_names,
            digits=4
        )
        f.write(report_str)
    
    logger.info(f"Saved evaluation results to {results_file}")
    
    # 绘制图表
    if args.plot:
        try:
            import numpy as np
            plot_file = os.path.join(args.output_dir, "evaluation_plots.png")
            plot_results(y_true, y_pred, y_probs, label_names, save_path=plot_file)
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
    
    # 打印最终总结
    print("\n" + "="*70)
    print(" " * 25 + "SUMMARY")
    print("="*70)
    print(f"✅ Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"✅ Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"✅ Weighted F1:    {metrics['weighted_f1']:.4f}")
    print(f"✅ Training F1:    {train_best_f1:.4f} (from epoch {train_epoch})")
    print("="*70)
    
    # 检查性能下降
    if train_best_f1 > 0 and metrics['macro_f1'] < train_best_f1 * 0.9:
        print("\n⚠️  WARNING: Test F1 is significantly lower than training F1!")
        print(f"   This might indicate overfitting.")
    
    print("\n💡 Detailed results saved to:")
    print(f"   - {results_file}")
    if args.plot:
        print(f"   - {os.path.join(args.output_dir, 'evaluation_plots.png')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
