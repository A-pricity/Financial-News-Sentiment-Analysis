#!/usr/bin/env python3
"""
模型评估脚本 - 在测试集和验证集上评估模型性能并生成详细报告

结果保存位置：
- 测试集结果: results/evaluation/<timestamp>/
- 验证集结果: results/val/<timestamp>/
- 训练历史: results/training/<timestamp>/

每个结果目录包含：
- metrics.csv: 量化指标
- report.md: Markdown格式报告
- evaluation_results.md: 详细文本报告
- confusion_matrix.png: 混淆矩阵可视化
- per_class_metrics.png: 各类别指标柱状图
"""

import os
import sys
import logging
import yaml
import torch
import pandas as pd
import argparse
from datetime import datetime
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
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from models.fusion_model import BilingualFusionSentimentModel
from utils.dataset_loader import load_dataset, load_kfold_dataset

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


def evaluate_model(
    model, data_loader, device, label_names=["negative", "neutral", "positive"]
):
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

    print("\n" + "=" * 70)
    print(" " * 20 + "MODEL EVALUATION REPORT")
    print("=" * 70)

    # 总体准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # 详细分类报告
    print("\n📋 Classification Report:")
    print("-" * 70)
    report = classification_report(
        y_true, y_pred, target_names=label_names, digits=4, output_dict=False
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
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )

    print(
        f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}"
    )
    print("-" * 70)
    for i, name in enumerate(label_names):
        print(
            f"{name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}"
        )

    # 宏平均和加权平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    print("\n" + "-" * 70)
    print(
        f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}"
    )
    print(
        f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}"
    )
    print("=" * 70)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "report_dict": classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True
        ),
    }


def plot_results(y_true, y_pred, y_probs, label_names, save_path=None):
    """绘制评估结果图表"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Evaluation Results", fontsize=16)

    # 1. 混淆矩阵热力图
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_ylabel("True Label")
    axes[0, 0].set_xlabel("Predicted Label")

    # 2. 各类别 F1 分数柱状图
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    x = range(len(label_names))
    width = 0.25

    axes[0, 1].bar([i - width for i in x], precision, width, label="Precision")
    axes[0, 1].bar(x, recall, width, label="Recall")
    axes[0, 1].bar([i + width for i in x], f1, width, label="F1-Score")
    axes[0, 1].set_xlabel("Class")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Per-class Performance")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(label_names)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3. ROC 曲线（One-vs-Rest）
    y_true_bin = pd.get_dummies(y_true).values
    y_prob_arr = np.array(y_probs)

    for i, name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_arr[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    axes[1, 0].plot([0, 1], [0, 1], "k--")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].set_title("ROC Curve (One-vs-Rest)")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(alpha=0.3)

    # 4. 预测分布
    unique, counts = np.unique(y_pred, return_counts=True)
    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]
    axes[1, 1].bar([label_names[i] for i in unique], counts, color=colors)
    axes[1, 1].set_xlabel("Predicted Class")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Prediction Distribution")
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
        default=None,
        help="Path to test data CSV (优先于 --dataset)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称 (datasets/ 下的子目录名)",
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
        default="results/evaluation",
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

    # 创建带时间戳的结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_eval_dir = os.path.join("results", "evaluation", timestamp)
    os.makedirs(results_eval_dir, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {results_eval_dir}")

    # 加载模型
    model, train_best_f1, train_epoch = load_model(args.checkpoint, config, device)
    model.to(device)

    # 加载测试数据
    if args.test_data:
        # 使用指定的测试数据文件
        logger.info(f"Loading test data from {args.test_data}...")
        test_df = pd.read_csv(args.test_data)
        val_df = None  # 没有验证集
    elif args.dataset:
        dataset_name = args.dataset
        dataset_dir = config.get("data", {}).get("dataset_dir", "datasets")

        kfold_dataset_name = f"{dataset_name}_kfold"
        kfold_check_path = os.path.join(PROJECT_ROOT, dataset_dir, kfold_dataset_name)

        if os.path.exists(kfold_check_path):
            _, val_df, test_df = load_kfold_dataset(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                fold=1,
                k=10,
            )
            logger.info(
                f"Loaded K-fold data from '{kfold_dataset_name}': val={len(val_df)}, test={len(test_df)} samples"
            )
        else:
            _, val_df, test_df = load_dataset(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                train_ratio=config.get("data", {}).get("train_ratio", 0.8),
                val_ratio=config.get("data", {}).get("val_ratio", 0.1),
                test_ratio=config.get("data", {}).get("test_ratio", 0.1),
                random_seed=config.get("data", {}).get("random_seed", 42),
            )
            logger.info(
                f"Loaded data from dataset '{dataset_name}': val={len(val_df)}, test={len(test_df)} samples"
            )
    else:
        # 默认方式 - 从 datasets 目录加载
        dataset_name = config.get("data", {}).get("dataset_name", "financial_sentiment")
        dataset_dir = config.get("data", {}).get("dataset_dir", "datasets")

        kfold_dataset_name = f"{dataset_name}_kfold"
        kfold_check_path = os.path.join(PROJECT_ROOT, dataset_dir, kfold_dataset_name)

        if os.path.exists(kfold_check_path):
            _, val_df, test_df = load_kfold_dataset(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                fold=1,
                k=10,
            )
            logger.info(
                f"Loaded K-fold data from '{kfold_dataset_name}': val={len(val_df)}, test={len(test_df)} samples"
            )
        else:
            _, val_df, test_df = load_dataset(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                train_ratio=config.get("data", {}).get("train_ratio", 0.8),
                val_ratio=config.get("data", {}).get("val_ratio", 0.1),
                test_ratio=config.get("data", {}).get("test_ratio", 0.1),
                random_seed=config.get("data", {}).get("random_seed", 42),
            )
            logger.info(
                f"Loaded data from dataset '{dataset_name}': val={len(val_df)}, test={len(test_df)} samples"
            )

    logger.info(f"Loaded test samples: {len(test_df)}")
    if val_df is not None:
        logger.info(f"Loaded validation samples: {len(val_df)}")

    # 设置模型缓存目录
    model_cache_dir = os.path.join(PROJECT_ROOT, "models/pretrained")

    # 加载 tokenizer（使用与训练时相同的）
    logger.info("Loading tokenizer...")
    # 根据测试数据的语言选择 tokenizer
    if test_df["language"].iloc[0] == "zh":
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["chinese"]["bert_name"],
            cache_dir=model_cache_dir,
            local_files_only=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["english"]["bert_name"],
            cache_dir=model_cache_dir,
            local_files_only=True,
        )

    # 创建数据集和数据加载器
    test_dataset = SentimentDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 评估模型
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device)

    # 生成报告
    label_names = ["negative", "neutral", "positive"]
    metrics = generate_report(y_true, y_pred, label_names)

    # ========== 如果有验证集，也评估验证集 ==========
    val_metrics = None
    val_pred = None
    val_true = None
    val_probs = None
    
    if val_df is not None and len(val_df) > 0:
        logger.info(f"\nEvaluating on validation set ({len(val_df)} samples)...")
        
        val_dataset = SentimentDataset(
            texts=val_df["text"].tolist(),
            labels=val_df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        
        val_pred, val_true, val_probs = evaluate_model(model, val_loader, device)
        val_metrics = generate_report(val_true, val_pred, label_names)
        logger.info("Validation evaluation completed!")
        
        # ========== 保存验证集结果到 results/val/<timestamp>/ ==========
        val_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_val_dir = os.path.join("results", "val", val_timestamp)
        os.makedirs(results_val_dir, exist_ok=True)
        logger.info(f"Validation results will be saved to: {results_val_dir}")
        
        # 1. 保存验证集 metrics.csv
        val_metrics_dict = {
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "train_best_f1": train_best_f1,
            "train_epoch": train_epoch,
            "val_set_size": len(val_df),
        }
        
        for label in label_names:
            val_metrics_dict[f"{label}_precision"] = val_metrics["report_dict"][label]["precision"]
            val_metrics_dict[f"{label}_recall"] = val_metrics["report_dict"][label]["recall"]
            val_metrics_dict[f"{label}_f1"] = val_metrics["report_dict"][label]["f1-score"]
            val_metrics_dict[f"{label}_support"] = val_metrics["report_dict"][label]["support"]
        
        val_metrics_df = pd.DataFrame([val_metrics_dict])
        val_metrics_csv_path = os.path.join(results_val_dir, "metrics.csv")
        val_metrics_df.to_csv(val_metrics_csv_path, index=False)
        logger.info(f"Saved validation metrics to {val_metrics_csv_path}")
        
        # 2. 保存验证集 markdown report
        val_report_path = os.path.join(results_val_dir, "report.md")
        with open(val_report_path, "w", encoding="utf-8") as f:
            f.write("# Validation Set Evaluation Report\n\n")
            f.write(f"## Overall Metrics\n\n")
            f.write(f"- **Accuracy**: {val_metrics['accuracy']:.4f}\n")
            f.write(f"- **Macro F1**: {val_metrics['macro_f1']:.4f}\n")
            f.write(f"- **Weighted F1**: {val_metrics['weighted_f1']:.4f}\n")
            f.write(
                f"- **Training Best F1**: {train_best_f1:.4f} (epoch {train_epoch})\n\n"
            )

            f.write(f"## Per-Class Metrics\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|-------|-----------|--------|----------|--------|\n")
            for label in label_names:
                f.write(
                    f"| {label} | {val_metrics['report_dict'][label]['precision']:.4f} | "
                    f"{val_metrics['report_dict'][label]['recall']:.4f} | {val_metrics['report_dict'][label]['f1-score']:.4f} | "
                    f"{val_metrics['report_dict'][label]['support']} |\n"
                )
        logger.info(f"Saved validation report to {val_report_path}")
        
        # 3. 保存验证集混淆矩阵图
        val_cm = confusion_matrix(val_true, val_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            val_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
        )
        plt.title("Validation Set - Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        val_cm_path = os.path.join(results_val_dir, "confusion_matrix.png")
        plt.savefig(val_cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved validation confusion matrix to {val_cm_path}")
        
        # 4. 保存验证集 per-class metrics 柱状图
        val_precision = [val_metrics["report_dict"][label]["precision"] for label in label_names]
        val_recall = [val_metrics["report_dict"][label]["recall"] for label in label_names]
        val_f1 = [val_metrics["report_dict"][label]["f1-score"] for label in label_names]

        x = np.arange(len(label_names))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x - width, val_precision, width, label="Precision", color="steelblue")
        plt.bar(x, val_recall, width, label="Recall", color="coral")
        plt.bar(x + width, val_f1, width, label="F1-Score", color="seagreen")

        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.title("Validation Set - Per-Class Metrics (Precision, Recall, F1-Score)")
        plt.xticks(x, label_names)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        val_per_class_path = os.path.join(results_val_dir, "per_class_metrics.png")
        plt.savefig(val_per_class_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved validation per-class metrics to {val_per_class_path}")
        
        # 5. 保存验证集详细报告
        val_results_file = os.path.join(results_val_dir, "evaluation_results.md")
        with open(val_results_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("VALIDATION SET EVALUATION RESULTS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Training Epoch: {train_epoch}\n")
            f.write(f"Training Best Val F1: {train_best_f1:.4f}\n")
            f.write(f"Validation Set Size: {len(val_df)}\n")
            f.write(f"Device: {device}\n\n")

            val_report_str = classification_report(
                val_true, val_pred, target_names=label_names, digits=4
            )
            f.write(val_report_str)
            
        logger.info(f"Saved validation results to {val_results_file}")

    # ========== 保存结果到 results/evaluation/<timestamp>/ ==========

    # 1. 保存 metrics.csv
    metrics_dict = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "train_best_f1": train_best_f1,
        "train_epoch": train_epoch,
        "test_set_size": len(test_df),
    }
    
    # 添加验证集指标（如果存在）
    if val_metrics is not None:
        metrics_dict["val_accuracy"] = val_metrics["accuracy"]
        metrics_dict["val_macro_f1"] = val_metrics["macro_f1"]
        metrics_dict["val_weighted_f1"] = val_metrics["weighted_f1"]
        metrics_dict["val_set_size"] = len(val_df)
    
    for label in label_names:
        metrics_dict[f"{label}_precision"] = metrics["report_dict"][label]["precision"]
        metrics_dict[f"{label}_recall"] = metrics["report_dict"][label]["recall"]
        metrics_dict[f"{label}_f1"] = metrics["report_dict"][label]["f1-score"]
        metrics_dict[f"{label}_support"] = metrics["report_dict"][label]["support"]
        
        # 添加验证集的各类别指标
        if val_metrics is not None:
            metrics_dict[f"val_{label}_precision"] = val_metrics["report_dict"][label]["precision"]
            metrics_dict[f"val_{label}_recall"] = val_metrics["report_dict"][label]["recall"]
            metrics_dict[f"val_{label}_f1"] = val_metrics["report_dict"][label]["f1-score"]
            metrics_dict[f"val_{label}_support"] = val_metrics["report_dict"][label]["support"]

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_csv_path = os.path.join(results_eval_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved metrics to {metrics_csv_path}")

    # 2. 保存 markdown report
    report_path = os.path.join(results_eval_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        
        # 测试集结果
        f.write(f"## Test Set Results\n\n")
        f.write(f"### Overall Metrics\n\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
        f.write(f"- **Macro F1**: {metrics['macro_f1']:.4f}\n")
        f.write(f"- **Weighted F1**: {metrics['weighted_f1']:.4f}\n")
        f.write(
            f"- **Training Best F1**: {train_best_f1:.4f} (epoch {train_epoch})\n\n"
        )

        f.write(f"### Per-Class Metrics\n\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|-------|-----------|--------|----------|--------|\n")
        for label in label_names:
            f.write(
                f"| {label} | {metrics['report_dict'][label]['precision']:.4f} | "
                f"{metrics['report_dict'][label]['recall']:.4f} | {metrics['report_dict'][label]['f1-score']:.4f} | "
                f"{metrics['report_dict'][label]['support']} |\n"
            )
        
        # 验证集结果（如果存在）
        if val_metrics is not None:
            f.write(f"\n---\n\n")
            f.write(f"## Validation Set Results\n\n")
            f.write(f"### Overall Metrics\n\n")
            f.write(f"- **Accuracy**: {val_metrics['accuracy']:.4f}\n")
            f.write(f"- **Macro F1**: {val_metrics['macro_f1']:.4f}\n")
            f.write(f"- **Weighted F1**: {val_metrics['weighted_f1']:.4f}\n\n")

            f.write(f"### Per-Class Metrics\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|-------|-----------|--------|----------|--------|\n")
            for label in label_names:
                f.write(
                    f"| {label} | {val_metrics['report_dict'][label]['precision']:.4f} | "
                    f"{val_metrics['report_dict'][label]['recall']:.4f} | {val_metrics['report_dict'][label]['f1-score']:.4f} | "
                    f"{val_metrics['report_dict'][label]['support']} |\n"
                )
    
    logger.info(f"Saved markdown report to {report_path}")

    # 3. 保存混淆矩阵图
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(results_eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to {cm_path}")

    # 4. 保存 per-class metrics 柱状图
    precision = [metrics["report_dict"][label]["precision"] for label in label_names]
    recall = [metrics["report_dict"][label]["recall"] for label in label_names]
    f1 = [metrics["report_dict"][label]["f1-score"] for label in label_names]

    x = np.arange(len(label_names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label="Precision", color="steelblue")
    plt.bar(x, recall, width, label="Recall", color="coral")
    plt.bar(x + width, f1, width, label="F1-Score", color="seagreen")

    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Per-Class Metrics (Precision, Recall, F1-Score)")
    plt.xticks(x, label_names)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    per_class_path = os.path.join(results_eval_dir, "per_class_metrics.png")
    plt.savefig(per_class_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved per-class metrics to {per_class_path}")

    # 保留旧的 txt 格式以兼容
    results_file = os.path.join(results_eval_dir, "evaluation_results.md")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Training Epoch: {train_epoch}\n")
        f.write(f"Training Best Val F1: {train_best_f1:.4f}\n\n")
        
        # 测试集结果
        f.write("-" * 70 + "\n")
        f.write("TEST SET RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Test Set Size: {len(test_df)}\n\n")
        report_str = classification_report(
            y_true, y_pred, target_names=label_names, digits=4
        )
        f.write(report_str)
        
        # 验证集结果（如果存在）
        if val_metrics is not None:
            f.write("\n" + "-" * 70 + "\n")
            f.write("VALIDATION SET RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Validation Set Size: {len(val_df)}\n\n")
            val_report_str = classification_report(
                val_true, val_pred, target_names=label_names, digits=4
            )
            f.write(val_report_str)
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Device: {device}\n")

    logger.info(f"Saved evaluation results to {results_file}")

    # 绘制图表
    if args.plot:
        try:
            plot_file = os.path.join(args.output_dir, "evaluation_plots.png")
            plot_results(y_true, y_pred, y_probs, label_names, save_path=plot_file)
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")

    # 打印最终总结
    print("\n" + "=" * 70)
    print(" " * 25 + "SUMMARY")
    print("=" * 70)
    print(
        f"✅ Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)"
    )
    print(f"✅ Test Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"✅ Test Weighted F1:    {metrics['weighted_f1']:.4f}")
    
    if val_metrics is not None:
        print(f"\n✅ Val Accuracy:   {val_metrics['accuracy']:.4f} ({val_metrics['accuracy'] * 100:.2f}%)")
        print(f"✅ Val Macro F1:        {val_metrics['macro_f1']:.4f}")
        print(f"✅ Val Weighted F1:     {val_metrics['weighted_f1']:.4f}")
    
    print(f"✅ Training F1:    {train_best_f1:.4f} (from epoch {train_epoch})")
    print("=" * 70)

    # 检查性能下降
    if train_best_f1 > 0 and metrics["macro_f1"] < train_best_f1 * 0.9:
        print("\n⚠️  WARNING: Test F1 is significantly lower than training F1!")
        print(f"   This might indicate overfitting.")

    print("\n💡 Detailed results saved to:")
    print(f"   Test Set:")
    print(f"   - {results_file}")
    print(f"   - {metrics_csv_path}")
    print(f"   - {report_path}")
    print(f"   - {cm_path}")
    print(f"   - {per_class_path}")
    
    if val_metrics is not None:
        print(f"\n   Validation Set:")
        print(f"   - {val_results_file}")
        print(f"   - {val_metrics_csv_path}")
        print(f"   - {val_report_path}")
        print(f"   - {val_cm_path}")
        print(f"   - {val_per_class_path}")
    
    if args.plot:
        print(f"   - {os.path.join(args.output_dir, 'evaluation_plots.png')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
