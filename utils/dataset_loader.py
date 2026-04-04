"""数据集加载工具"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from pathlib import Path


def get_dataset_path(dataset_name: str, dataset_dir: str = "datasets") -> str:
    """获取数据集路径"""
    project_root = Path(__file__).parent.parent
    return os.path.join(project_root, dataset_dir, dataset_name)


def load_dataset(
    dataset_name: str,
    dataset_dir: str = "datasets",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载数据集

    优先检查是否有已划分的 train.csv/val.csv/test.csv，
    如果没有则从 data.csv 自动划分。

    Args:
        dataset_name: 数据集名称 (datasets/ 下的子目录名)
        dataset_dir: 数据集根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        (train_df, val_df, test_df)
    """
    dataset_path = get_dataset_path(dataset_name, dataset_dir)

    # 检查是否存在已划分的文件
    train_file = os.path.join(dataset_path, "train.csv")
    val_file = os.path.join(dataset_path, "val.csv")
    test_file = os.path.join(dataset_path, "test.csv")

    if (
        os.path.exists(train_file)
        and os.path.exists(val_file)
        and os.path.exists(test_file)
    ):
        # 使用已划分的文件
        return (pd.read_csv(train_file), pd.read_csv(val_file), pd.read_csv(test_file))

    # 检查是否存在原始数据文件
    data_file = os.path.join(dataset_path, "data.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # 自动划分数据
    df = pd.read_csv(data_file)
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio_adj = train_ratio / total_ratio
    val_ratio_adj = val_ratio / total_ratio

    # 划分训练集和临时集
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio_adj, random_state=random_seed
    )
    # 划分验证集和测试集
    val_ratio_from_temp = val_ratio_adj / (val_ratio_adj + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio_from_temp, random_state=random_seed
    )

    return train_df, val_df, test_df


def get_available_datasets(dataset_dir: str = "datasets") -> list:
    """获取可用的数据集列表"""
    project_root = Path(__file__).parent.parent
    datasets_path = os.path.join(project_root, dataset_dir)

    if not os.path.exists(datasets_path):
        return []

    return [
        d
        for d in os.listdir(datasets_path)
        if os.path.isdir(os.path.join(datasets_path, d))
    ]


def load_kfold_dataset(
    dataset_name: str,
    dataset_dir: str = "datasets",
    fold: int = 1,
    k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a specific fold from K-fold dataset.

    Args:
        dataset_name: Base dataset name (e.g., "financial_sentiment")
        dataset_dir: Dataset root directory
        fold: Fold number (1-based)
        k: Total number of folds

    Returns:
        (train_df, val_df, test_df)
    """
    project_root = Path(__file__).parent.parent
    kfold_dir = os.path.join(project_root, dataset_dir, f"{dataset_name}_kfold")

    if not os.path.exists(kfold_dir):
        raise FileNotFoundError(f"K-fold dataset not found: {kfold_dir}")

    test_file = os.path.join(kfold_dir, "test.csv")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"test.csv not found in {kfold_dir}")
    test_df = pd.read_csv(test_file)

    fold_dir = os.path.join(kfold_dir, f"fold_{fold}")
    if not os.path.exists(fold_dir):
        raise FileNotFoundError(f"Fold {fold} not found: {fold_dir}")

    train_file = os.path.join(fold_dir, "train.csv")
    val_file = os.path.join(fold_dir, "val.csv")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(f"Train/val files not found in {fold_dir}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    return train_df, val_df, test_df
